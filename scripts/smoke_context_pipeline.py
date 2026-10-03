#!/usr/bin/env python
"""Small synthetic mechanics check. Its labels are test fixtures, not human annotations."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
from diffusion_policy.context.data import load_base, episode_bounds
from diffusion_policy.context.labels import definitions, LabelIndex, review_segment, save_review, atomic_json
from scripts.generate_context_labels import generate
from train_context_encoder import train


def make_fixture(directory, episodes=6, frames=64):
    import zarr
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'synthetic.zarr'
    if path.exists(): raise FileExistsError(path)
    root=zarr.open_group(str(path),mode='w')
    data=root.create_group('data');meta=root.create_group('meta')
    n=episodes*frames; nf=int(np.ceil(frames/59.94*100))+1
    rgb_t=np.concatenate([10*e+np.arange(frames)/59.94 for e in range(episodes)])
    ft_t=np.concatenate([10*e+np.arange(nf)/100 for e in range(episodes)])
    data.create_dataset('rgb_0',shape=(n,224,224,3),dtype='u1',chunks=(1,224,224,3),fill_value=96)
    for key in ['rgb_time_stamps_0','robot_time_stamps_0','gripper_time_stamps_0']:data.create_dataset(key,data=rgb_t)
    pose=np.zeros((n,7),dtype=np.float32);pose[:,6]=1;pose[:,0]=np.tile(np.arange(frames)*.0001,episodes)
    data.create_dataset('ts_pose_fb_0',data=pose)
    data.create_dataset('gripper_0',data=np.full((n,1),.04,dtype=np.float32))
    data.create_dataset('grasp_force_0',data=np.zeros((n,1),dtype=np.float32))
    data.create_dataset('wrench_time_stamps_0',data=ft_t)
    for key in ['wrench_left_0','wrench_right_0']:data.create_dataset(key,data=np.zeros((episodes*nf,6),dtype=np.float32))
    for key in ['episode_rgb0_len','episode_robot0_len','episode_gripper0_len']:meta.create_dataset(key,data=np.arange(1,episodes+1)*frames)
    meta.create_dataset('episode_wrench0_len',data=np.arange(1,episodes+1)*nf)
    return ['task=umi_dual_ft',f'task.dataset_path={path}']


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    overrides=make_fixture(output)
    base,policy_cfg=load_base(overrides=overrides)
    cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    num_classes=len(cfg['contexts'])
    cfg.update(output_dir=str(output/'labels'),model_path='synthetic-response-fixture',history_length=8,label_stride=2)
    counter=[0]
    def fake_labeler(prompt, images=None):
        counter[0]+=1
        return json.dumps(dict(class_id=(counter[0]//3)%num_classes,confidence=.95,reason='Synthetic test response'))
    automatic=generate(base,cfg,digest,fake_labeler,end_index=12)
    before=counter[0]
    generate(base,cfg,digest,fake_labeler,end_index=12,resume=True)
    assert counter[0]==before
    reviewed=[];index=LabelIndex(automatic,num_classes=num_classes)
    for ep in range(base.n_episodes):
        lo,hi=episode_bounds(base,ep)
        reviewed=review_segment(reviewed,index,ep,0,11,base.rgb_timestamps[lo:hi])
    save_review(output/'labels/reviewed_labels.parquet',reviewed)
    train_cfg=yaml.safe_load(Path('context/qwen/config/train_context.yaml').read_text())
    train_cfg.update(dataset_overrides=overrides,epochs=1,batch_size=4,max_batches=2,statistics_max_batches=2,
        checkpoint=str(output/'context_encoder_best.pt'),output_dir=str(output/'evaluation'),split_path=str(output/'labels/episode_splits.json'),cpu_threads=2)
    train_cfg['labels'].update(auto_path=str(output/'labels/auto_labels.jsonl'),reviewed_path=str(output/'labels/reviewed_labels.parquet'))
    train(train_cfg)
    # Reuse the existing Hydra workspace with a smaller real encoder/U-Net for CPU smoke.
    from hydra import compose,initialize_config_dir
    with initialize_config_dir(version_base=None,config_dir=str(Path('diffusion_policy/config').resolve())):
        stage_b=compose(config_name='train_context_aware_policy',overrides=overrides)
    stage_b.context_training.encoder_checkpoint=str(output/'context_encoder_best.pt')
    stage_b.task.dataset.context_split_path=str(output/'labels/episode_splits.json')
    stage_b.task.dataset.context_labels.auto_path=str(output/'labels/auto_labels.jsonl')
    stage_b.task.dataset.context_labels.reviewed_path=str(output/'labels/reviewed_labels.parquet')
    stage_b.policy.obs_encoder.pretrained=False
    stage_b.policy.obs_encoder.model_name='resnet18'
    stage_b.policy.obs_encoder.vision_feature_dim=512
    stage_b.policy.obs_encoder.feature_aggregation='avg'
    stage_b.policy.obs_encoder.transforms=None
    stage_b.policy.down_dims=[32,64]
    stage_b.policy.kernel_size=3
    stage_b.policy.num_inference_steps=2
    stage_b.policy.context.curriculum='full'
    stage_b.policy.context.freeze_encoder=False
    stage_b.training.use_ema=False
    stage_b.training.num_epochs=1
    stage_b.training.max_train_steps=1
    stage_b.training.max_val_steps=1
    stage_b.training.lr_warmup_steps=0
    stage_b.training.sample_every=1
    stage_b.training.checkpoint_every=1
    stage_b.logging.mode='disabled'
    stage_b.dataloader.batch_size=2;stage_b.val_dataloader.batch_size=2
    stage_b.dataloader.num_workers=0;stage_b.val_dataloader.num_workers=0
    stage_b.dataloader.persistent_workers=False;stage_b.val_dataloader.persistent_workers=False
    from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
    (output/'policy').mkdir(parents=True,exist_ok=True)
    workspace=TrainDiffusionUnetImageWorkspace(stage_b,output_dir=str(output/'policy'))
    workspace.run()
    dataset=instantiate(stage_b.task.dataset)
    batch=next(iter(DataLoader(dataset,batch_size=2)))
    workspace.model.eval()
    with torch.no_grad():result=workspace.model.predict_action(batch['obs'])
    assert result['action_pred'].shape==(2,16,11)
    assert result['context']['raw_probabilities'].shape==(2,num_classes)
    assert torch.allclose(result['context']['raw_probabilities'].sum(-1),torch.ones(2),atol=1e-6)
    # Strict self-contained restore: external Stage A checkpoint is unnecessary.
    checkpoint=output/'policy/checkpoints/latest.ckpt'
    import dill
    payload=torch.load(checkpoint,map_location='cpu',pickle_module=dill)
    assert 'optimizer' in payload['state_dicts']
    assert not OmegaConf.is_interpolation(payload['cfg'].policy.context,'num_classes')
    assert payload['cfg'].policy.context.num_classes==num_classes
    restored=instantiate(payload['cfg'].policy)
    restored.load_state_dict(payload['state_dicts']['model'],strict=True)
    restored.eval()
    from eval_dual_ft_offline import evaluate_loader
    metrics=evaluate_loader(restored,DataLoader(Subset(dataset,[0,1]),batch_size=2),device=torch.device('cpu'),
        max_batches=1,prediction_repeats=1,seed=42,compute_diffusion_loss=True,
        expected_obs_meta=stage_b.shape_meta.obs,ft_max_age_sec=.012)
    atomic_json(output/'policy_metrics.json',metrics)
    # Execute the normal evaluation CLI function on the Stage A checkpoint.
    from eval_context_aware import evaluate
    evaluate(argparse.Namespace(checkpoint=str(output/'context_encoder_best.pt'),cpu_threads=2,split='test',max_samples=4,
        batch_size=2,device='cpu',output_dir=str(output/'eval_reload'),weights='auto',num_inference_steps=None))
    # Exercise the actual offline transformers loader using a tiny random Qwen fixture.
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    local=output/'tiny_random_qwen';local.mkdir()
    vocabulary={'[UNK]':0,'[EOS]':1,'robot':2,'context':3,'JSON':4}
    tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',eos_token='[EOS]',
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}")
    fast.save_pretrained(local)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2)).save_pretrained(local)
    from scripts.generate_context_labels import LocalLabeler,label_with_retries
    response,_,_=label_with_retries(LocalLabeler(str(local),max_new_tokens=3),'robot context JSON',0)
    assert response['class_id']==-1
    atomic_json(output/'smoke_result.json',dict(status='passed',synthetic_labels=True,
        real_robot_used=False,num_classes=num_classes,stage_a_checkpoint=str(output/'context_encoder_best.pt'),stage_b_checkpoint=str(checkpoint)))
    print(f'Smoke passed: {output}')
    base.close();dataset.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    run(parser.parse_args().output_dir)
