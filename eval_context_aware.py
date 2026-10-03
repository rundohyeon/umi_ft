#!/usr/bin/env python
"""Evaluate Stage A or Stage B, reusing the existing UMI action metrics."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
from diffusion_policy.context.data import load_base, ContextFrames, validate_label_alignment, source_fingerprint
from diffusion_policy.context.labels import LabelIndex, read_rows, atomic_json
from diffusion_policy.context.evaluation import write_evaluation
from models.context_encoder import ContextEncoder, benchmark_encoder
from train_context_encoder import predict


def evaluate(args):
    import dill
    payload=torch.load(args.checkpoint,map_location='cpu',pickle_module=dill)
    torch.set_num_threads(args.cpu_threads)
    if 'encoder_config' in payload:
        cfg=payload['training_config']
        base,_=load_base(cfg['policy_config'],cfg.get('dataset_overrides',[]))
        if source_fingerprint(base)!=payload['source_fingerprint']:
            raise ValueError('Checkpoint evaluation dataset differs from training split manifest')
        label_cfg=cfg['labels']
        num_classes=payload['encoder_config'].get('num_classes',5)
        auto=read_rows(label_cfg['auto_path'],num_classes);reviews=read_rows(label_cfg['reviewed_path'],num_classes)
        validate_label_alignment(base,auto+reviews)
        from diffusion_policy.context.labels import validate_definition_version
        validate_definition_version(auto+reviews,payload['context_definition_hash'])
        labels=LabelIndex(auto,reviews,num_classes=num_classes,**{k:v for k,v in label_cfg.items() if not k.endswith('_path')})
        dataset=ContextFrames(base,payload['episode_splits'][args.split],labels)
        if args.max_samples: dataset=Subset(dataset,range(min(args.max_samples,len(dataset))))
        loader=DataLoader(dataset,batch_size=args.batch_size)
        model=ContextEncoder(**payload['encoder_config']).to(args.device)
        model.load_state_dict(payload['state_dict'],strict=True)
        result=predict(model,loader,args.device)
        batch=next(iter(loader))['obs']
        benchmark=benchmark_encoder(model,{k:v[:1] for k,v in batch.items()})
        return write_evaluation(args.output_dir,*result,
            names=[payload['definitions'][i]['name'] for i in range(num_classes)],
            extra=dict(benchmark=benchmark,split=args.split,label_source=label_cfg['source']))
    from eval_dual_ft_offline import load_policy, evaluate_loader
    from omegaconf import OmegaConf
    cfg=payload['cfg']
    # Retain all normal observation/action contracts; only turn off deployment state.
    if OmegaConf.select(cfg,'policy.context') is not None:
        cfg.policy.context.smoothing_alpha=None
    loaded=load_policy(payload,dataset_path=Path(cfg.task.dataset.dataset_path),
        force_sidecar_path=Path(cfg.task.dataset.force_sidecar_path),context_sidecar_path=None,
        device_spec=args.device,weights=args.weights,num_inference_steps=args.num_inference_steps)
    base=instantiate(loaded.cfg.task.dataset)
    metadata=OmegaConf.select(loaded.cfg,'context_training.metadata',default=None)
    if metadata is not None:
        if (source_fingerprint(base)!=metadata.source_fingerprint
                or base.context_splits!=dict(metadata.episode_splits)):
            raise ValueError('Evaluation source/splits differ from the policy checkpoint')
    if not hasattr(base,'context_splits'):
        raise ValueError('Use eval_dual_ft_offline.py for original policies without context split metadata')
    mask=np.isin(np.arange(len(base.rgb_episode_ends)),base.context_splits[args.split])
    base.indices=base._build_indices(mask);base.split='validation'
    if loaded.policy.context_mode=='oracle':
        base.indices=[entry for entry in base.indices if base.label_index.resolve(entry[0],entry[1]-(0 if entry[0]==0 else int(base.rgb_episode_ends[entry[0]-1])))[0]>=0]
    dataset=Subset(base,range(min(args.max_samples,len(base)))) if args.max_samples else base
    loader=DataLoader(dataset,batch_size=args.batch_size)
    metrics=evaluate_loader(loaded.policy,loader,device=loaded.device,seed=42,max_batches=None,prediction_repeats=1,
        compute_diffusion_loss=True,expected_obs_meta=loaded.cfg.shape_meta.obs,ft_max_age_sec=loaded.cfg.task.ft_max_age_sec)
    rows=metrics.pop('context_records')
    metrics.update(task_success=None,trajectory_completion=None,rollout_success=None,
        rollout_note='Offline demonstration evaluation does not measure robot task outcomes.',split=args.split)
    atomic_json(Path(args.output_dir)/'policy_metrics.json',metrics)
    if rows and any(r['label']>=0 for r in rows):
        write_evaluation(args.output_dir,[r['label'] for r in rows],[r['probabilities'] for r in rows],
            [r['episode'] for r in rows],[r['timestamp'] for r in rows],
            names=None if metadata is None else [metadata.definitions[i]['name'] for i in range(loaded.policy.num_context_classes)],
            extra={'policy_metrics':metrics})
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',default='outputs/context_eval')
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--weights',choices=['auto','model','ema_model'],default='auto')
    p.add_argument('--num-inference-steps',type=int)
    args=p.parse_args();evaluate(args)

if __name__=='__main__':main()
