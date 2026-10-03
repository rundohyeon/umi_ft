#!/usr/bin/env python
"""Stage A only; Stage B uses the repository's existing Hydra workspace."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from omegaconf import OmegaConf
from diffusion_policy.context.data import load_base, split_episodes, ContextFrames, validate_label_alignment, source_fingerprint
from diffusion_policy.context.labels import definitions, read_rows, LabelIndex, atomic_json
from diffusion_policy.context.evaluation import write_evaluation, classification_metrics
from models.context_encoder import ContextEncoder, context_loss, benchmark_encoder


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); labels=[]; probabilities=[]; episodes=[]; timestamps=[]
    for batch in loader:
        logits=model({k:v.to(device) for k,v in batch['obs'].items()})
        probabilities.extend(logits.softmax(-1).cpu().tolist()); labels.extend(batch['context_label'].tolist())
        episodes.extend(batch['episode_id'].tolist()); timestamps.extend(batch['timestamp'].tolist())
    return labels,probabilities,episodes,timestamps


def train(cfg):
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    seed=cfg['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    definitions_cfg,digest=definitions(cfg['context_definitions'])
    num_classes=len(definitions_cfg['contexts'])
    base,policy_cfg=load_base(cfg['policy_config'],cfg.get('dataset_overrides',[]))
    auto=read_rows(cfg['labels']['auto_path'],num_classes); reviewed=read_rows(cfg['labels']['reviewed_path'],num_classes)
    validate_label_alignment(base,auto+reviewed)
    from diffusion_policy.context.labels import validate_definition_version
    validate_definition_version(auto+reviewed,digest)
    labels=LabelIndex(auto,reviewed,num_classes=num_classes,**{k:v for k,v in cfg['labels'].items() if not k.endswith('_path')})
    splits=split_episodes(len(base.rgb_episode_ends),seed,cfg['val_ratio'],cfg['test_ratio'])
    split_record=dict(splits=splits,source_fingerprint=source_fingerprint(base),context_definition_hash=digest,num_classes=num_classes)
    split_path=Path(cfg['split_path'])
    if split_path.exists() and json.loads(split_path.read_text()) != split_record:
        raise ValueError('Existing split differs; use a separate experiment split_path')
    atomic_json(split_path,split_record)
    datasets={key:ContextFrames(base,eps,labels) for key,eps in splits.items()}
    for key,ds in datasets.items():
        if not len(ds): raise ValueError(f'No usable {cfg["labels"]["source"]} labels in {key} episodes; review that split first')
    loaders={k:DataLoader(ds,batch_size=cfg['batch_size'],shuffle=k=='train',num_workers=cfg['num_workers']) for k,ds in datasets.items()}
    shape_meta=OmegaConf.to_container(policy_cfg.shape_meta,resolve=True)
    encoder_config=dict(cfg['encoder'])
    if encoder_config.get('num_classes',num_classes)!=num_classes:
        raise ValueError('Encoder num_classes differs from context definitions')
    encoder_config['num_classes']=num_classes
    model=ContextEncoder(shape_meta,**encoder_config)
    model.fit_statistics(loaders['train'],cfg.get('statistics_max_batches'))
    device=cfg['device']; model.to(device)
    distribution=np.bincount([e[1] for e in datasets['train'].entries],minlength=num_classes)
    weights=None
    if cfg['class_balancing']:
        weights=torch.tensor(np.divide(distribution.sum(),num_classes*distribution,out=np.zeros(num_classes,dtype=float),where=distribution>0),dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    checkpoint=Path(cfg['checkpoint']); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    best=-1.; history=[]
    for epoch in range(cfg['epochs']):
        model.train(); losses=[]
        for i,batch in enumerate(loaders['train']):
            if cfg.get('max_batches') is not None and i>=cfg['max_batches']: break
            logits=model({k:v.to(device) for k,v in batch['obs'].items()})
            loss=context_loss(logits,batch['context_label'].to(device),batch['context_weight'].to(device),weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        result=predict(model,loaders['validation'],device)
        metrics=classification_metrics(result[0],result[1])
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_macro_f1=metrics['macro_f1']))
        print(json.dumps(history[-1]),flush=True)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1']
            payload=dict(state_dict=model.state_dict(),encoder_config=model.config,context_definition_hash=digest,
                definitions=definitions_cfg['contexts'],training_config=cfg,episode_splits=splits,
                source_fingerprint=split_record['source_fingerprint'],train_distribution=distribution.tolist(),
                validation_metrics=metrics,epoch=epoch)
            temporary=checkpoint.with_suffix('.tmp'); torch.save(payload,temporary); temporary.replace(checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict'])
    result=predict(model,loaders['test'],device)
    sample=next(iter(loaders['test']))['obs']; sample={k:v[:1] for k,v in sample.items()}
    benchmark=benchmark_encoder(model,sample)
    metrics=write_evaluation(cfg['output_dir'],*result,names=[v['name'] for _,v in sorted(definitions_cfg['contexts'].items())],
        extra=dict(benchmark=benchmark,train_distribution=distribution.tolist(),split='test',label_source=cfg['labels']['source'],best_validation_macro_f1=best))
    atomic_json(Path(cfg['output_dir'])/'training_history.json',history)
    base.close()
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='context/qwen/config/train_context.yaml')
    parser.add_argument('--set',action='append',default=[],help='OmegaConf dotlist override, e.g. epochs=1')
    args=parser.parse_args()
    cfg=OmegaConf.merge(OmegaConf.load(args.config),OmegaConf.from_dotlist(args.set))
    train(OmegaConf.to_container(cfg,resolve=True))

if __name__=='__main__': main()
