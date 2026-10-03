#!/usr/bin/env python
"""Train the RGB/native-F/T context classifier; torchrun provides multi-GPU DDP."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_policy.context.canonical_dataset import (
    CanonicalRGBForceDataset, PHASE_NAMES, canonical_episode_splits)
from diffusion_policy.context.evaluation import classification_metrics, write_evaluation
from diffusion_policy.context.labels import atomic_json
from diffusion_policy.context.ft_features import ft_feature_contract
from models.rgb_force_context_encoder import RGBForceContextEncoder


def autocast_context(device, enabled):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled and device.type == 'cuda' else nullcontext()


def move_obs(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch['obs'].items()}


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=None):
    model.eval()
    results = {k: [] for k in ('labels', 'probabilities', 'episodes', 'timestamps', 'sources')}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with autocast_context(device, amp):
            logits = model(move_obs(batch, device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        results['probabilities'].extend(logits.float().softmax(-1).cpu().tolist())
        for name, key in [('labels', 'context_label'), ('episodes', 'episode_id'),
                          ('timestamps', 'timestamp'), ('sources', 'label_source')]:
            results[name].extend(batch[key].tolist())
    metrics = classification_metrics(results['labels'], results['probabilities'])
    source = np.asarray(results['sources'])
    manual = source > 0
    metrics['manual_only'] = (classification_metrics(np.asarray(results['labels'])[manual],
        np.asarray(results['probabilities'])[manual]) if manual.any() else None)
    return metrics, results


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def run(cfg, resume=None, smoke_batches=None, dry_run=False):
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1 or cfg['num_workers'] < 0:
        raise ValueError('Invalid epoch/batch/worker settings')
    if smoke_batches is not None and smoke_batches < 1:
        raise ValueError('--smoke-batches must be positive')
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world > 1
    if cfg['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; activate the umi environment and check GPU access')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if distributed:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=30))
    try:
        train(cfg, device, rank, world, resume, smoke_batches, dry_run)
    finally:
        if distributed:
            dist.destroy_process_group()


def train(cfg, device, rank, world, resume, smoke_batches, dry_run):
    distributed = world > 1
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    model_force_contract = ft_feature_contract(cfg['model'].get('ft_feature_mode', 'raw'),
        cfg['model'].get('ft_mean_window', 5), cfg['model'].get('ft_delta_lag', 5))
    data_force_contract = ft_feature_contract(cfg['dataset'].get('ft_feature_mode', 'raw'),
        cfg['dataset'].get('ft_mean_window', 5), cfg['dataset'].get('ft_delta_lag', 5))
    if (data_force_contract != model_force_contract
            or cfg['dataset']['ft_history'] != cfg['model']['ft_horizon']):
        raise ValueError('Dataset and model F/T feature/history contracts differ')
    base = CanonicalRGBForceDataset(**cfg['dataset'])
    splits = canonical_episode_splits(len(base.rgb_ends), cfg['seed'], cfg['val_ratio'], cfg['test_episode_numbers'])
    datasets = {k: base.subset(v) for k, v in splits.items()}
    counts = {k: np.bincount(base.targets[ds.indices], minlength=4) for k, ds in datasets.items()}
    for split, values in counts.items():
        if np.any(values == 0):
            raise ValueError(f'Every class must occur in {split}: {values.tolist()}')
    output = Path(cfg['output_dir'])
    manifest = dict(schema='context_rgb_force_4state_v1', source_fingerprint=base.fingerprint,
                    phase_names=list(PHASE_NAMES), episode_splits=splits)
    if model_force_contract['mode'] != 'raw':
        manifest.update(schema='context_rgb_force_4state_ft_features_v2',
                        force_features=model_force_contract)
    report = dict(**manifest, physical_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        world_size=world, batch_size_per_gpu=cfg['batch_size'], global_batch_size=cfg['batch_size'] * world,
        input_keys=list(RGBForceContextEncoder.input_keys), labelled_frames=int(base.valid.sum()),
        usable_frames=int(base.usable.sum()), excluded_force_alignment=int((base.valid & ~base.usable).sum()),
        class_counts={k: v.tolist() for k, v in counts.items()},
        label_source_names=base.source_names, rule_label_weight=cfg['dataset']['rule_label_weight'],
        manual_label_weight=cfg['dataset']['manual_label_weight'],
        force_raw_history=base.required_ft_history, force_cnn_steps=base.ft_history,
        force_channels_per_finger=model_force_contract['channels_per_finger'])
    payload = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if payload:
        if Path(resume).resolve().parent != output.resolve():
            raise ValueError('Resume from a checkpoint in the same output_dir to preserve the best model')
        if bool(payload.get('smoke_test')) != (smoke_batches is not None):
            raise ValueError('Smoke-test checkpoints must not be used to resume a full training run')
        expected_model_config = dict(cfg['model'])
        if model_force_contract['mode'] == 'raw':
            for key in ('ft_feature_mode', 'ft_mean_window', 'ft_delta_lag'):
                expected_model_config.pop(key, None)
        if payload['manifest'] != manifest or payload['model_config'] != expected_model_config:
            raise ValueError('Checkpoint model/data/splits do not match this run')
        for key in ('dataset', 'seed', 'val_ratio', 'test_episode_numbers'):
            if payload['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume changed the training contract: {key}')
    if rank == 0:
        if output.exists() and any(output.iterdir()) and resume is None:
            raise FileExistsError(f'{output} is not empty; use --resume or a new output_dir')
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'episode_splits.json').exists():
            if json.loads((output / 'episode_splits.json').read_text()) != manifest:
                raise ValueError('Existing experiment has different data, labels, or episode splits')
        atomic_json(output / 'episode_splits.json', manifest)
        atomic_json(output / 'dataset_report.json', report)
        OmegaConf.save(OmegaConf.create(cfg), output / 'resolved_config.yaml')
        print(json.dumps(report), flush=True)
    if distributed:
        dist.barrier()
    if dry_run:
        return
    if payload:
        model = RGBForceContextEncoder.from_checkpoint(payload)
    else:
        # Must succeed with actual pretrained CLIP weights, never a random fallback.
        model = RGBForceContextEncoder(**cfg['model'], pretrained=True)
        model.set_force_statistics(*base.force_statistics(splits['train']))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    first_epoch, best = 0, -1.0
    if payload:
        optimizer.load_state_dict(payload['optimizer'])
        first_epoch, best = int(payload['epoch']) + 1, float(payload['best_validation_macro_f1'])
    if first_epoch >= cfg['epochs']:
        raise ValueError('Checkpoint already reached configured epochs; increase epochs to resume')
    network = (DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False) if distributed else model)
    # Rank-specific dropout streams after DDP has synchronized initial weights.
    torch.manual_seed(cfg['seed'] + rank)
    sampler = DistributedSampler(datasets['train'], num_replicas=world, rank=rank,
        shuffle=True, seed=cfg['seed']) if distributed else None
    loader_kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                     pin_memory=device.type == 'cuda', persistent_workers=cfg['num_workers'] > 0)
    train_loader = DataLoader(datasets['train'], shuffle=sampler is None, sampler=sampler, **loader_kw)
    # Evaluate the full held-out sets once, without DistributedSampler padding.
    val_loader = DataLoader(datasets['validation'], shuffle=False, **loader_kw) if rank == 0 else None
    test_loader = DataLoader(datasets['test'], shuffle=False, **loader_kw) if rank == 0 else None
    class_weights = torch.tensor(counts['train'].sum() / (4 * counts['train']),
                                 dtype=torch.float32, device=device)
    amp = bool(cfg['amp']) and device.type == 'cuda'
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bfloat16; use --set amp=false')
    if rank == 0:
        print(json.dumps(dict(event='model_ready', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_vision_parameters=sum(p.numel() for p in model.vision.parameters()), device=str(device),
            smoke_batches=smoke_batches)), flush=True)
    epochs = range(first_epoch, min(cfg['epochs'], first_epoch + 1) if smoke_batches else cfg['epochs'])
    for epoch in epochs:
        network.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total = torch.zeros(2, device=device, dtype=torch.float64)
        for step, batch in enumerate(train_loader):
            if smoke_batches is not None and step >= smoke_batches:
                break
            target = batch['context_label'].to(device, non_blocking=True)
            weight = batch['context_weight'].to(device, non_blocking=True) * class_weights[target]
            denominator = weight.sum()
            if distributed:
                dist.all_reduce(denominator)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits = network(move_obs(batch, device))
                numerator = (F.cross_entropy(logits.float(), target, reduction='none') * weight).sum()
                # DDP averages gradients; this yields the global weighted mean.
                loss = numerator * world / denominator.clamp_min(1e-8)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}, rank {rank}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                cfg['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            total += torch.stack([numerator.detach().double(), weight.sum().double()])
            if rank == 0 and (step == 0 or (step + 1) % cfg['log_every'] == 0):
                print(json.dumps(dict(epoch=epoch + 1, step=step + 1, steps=len(train_loader),
                    rank0_loss=float((numerator.detach() / weight.sum()).cpu()))), flush=True)
        if distributed:
            dist.all_reduce(total)
        if rank == 0:
            metrics, _ = evaluate(model, val_loader, device, amp, smoke_batches)
            improved = metrics['macro_f1'] > best
            best = max(best, metrics['macro_f1'])
            record = dict(epoch=epoch + 1, train_loss=float((total[0] / total[1]).cpu()),
                          validation=metrics, smoke_test=smoke_batches is not None)
            with (output / 'training_history.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            payload = dict(schema=manifest['schema'], state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                model_config=model.config, training_config=cfg, manifest=manifest, epoch=epoch,
                optimizer=optimizer.state_dict(), best_validation_macro_f1=best,
                validation_metrics=metrics, smoke_test=smoke_batches is not None)
            save_checkpoint(output / 'context_encoder_last.pt', payload)
            if improved:
                save_checkpoint(output / 'context_encoder_best.pt', payload)
        if distributed:
            dist.barrier()
    if rank == 0:
        saved = torch.load(output / 'context_encoder_best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        metrics, result = evaluate(model, test_loader, device, amp, smoke_batches)
        write_evaluation(output / ('smoke_eval' if smoke_batches else 'test_eval'), result['labels'],
            result['probabilities'], result['episodes'], result['timestamps'], names=list(PHASE_NAMES),
            extra=dict(manual_only=metrics['manual_only'], checkpoint_selected_by='validation_macro_f1',
                       smoke_test=smoke_batches is not None))
        print(json.dumps(dict(event='complete', checkpoint=str(output / 'context_encoder_best.pt'),
                              test_macro_f1=metrics['macro_f1'], smoke_test=smoke_batches is not None)), flush=True)
    if distributed:
        dist.barrier()
    for dataset in [base, *datasets.values()]:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='context/qwen/config/train_context_rgb_force.yaml')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--resume')
    parser.add_argument('--smoke-batches', type=int, help='One short epoch and truncated evaluation in a separate output_dir')
    parser.add_argument('--dry-run', action='store_true', help='Check data, label alignment, and splits without constructing a model')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set)), resolve=True)
    if args.smoke_batches and cfg['output_dir'] in (
            'outputs/context_rgb_force_4state', 'outputs/context_rgb_force_4state_ft_features'):
        raise ValueError('Set a separate output_dir for the smoke test')
    run(cfg, args.resume, args.smoke_batches, args.dry_run)


if __name__ == '__main__':
    main()
