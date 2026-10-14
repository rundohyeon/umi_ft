#!/usr/bin/env python
"""Train a slip classifier with per-RGB F/T and causal predicted-context history."""
from __future__ import annotations

import argparse
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

from diffusion_policy.context.evaluation import classification_metrics, write_evaluation
from diffusion_policy.context.labels import atomic_json
from diffusion_policy.context.slip_feature_cache import ensure_feature_cache
from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset, split_slip_episodes
from diffusion_policy.context.slip_timing import rgb_window_spec
from models.slip_temporal_context import SlipTemporalContextEncoder, PHASE_NAMES, SCHEMA
from train_context_rgb_force import autocast_context, move_obs, save_checkpoint


def temporal_metrics(labels, probabilities, episodes, timestamps):
    """Measure prediction flips inside stable labels and delay after observed error onsets."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    pred, ep, ts = probabilities.argmax(-1), np.asarray(episodes), np.asarray(timestamps)
    connected = (ep[1:] == ep[:-1]) & (np.diff(ts) < .15)
    stable = connected & (labels[1:] == labels[:-1])
    flips = int((stable & (pred[1:] != pred[:-1])).sum())
    transitions, missed, delays = 0, 0, []
    for i in np.flatnonzero(connected & (labels[1:] != labels[:-1])) + 1:
        if labels[i] not in (1, 2):
            continue
        transitions += 1
        j = i + 1
        while j < len(labels) and connected[j - 1] and labels[j] == labels[i]:
            j += 1
        matches = np.flatnonzero(pred[i:j] == labels[i])
        if len(matches):
            delays.append(float(ts[i + matches[0]] - ts[i]))
        else:
            missed += 1
    return dict(stable_label_prediction_flips=flips, stable_adjacent_pairs=int(stable.sum()),
                error_onsets=transitions, missed_error_segments=missed,
                mean_detected_error_delay_s=float(np.mean(delays)) if delays else None)


@torch.inference_mode()
def evaluate(model, loader, device, amp=False, max_batches=None):
    model.eval()
    result = {k: [] for k in ('labels', 'probabilities', 'observation_probabilities', 'episodes', 'timestamps')}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with autocast_context(device, amp):
            out = model(move_obs(batch, device))
        for name, logits in (('probabilities', out['logits']),
                              ('observation_probabilities', out['observation_logits'][:, -1])):
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite evaluation prediction')
            result[name].extend(logits.float().softmax(-1).cpu().tolist())
        for name, key in (('labels', 'context_label'), ('episodes', 'episode_id'), ('timestamps', 'timestamp')):
            result[name].extend(batch[key].tolist())
    metrics = classification_metrics(result['labels'], result['probabilities'])
    baseline = classification_metrics(result['labels'], result['observation_probabilities'])
    metrics['temporal'] = temporal_metrics(result['labels'], result['probabilities'], result['episodes'], result['timestamps'])
    baseline['temporal'] = temporal_metrics(result['labels'], result['observation_probabilities'], result['episodes'], result['timestamps'])
    metrics['observation_only'] = baseline
    return metrics, result


def train(cfg, *, resume=None, dry_run=False, prepare_cache_only=False, smoke_batches=None,
          rank=0, world=1, device=torch.device('cpu')):
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if cfg['model']['history_steps'] != cfg['dataset']['history_steps']:
        raise ValueError('Model and data context-history lengths differ')
    rgb_horizon = cfg['dataset'].get('rgb_horizon', 4)
    rgb_span, _ = rgb_window_spec(rgb_horizon, cfg['dataset'].get('rgb_span_s'))
    model_span, _ = rgb_window_spec(cfg['model'].get('rgb_horizon', 4), cfg['model'].get('rgb_span_s'))
    if cfg['model'].get('rgb_horizon', 4) != rgb_horizon or abs(rgb_span - model_span) > 1e-9:
        raise ValueError('Model and data RGB window contracts differ')
    period = cfg['dataset']['history_stride'] / (60000 / 1001)
    if abs(cfg['model']['history_period_s'] - period) > 1e-6:
        raise ValueError('Context history must use the same ~10 Hz time grid')
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1 or cfg['num_workers'] < 0 or cfg['auxiliary_loss_weight'] <= 0:
        raise ValueError('Invalid training settings')
    base = SlipSequenceDataset(**cfg['dataset'])
    splits = split_slip_episodes(base, cfg['seed'], cfg['validation_ratio'], cfg['test_ratio'])
    datasets = {name: base.subset(episodes) for name, episodes in splits.items()}
    counts = {name: np.bincount(base.targets[ds.indices], minlength=4).tolist() for name, ds in datasets.items()}
    manifest = dict(schema=SCHEMA, phase_names=list(PHASE_NAMES), source_fingerprint=base.fingerprint,
                    episode_splits=splits, original_episode_splits={k: base.original_episode_numbers[v].tolist()
                                                                   for k, v in splits.items()},
                    dataset_contract=cfg['dataset'], model_config=cfg['model'])
    report = dict(manifest, all_frames=len(base.targets), ignored_frames=int((~base.valid).sum()),
                  labelled_frames=int(base.valid.sum()), full_window_frames=int(base.window_usable.sum()),
                  sampled_10hz_frames=len(base), class_counts=counts,
                  rgb_offsets_frames=base.rgb_offsets.tolist(), rgb_history_span_s=base.rgb_span_s, raw_ft_per_rgb=41,
                  history_source='same_model_observation_probabilities_detached',
                  history_span_s=cfg['model']['history_steps'] * period,
                  batch_size_per_gpu=cfg['batch_size'], world_size=world,
                  physical_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if dry_run:
        if rank == 0:
            print(json.dumps(report, indent=2), flush=True)
        base.close()
        return report
    outdir = Path(cfg['output_dir'])
    saved = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if saved:
        if saved['manifest'] != manifest or saved['training_config'] != dict(cfg, epochs=saved['training_config']['epochs']):
            raise ValueError('Resume changed the data, model, cache, optimizer or split contract')
        if saved.get('smoke_test', False) != (smoke_batches is not None):
            raise ValueError('Smoke checkpoints cannot resume a full training run')
        if Path(resume).resolve().parent != outdir.resolve():
            raise ValueError('Resume must use the same output directory')
    if not prepare_cache_only and not resume and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('Output is not empty; use a new output_dir or --resume')
    have_cache = (Path(cfg['cache_dir']) / 'manifest.json').exists()
    model = (SlipTemporalContextEncoder.from_checkpoint(saved) if saved else
             SlipTemporalContextEncoder(**cfg['model'], pretrained=prepare_cache_only and not have_cache))
    cache = ensure_feature_cache(base, model, cfg['cache_dir'], device, rank=rank, world=world,
                                 batch_size=cfg['cache_batch_size'], workers=cfg['num_workers'], create=prepare_cache_only)
    if saved and saved['cache_manifest'] != cache:
        raise ValueError('Resume feature-cache fingerprint changed')
    if prepare_cache_only:
        if rank == 0:
            print(json.dumps(dict(event='cache_complete', directory=cfg['cache_dir'])), flush=True)
        base.close()
        return report
    for ds in datasets.values():
        ds.attach_features(base.feature_path)
    if not saved:
        model.set_force_statistics(*base.force_statistics(splits['train']))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    first_epoch, best = 0, -1.
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch, best = int(saved['epoch']) + 1, float(saved['best_validation_macro_f1'])
    if first_epoch >= cfg['epochs']:
        raise ValueError('Increase epochs to resume this checkpoint')
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        atomic_json(outdir / 'dataset_report.json', report)
        atomic_json(outdir / 'episode_splits.json', manifest)
        OmegaConf.save(OmegaConf.create(cfg), outdir / 'resolved_config.yaml')
        print(json.dumps(dict(event='model_ready', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              samples=len(base), class_counts=counts)), flush=True)
    if world > 1:
        dist.barrier()
    network = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                     broadcast_buffers=False) if world > 1 else model
    torch.manual_seed(cfg['seed'] + rank)
    sampler = DistributedSampler(datasets['train'], num_replicas=world, rank=rank, seed=cfg['seed']) if world > 1 else None
    kwargs = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                  persistent_workers=cfg['num_workers'] > 0, pin_memory=device.type == 'cuda')
    loader = DataLoader(datasets['train'], sampler=sampler, shuffle=sampler is None, **kwargs)
    validation = DataLoader(datasets['validation'], shuffle=False, **kwargs) if rank == 0 else None
    weights = torch.tensor(np.sum(counts['train']) / (4 * np.asarray(counts['train'])), device=device, dtype=torch.float32)
    amp = bool(cfg['amp']) and device.type == 'cuda'
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU lacks bfloat16; set amp=false')
    epochs = range(first_epoch, min(cfg['epochs'], first_epoch + 1) if smoke_batches else cfg['epochs'])
    for epoch in epochs:
        network.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total = torch.zeros(2, device=device, dtype=torch.float64)
        for step, batch in enumerate(loader):
            if smoke_batches is not None and step >= smoke_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            targets = batch['context_label'].to(device)
            aux_targets = batch['sequence_labels'].to(device).flatten()
            aux_valid = aux_targets >= 0
            denominator = torch.stack([weights[targets].sum(), weights[aux_targets[aux_valid]].sum()])
            if world > 1:
                dist.all_reduce(denominator)
            with autocast_context(device, amp):
                result = network(move_obs(batch, device))
                main = F.cross_entropy(result['logits'].float(), targets, weight=weights, reduction='sum')
                auxiliary = F.cross_entropy(result['observation_logits'].float().reshape(-1, 4), aux_targets,
                                             weight=weights, ignore_index=-100, reduction='sum')
                loss = world * (main / denominator[0] + cfg['auxiliary_loss_weight'] * auxiliary / denominator[1])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            total += torch.stack([main.detach().double(), weights[targets].sum().double()])
            if rank == 0 and (step == 0 or (step + 1) % cfg['log_every'] == 0):
                print(json.dumps(dict(epoch=epoch + 1, step=step + 1, batches=len(loader), loss=float(loss.detach()))), flush=True)
        if world > 1:
            dist.all_reduce(total)
        if rank == 0:
            metrics, _ = evaluate(model, validation, device, amp, smoke_batches)
            improved = metrics['macro_f1'] > best
            best = max(best, metrics['macro_f1'])
            record = dict(epoch=epoch + 1, train_loss=float(total[0] / total[1]), validation=metrics,
                          smoke_test=smoke_batches is not None)
            with (outdir / 'training_history.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            checkpoint = dict(schema=SCHEMA, phase_names=list(PHASE_NAMES),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, model_config=model.config,
                training_config=cfg, manifest=manifest, cache_manifest=cache, epoch=epoch,
                optimizer=optimizer.state_dict(), best_validation_macro_f1=best,
                validation_metrics=metrics, smoke_test=smoke_batches is not None)
            save_checkpoint(outdir / 'last.pt', checkpoint)
            if improved:
                save_checkpoint(outdir / 'best.pt', checkpoint)
        if world > 1:
            dist.barrier()
    if rank == 0:
        best_payload = torch.load(outdir / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(best_payload['state_dict'], strict=True)
        test_loader = DataLoader(datasets['test'], shuffle=False, **kwargs)
        metrics, result = evaluate(model, test_loader, device, amp, smoke_batches)
        write_evaluation(outdir / ('smoke_eval' if smoke_batches else 'test_eval'), result['labels'], result['probabilities'],
                         result['episodes'], result['timestamps'], names=list(PHASE_NAMES),
                         extra=dict(observation_only=metrics['observation_only'], temporal=metrics['temporal'],
                                    smoke_test=smoke_batches is not None, checkpoint_selected_by='validation_macro_f1'))
        atomic_json(outdir / 'test_comparison.json', metrics)
        print(json.dumps(dict(event='complete', checkpoint=str(outdir / 'best.pt'), test_macro_f1=metrics['macro_f1'])), flush=True)
    if world > 1:
        dist.barrier()
    for ds in [base, *datasets.values()]:
        ds.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='context/qwen/config/train_slip_context.yaml')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--resume')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare-cache-only', action='store_true')
    mode.add_argument('--smoke-batches', type=int)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set)), resolve=True)
    if args.smoke_batches is not None and (args.smoke_batches < 1 or cfg['output_dir'] in
            ('outputs/slip_context_sequence', 'outputs/slip_context_rgb8_1s_history10')):
        raise ValueError('Smoke tests need positive batches and a separate output_dir')
    rank, world, local = (int(os.environ.get(k, default)) for k, default in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    device = torch.device('cuda', local) if cfg['device'] == 'cuda' and not args.dry_run else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=30))
    try:
        train(cfg, resume=args.resume, dry_run=args.dry_run, prepare_cache_only=args.prepare_cache_only,
              smoke_batches=args.smoke_batches, rank=rank, world=world, device=device)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
