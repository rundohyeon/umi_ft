#!/usr/bin/env python
"""Two optimization steps on real slip samples; never saves a usable checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F

from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset, split_slip_episodes
from models.slip_temporal_context import SlipTemporalContextEncoder
from train_context_rgb_force import autocast_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='context/qwen/config/train_slip_context.yaml')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    rank, world, local = (int(os.environ.get(k, default)) for k, default in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    if world not in (1, 2):
        raise ValueError('Smoke check supports one or two ranks')
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    ds = SlipSequenceDataset(**cfg['dataset'])
    try:
        splits = split_slip_episodes(ds, cfg['seed'], cfg['validation_ratio'], cfg['test_ratio'])
        train = ds.subset(splits['train'])
        selected = []
        for cid in range(4):
            candidates = train.indices[train.targets[train.indices] == cid]
            candidates = [int(f) for f in candidates if ds.sequence_indices(int(f))[3].all()]
            if not candidates:
                raise ValueError(f'No full-history real sample for class {cid}')
            selected.append(candidates[len(candidates) // 2])
        selected = selected[rank::world]
        windows = [ds.sequence_indices(f) for f in selected]
        model = SlipTemporalContextEncoder(**cfg['model'], pretrained=True).to(device)
        model.set_force_statistics(*ds.force_statistics(splits['train']))
        model.eval()
        images = np.unique(np.concatenate([w[1].ravel() for w in windows]))
        features = {}
        with torch.no_grad():
            for start in range(0, len(images), 32):
                ids = images[start:start + 32]
                rgb = np.stack([ds._get_rgb()[int(i)] for i in ids]).transpose(0, 3, 1, 2).copy()
                encoded = model.encode_images(torch.from_numpy(rgb).to(device).float() / 255)
                features.update(zip(ids.tolist(), encoded.cpu()))
        obs = dict(rgb_features=torch.stack([torch.stack([features[int(i)] for i in w[1].ravel()]).reshape(ds.history_steps + 1, ds.rgb_horizon, 768)
                                             for w in windows]),
                   robot0_ft_left=torch.from_numpy(np.stack([ds.wrench[w[2]][..., :6] for w in windows])),
                   robot0_ft_right=torch.from_numpy(np.stack([ds.wrench[w[2]][..., 6:] for w in windows])),
                   sequence_valid=torch.from_numpy(np.stack([w[3] for w in windows])),
                   sequence_ages=torch.from_numpy(np.stack([w[4] for w in windows])))
        obs = {k: v.to(device) for k, v in obs.items()}
        labels = torch.from_numpy(np.stack([ds.targets[w[0]] for w in windows])).to(device)
        network = DistributedDataParallel(model, device_ids=[local] if device.type == 'cuda' else None,
                                         broadcast_buffers=False) if world > 1 else model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'])
        torch.manual_seed(cfg['seed'] + rank)
        network.train()
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg['amp']):
                result = network(obs)
                loss = F.cross_entropy(result['logits'].float(), labels[:, -1])
                loss = loss + .5 * F.cross_entropy(result['observation_logits'].float().reshape(-1, 4), labels.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite real-data smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(json.dumps(dict(event='real_data_smoke_passed', rank=rank, visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              frames=selected, labels=labels[:, -1].tolist(), rgb_features_shape=list(obs['rgb_features'].shape),
                              force_shape=list(obs['robot0_ft_left'].shape), losses=losses,
                              peak_gpu_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                              checkpoint_saved=False)), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        ds.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
