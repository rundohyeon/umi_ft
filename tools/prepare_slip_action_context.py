#!/usr/bin/env python3
"""Freeze slip context predictions for action training; no action optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from diffusion_policy.common.nested_zarr import sha256_file
from diffusion_policy.context.slip_feature_cache import ensure_feature_cache
from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset
from diffusion_policy.dataset.slip_action_dataset import CACHE_SCHEMA, PHASE_NAMES
from models.slip_temporal_context import SlipTemporalContextEncoder


@torch.inference_mode()
def infer_context_anchors(model, ds, device, batch_size=128):
    """Encode each current window once, then use strictly past base probabilities.

    Equivalent to the sequence classifier's full forward(), without repeatedly
    re-encoding the same past windows. Corrected probabilities never feed back
    into their own history.
    """
    features = np.load(ds.feature_path, mmap_mode='r', allow_pickle=False)
    anchors = ds.indices
    encoded, logits = [], []
    for begin in range(0, len(anchors), batch_size):
        rgb = anchors[begin:begin + batch_size, None] - ds.rgb_offsets
        force = ds.force_index[rgb][..., None] - np.arange(40, -1, -1)
        wrench = ds.wrench[force]
        obs = dict(rgb_features=torch.from_numpy(np.asarray(features[rgb]).copy()).to(device),
                   robot0_ft_left=torch.from_numpy(wrench[..., :6].copy()).to(device),
                   robot0_ft_right=torch.from_numpy(wrench[..., 6:].copy()).to(device))
        f, p = model.observe(obs)
        encoded.append(f.cpu()); logits.append(p.float().cpu())
        if begin % (batch_size * 25) == 0:
            print(json.dumps(dict(event='observe', completed=begin, total=len(anchors))), flush=True)
    encoded, logits = torch.cat(encoded), torch.cat(logits)
    base = logits.softmax(-1)
    lookup = np.full(len(ds.rgb_times), -1, np.int64)
    lookup[anchors] = np.arange(len(anchors))
    output = []
    for begin in range(0, len(anchors), batch_size):
        sequences = [ds.sequence_indices(int(frame)) for frame in anchors[begin:begin + batch_size]]
        history = np.stack([lookup[x[0][:-1]] for x in sequences])
        valid = np.stack([x[3][:-1] for x in sequences])
        ages = np.stack([x[4][:-1] for x in sequences])
        if (history[valid] < 0).any():
            raise ValueError('Missing past observation prediction')
        prediction = model.fuse(encoded[begin:begin + len(sequences)].to(device),
            logits[begin:begin + len(sequences)].to(device),
            base[torch.from_numpy(np.maximum(history, 0))].to(device),
            torch.from_numpy(valid).to(device), torch.from_numpy(ages).to(device))
        output.append(prediction.float().softmax(-1).cpu().numpy())
    return np.concatenate(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='outputs/slip_context_sequence/best.pt')
    p.add_argument('--cache-dir', default='outputs/slip_clip_features')
    p.add_argument('--output', default='outputs/slip_action/context_rgb4.npz')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output path; already exists: {output}')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('smoke_test', False):
        raise ValueError('A smoke-test checkpoint cannot supply training context')
    model = SlipTemporalContextEncoder.from_checkpoint(payload).eval()
    manifest = payload['manifest']
    cfg = dict(payload['training_config']['dataset'])
    cfg.update(rgb_horizon=model.rgb_horizon, rgb_span_s=model.rgb_span_s,
               history_steps=model.history_steps)
    ds = SlipSequenceDataset(**cfg)
    if ds.fingerprint != manifest['source_fingerprint']:
        raise ValueError('The context checkpoint belongs to a different recording')
    cache_manifest = json.loads((Path(args.cache_dir) / 'manifest.json').read_text())
    if cache_manifest != payload['cache_manifest']:
        raise ValueError('Feature cache differs from the classifier training cache')
    vision = torch.load(Path(args.cache_dir) / 'vision.pt', map_location='cpu', weights_only=False)
    if any(not torch.equal(value, payload['state_dict']['vision.' + key]) for key, value in vision.items()):
        raise ValueError('Cached CLIP features do not match the checkpoint vision weights')
    device = torch.device(args.device)
    ensure_feature_cache(ds, model, args.cache_dir, device, create=False)
    model.to(device).eval()
    probabilities = infer_context_anchors(model, ds, device, args.batch_size)
    metadata = dict(schema=CACHE_SCHEMA, phase_names=list(PHASE_NAMES),
        source_fingerprint=ds.fingerprint, episode_splits=manifest['episode_splits'],
        checkpoint_sha256=sha256_file(args.checkpoint), model_config=payload['model_config'],
        context_cadence_frames=6, history_source='past_observation_probabilities',
        prediction_source='frozen_classifier', precision='float32')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, frames=ds.indices, probabilities=probabilities,
                            metadata=json.dumps(metadata))
    print(json.dumps(dict(output=str(output), predictions=len(ds.indices),
                         class_counts=np.bincount(probabilities.argmax(-1), minlength=4).tolist())), flush=True)
    ds.close()


if __name__ == '__main__':
    main()
