"""Cache frozen CLIP frame features without fitting anything on held-out labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from diffusion_policy.common.nested_zarr import sha256_file
from diffusion_policy.context.labels import atomic_json


class _Images(Dataset):
    def __init__(self, recording, frames):
        self.recording, self.frames = recording, frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        frame = int(self.frames[i])
        rgb = np.asarray(self.recording._get_rgb()[frame]).transpose(2, 0, 1).copy()
        return frame, torch.from_numpy(rgb).float().div_(255)


def ensure_feature_cache(recording, model, directory, device, *, rank=0, world=1,
                         batch_size=64, workers=4, create=False):
    """Two ranks can write disjoint rows; the completion manifest is published last."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    expected = dict(schema='slip_frozen_clip_v1', data_fingerprint=recording.fingerprint,
                    model_name=model.config['model_name'],
                    shape=[len(recording.rgb_times), model.vision.num_features],
                    rgb_mean=model.rgb_mean.flatten().tolist(), rgb_std=model.rgb_std.flatten().tolist(),
                    dtype='float32', input_scale='uint8 / 255', augmentation='none', precision='float32')
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['contract'] != expected:
            raise ValueError('CLIP cache belongs to a different recording or encoder contract')
        for name in ('vision.pt', 'features.npy'):
            if sha256_file(directory / name) != saved['sha256'][name]:
                raise ValueError(f'Incomplete or modified CLIP cache: {name}')
        model.vision.load_state_dict(torch.load(directory / 'vision.pt', map_location='cpu', weights_only=False), strict=True)
        recording.attach_features(directory / 'features.npy')
        return saved
    if not create:
        raise FileNotFoundError('Run with --prepare-cache-only before training')
    if rank == 0:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Incomplete cache directory; choose a new cache_dir')
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.vision.state_dict().items()}, directory / 'vision.pt')
        mem = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype=np.float32,
                                       shape=tuple(expected['shape']))
        mem.flush()
        del mem
    if world > 1:
        dist.barrier()
    model.vision.load_state_dict(torch.load(directory / 'vision.pt', map_location='cpu', weights_only=False), strict=True)
    model.to(device).eval()
    # Invalid/ignore RGB is never encoded. All actual training inputs are covered.
    frames = np.flatnonzero(recording.valid)[rank::world]
    loader = DataLoader(_Images(recording, frames), batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == 'cuda', shuffle=False)
    mem = np.load(directory / 'features.npy', mmap_mode='r+', allow_pickle=False)
    with torch.inference_mode():
        for step, (ids, rgb) in enumerate(loader):
            features = model.encode_images(rgb.to(device, non_blocking=True)).float().cpu().numpy()
            if not np.isfinite(features).all():
                raise FloatingPointError('Nonfinite CLIP features')
            mem[ids.numpy()] = features
            if step % 100 == 0:
                print(json.dumps(dict(event='clip_cache', rank=rank, batch=step + 1, batches=len(loader))), flush=True)
    mem.flush()
    del mem
    if world > 1:
        dist.barrier()
    if rank == 0:
        saved = dict(contract=expected, sha256={name: sha256_file(directory / name)
                                              for name in ('vision.pt', 'features.npy')})
        atomic_json(manifest_path, saved)
    if world > 1:
        dist.barrier()
    recording.attach_features(directory / 'features.npy')
    return json.loads(manifest_path.read_text())
