"""Pose-free policy observations and demonstrated relative action targets."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from diffusion_policy.common.nested_zarr import open_nested_zip_group, sha256_file
from diffusion_policy.common.normalize_util import (
    concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat,
)
from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.umi_dual_ft_dataset import _StatsAccumulator, _stats
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d


PHASE_NAMES = ('approach', 'error1', 'error2', 'turning')
CACHE_SCHEMA = 'slip_action_context_predictions_v1'
INPUT_KEYS = ('camera0_rgb', 'robot0_ft_left', 'robot0_ft_right', 'valve_context')


def expand_context_predictions(recording, frames, probabilities):
    """Hold only a past 10 Hz prediction; never bridge ignore/episode/gap boundaries."""
    frames = np.asarray(frames, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float32)
    n = len(recording.rgb_times)
    if (frames.ndim != 1 or len(frames) == 0 or np.any(np.diff(frames) <= 0)
            or frames.min() < 0 or frames.max() >= n):
        raise ValueError('Context anchors must be ordered unique source frames')
    if (p.shape != (len(frames), 4) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(-1), 1, atol=1e-5)):
        raise ValueError('Context must contain four normalized probabilities')
    if not recording.valid[frames].all() or np.any(
            (frames - recording.rgb_starts[recording.frame_episode[frames]]) % 6):
        raise ValueError('Context anchors must be valid 10 Hz source frames')
    current = np.arange(n)
    rows = np.searchsorted(frames, current, side='right') - 1
    safe_rows = np.maximum(rows, 0)
    source = frames[safe_rows]
    age = recording.rgb_times - recording.rgb_times[source]
    valid = ((rows >= 0) & (source <= current) & (current - source < 6)
             & (recording.frame_episode[source] == recording.frame_episode)
             & (source >= recording.run_starts) & recording.valid
             & (age >= 0) & (age <= .125))
    values = np.zeros((n, 5), np.float32)
    values[:, :4] = .25
    values[valid, :4] = p[safe_rows[valid]]
    values[valid, 4] = 1
    return values, np.where(valid, source, -1)


class SlipActionDataset(BaseImageDataset):
    """Four RGB/left-FT/right-FT triplets; TCP appears ONLY in action targets.

    Each image owns 41 preceding native F/T rows. The encoder converts these
    to 32 raw/mean/delta rows. Inputs and the complete future action interval
    must lie in a contiguous non-ignore run. The context checkpoint owns the
    train/validation/test episode split.
    """
    def __init__(self, shape_meta, dataset_path, force_sidecar_path, label_path,
                 context_cache_path, context_checkpoint_path, split='train'):
        self.shape_meta = shape_meta
        expected = {'camera0_rgb': (4, [3, 224, 224]),
                    'robot0_ft_left': (4, [41, 6]), 'robot0_ft_right': (4, [41, 6]),
                    'valve_context': (1, [5])}
        if set(shape_meta['obs']) != set(expected):
            raise ValueError('Slip action observations must exclude TCP and use only RGB/F/T/context')
        for key, (horizon, shape) in expected.items():
            if int(shape_meta['obs'][key]['horizon']) != horizon or list(shape_meta['obs'][key]['shape']) != shape:
                raise ValueError(f'Invalid slip action observation shape: {key}')
        if (list(shape_meta['action']['shape']) != [11]
                or int(shape_meta['action']['horizon']) != 16
                or int(shape_meta['action']['down_sample_steps']) != 3
                or int(shape_meta['obs']['camera0_rgb']['down_sample_steps']) != 10):
            raise ValueError('Require four RGB at stride 10 and 16 actions at stride 3')
        self.recording = SlipSequenceDataset(dataset_path, force_sidecar_path, label_path,
                                             rgb_horizon=4, rgb_span_s=.5)
        ds = self.recording
        with np.load(context_cache_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata'].item()))
            if (metadata.get('schema') != CACHE_SCHEMA
                    or tuple(metadata.get('phase_names', ())) != PHASE_NAMES
                    or metadata.get('source_fingerprint') != ds.fingerprint
                    or metadata.get('checkpoint_sha256') != sha256_file(context_checkpoint_path)):
                raise ValueError('Context cache does not match the slip recording/checkpoint/class order')
            self.context, self.context_source = expand_context_predictions(
                ds, saved['frames'], saved['probabilities'])
        self.context_splits = metadata['episode_splits']
        if set(self.context_splits) != {'train', 'validation', 'test'}:
            raise ValueError('Context cache must provide train/validation/test episodes')
        all_episodes = sum([list(v) for v in self.context_splits.values()], [])
        if sorted(all_episodes) != list(range(len(ds.rgb_ends))):
            raise ValueError('Episode splits must be disjoint and cover the recording')
        self.action_training_metadata = dict(metadata, context_cache_path=str(context_cache_path),
            context_checkpoint_path=str(context_checkpoint_path),
            observation_contract='rgb4_500ms_paired_ft41_no_tcp_v1')
        store, root, _ = open_nested_zip_group(dataset_path)
        try:
            pos = np.asarray(root['data/robot0_eef_pos'][:], dtype=np.float64)
            rot = np.asarray(root['data/robot0_eef_rot_axis_angle'][:], dtype=np.float64)
            self.width = np.asarray(root['data/robot0_gripper_width'][:], dtype=np.float32)
        finally:
            store.close()
        n = len(ds.rgb_times)
        if pos.shape != (n, 3) or rot.shape != (n, 3) or self.width.shape != (n, 1):
            raise ValueError('Demonstrated pose/width must match the RGB timeline')
        if not all(np.isfinite(a).all() for a in (pos, rot, self.width)):
            raise ValueError('Nonfinite demonstrated action source')
        self.pose = pose_to_mat(np.concatenate([pos, rot], -1))
        self.inverse_pose = np.linalg.inv(self.pose)
        self.grasp_force = np.empty(n, np.float32)
        for rs, re, fs, fe in zip(ds.rgb_starts, ds.rgb_ends, ds.force_starts, ds.force_ends):
            # Bias has already been removed in the native sidecar. As in the
            # existing policy, signed Fz difference is an action reference,
            # not a command to the gripper force register.
            force = .5 * (ds.wrench[fs:fe, 8] - ds.wrench[fs:fe, 2])
            self.grasp_force[rs:re] = np.interp(ds.rgb_times[rs:re], ds.force_times[fs:fe], force)
        current = np.arange(n)
        end = current + 45  # (16 - 1) * 3
        safe_end = np.minimum(end, n - 1)
        # Check ALL frames between current and final target, not only every
        # third frame; invalid intermediate observations must not be crossed.
        bad = np.r_[0, np.cumsum(~(ds.valid & ds.force_available))]
        usable = (ds.window_usable & (self.context_source >= 0)
                  & (end < ds.rgb_ends[ds.frame_episode])
                  & (bad[safe_end + 1] == bad[current]))
        self.all_indices = np.flatnonzero(usable)
        self._set_split(split)

    def _set_split(self, split):
        if split not in self.context_splits:
            raise ValueError(f'Unknown episode split {split}')
        self.split = split
        ds = self.recording
        self.indices = self.all_indices[np.isin(ds.frame_episode[self.all_indices], self.context_splits[split])]
        if not len(self.indices):
            raise ValueError(f'No valid action windows in {split}')

    def subset(self, split):
        result = copy.copy(self)
        result.recording = copy.copy(self.recording)
        result.recording._store = result.recording._rgb = result.recording._pid = None
        result._set_split(split)
        return result

    def get_validation_dataset(self):
        return self.subset('validation')

    def window_indices(self, frame):
        ds = self.recording
        rgb = int(frame) - ds.rgb_offsets
        force = ds.force_index[rgb, None] - np.arange(40, -1, -1)
        return rgb, force

    def action_targets(self, frames):
        frames = np.asarray(frames, dtype=np.int64)
        ids = frames[:, None] + np.arange(16) * 3
        relative = self.inverse_pose[frames, None] @ self.pose[ids]
        return np.concatenate([mat_to_pose10d(relative), self.width[ids],
                               self.grasp_force[ids, None]], -1).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame = int(self.indices[index])
        ds = self.recording
        rgb, force = self.window_indices(frame)
        rgb_array = ds._get_rgb()
        image = np.stack([rgb_array[int(i)] for i in rgb]).transpose(0, 3, 1, 2).copy()
        wrench = ds.wrench[force]
        # No pose, manual class ID, future state, or outcome enters obs.
        obs = dict(camera0_rgb=torch.from_numpy(image).float().div_(255),
                   robot0_ft_left=torch.from_numpy(wrench[..., :6].copy()),
                   robot0_ft_right=torch.from_numpy(wrench[..., 6:].copy()),
                   valve_context=torch.from_numpy(self.context[frame, None].copy()))
        return dict(obs=obs, action=torch.from_numpy(self.action_targets([frame])[0]))

    def get_normalizer(self, **kwargs):
        # Even when invoked on validation/test, fit exclusively on eligible
        # training observations and demonstrated targets.
        ds = self.recording
        frames = self.all_indices[np.isin(ds.frame_episode[self.all_indices], self.context_splits['train'])]
        rgb = np.unique((frames[:, None] - ds.rgb_offsets).ravel())
        force_rows = np.unique((ds.force_index[rgb, None] - np.arange(40, -1, -1)).ravel())
        normalizer = LinearNormalizer()
        normalizer['camera0_rgb'] = get_image_identity_normalizer()
        normalizer['valve_context'] = get_image_identity_normalizer()
        for key, side in (('robot0_ft_left', slice(0, 6)), ('robot0_ft_right', slice(6, 12))):
            normalizer[key] = get_range_normalizer_from_stat(_stats(ds.wrench[force_rows, side]))
        stats = _StatsAccumulator(11)
        for begin in range(0, len(frames), 1024):
            stats.update(self.action_targets(frames[begin:begin + 1024]))
        stat = stats.result()
        fields = []
        for columns, identity in ((slice(0, 3), False), (slice(3, 9), True),
                                  (slice(9, 10), False), (slice(10, 11), False)):
            values = {k: v[columns] for k, v in stat.items()}
            fields.append((get_identity_normalizer_from_stat if identity else get_range_normalizer_from_stat)(values))
        normalizer['action'] = concatenate_normalizer(fields)
        return normalizer

    def close(self):
        self.recording.close()
