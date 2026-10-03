"""Frame-wise A/T/R/E labels with causal RGB + native wrench, without TCP."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.nested_zarr import open_nested_zip_group
from diffusion_policy.context.ft_features import causal_ft_features, ft_feature_contract


PHASE_NAMES = ('approach', 'turning', 'recovery', 'error')


class CanonicalRGBForceDataset(Dataset):
    def __init__(self, dataset_path, force_sidecar_path, label_path,
                 rgb_stride=3, ft_history=32, max_force_age_s=0.012,
                 rule_label_weight=0.3, manual_label_weight=1.0,
                 ft_feature_mode='raw', ft_mean_window=5, ft_delta_lag=5):
        register_codecs()
        self.dataset_path = str(Path(dataset_path).resolve())
        self.force_sidecar_path = str(Path(force_sidecar_path).resolve())
        self.label_path = str(Path(label_path).resolve())
        self.rgb_stride, self.ft_history = int(rgb_stride), int(ft_history)
        self.ft_contract = ft_feature_contract(ft_feature_mode, ft_mean_window, ft_delta_lag)
        self.required_ft_history = self.ft_history + self.ft_contract['prefix_samples']
        if self.rgb_stride < 1 or self.ft_history != 32:
            raise ValueError('Require positive RGB stride and 32 native F/T samples')
        if max_force_age_s <= 0 or rule_label_weight <= 0 or manual_label_weight <= 0:
            raise ValueError('Force age and source weights must be positive')
        with np.load(self.label_path, allow_pickle=False) as labels:
            if str(labels['schema']) != 'valve_context_manual_v12_supervision_v2_4state':
                raise ValueError('Unexpected canonical label schema')
            if not bool(labels['training_eligible']):
                raise ValueError('Labels are not eligible for training')
            if tuple(labels['phase_names'].tolist()) != PHASE_NAMES:
                raise ValueError('Expected approach/turning/recovery/error in that order')
            self.targets = labels['state_phase'].astype(np.int64)
            raw_valid = labels['state_phase_valid']
            if not np.isin(raw_valid, [0, 1]).all():
                raise ValueError('Label validity must be binary')
            self.valid = raw_valid.astype(bool)
            self.label_source = labels['state_label_source'].astype(np.int64)
            self.source_names = labels['label_source_names'].tolist()
            self.rgb_times = labels['rgb_timestamp_s'].astype(np.float64)
            self.rgb_ends = labels['rgb_episode_ends'].astype(np.int64)
            source_episode = labels['source_episode'].copy()
            approved = labels['episode_is_approved'].astype(bool)
        if self.source_names != ['v12_001_193', 'manual_194_214', 'manual_215_284']:
            raise ValueError('Unknown label provenance; source weighting must be explicit')
        n = len(self.rgb_times)
        if any(a.shape != (n,) for a in (self.targets, self.valid, self.label_source)):
            raise ValueError('Label arrays do not match the RGB timeline')
        if not np.isfinite(self.rgb_times).all() or not len(self.rgb_ends):
            raise ValueError('Invalid RGB timestamps or empty episodes')
        if self.rgb_ends[-1] != n or np.any(np.diff(np.r_[0, self.rgb_ends]) <= 0):
            raise ValueError('Invalid RGB episode boundaries')
        if approved.shape != self.rgb_ends.shape or not approved.all():
            raise ValueError('Canonical episodes must all be approved')
        if not np.isin(self.targets[self.valid], np.arange(4)).all():
            raise ValueError('A valid label has an invalid class ID')
        if not np.isin(self.label_source[self.valid], np.arange(3)).all():
            raise ValueError('A valid label has an invalid source ID')
        self.rgb_starts = np.r_[0, self.rgb_ends[:-1]]
        self.frame_episode = np.repeat(np.arange(len(self.rgb_ends)), np.diff(np.r_[0, self.rgb_ends]))

        store, root, self.prefix = open_nested_zip_group(self.dataset_path)
        try:
            if not np.array_equal(root['meta/episode_ends'][:], self.rgb_ends):
                raise ValueError('Base RGB and label episode boundaries differ')
            if root['data/camera0_rgb'].shape != (n, 224, 224, 3):
                raise ValueError('Expected aligned uint8 RGB frames at 224x224')
            if root['data/camera0_rgb'].dtype != np.dtype('uint8'):
                raise ValueError('Expected uint8 RGB storage')
        finally:
            store.close()
        store, sidecar, _ = open_nested_zip_group(self.force_sidecar_path)
        try:
            expected_channels = [f'{axis}_{side}' for side in ('l', 'r')
                                 for axis in ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')]
            if (sidecar.attrs.get('schema') != 'umi_force_sidecar_v1'
                    or sidecar.attrs.get('train_wrench_key') != 'wrench_12d'
                    or list(sidecar.attrs.get('wrench_channel_order', [])) != expected_channels):
                raise ValueError('Expected bias-removed native-frame 12-D force sidecar')
            if not np.array_equal(sidecar['meta/rgb_episode_ends'][:], self.rgb_ends):
                raise ValueError('Force sidecar and label RGB boundaries differ')
            if not np.array_equal(sidecar['meta/source_episode'][:], source_episode):
                raise ValueError('Force sidecar and label source episodes differ')
            sidecar_times = sidecar['data/rgb_timestamp_s'][:]
            if sidecar_times.shape != self.rgb_times.shape or not np.allclose(
                    sidecar_times, self.rgb_times, atol=1e-6, rtol=0):
                raise ValueError('Force sidecar and label RGB timestamps differ')
            self.wrench = np.asarray(sidecar['data/wrench_12d'][:], dtype=np.float32)
            self.force_times = np.asarray(sidecar['data/wrench_timestamp_s'][:], dtype=np.float64)
            self.force_ends = np.asarray(sidecar['meta/wrench_episode_ends'][:], dtype=np.int64)
            self.force_index = np.asarray(sidecar['data/rgb_to_wrench_end_idx'][:], dtype=np.int64)
            force_valid = np.asarray(sidecar['data/rgb_wrench_valid'][:], dtype=bool)
            force_age = np.asarray(sidecar['data/rgb_wrench_age_s'][:], dtype=np.float64)
        finally:
            store.close()
        if (self.wrench.shape != (len(self.force_times), 12)
                or self.force_ends.shape != self.rgb_ends.shape
                or self.force_ends[-1] != len(self.force_times)
                or np.any(np.diff(np.r_[0, self.force_ends]) <= 0)
                or not np.isfinite(self.wrench).all() or not np.isfinite(self.force_times).all()):
            raise ValueError('Invalid native F/T arrays or episode boundaries')
        if any(a.shape != (n,) for a in (self.force_index, force_valid, force_age)):
            raise ValueError('Invalid RGB-to-force mapping shapes')
        self.force_starts = np.r_[0, self.force_ends[:-1]]
        # Verify every index using its own episode's timestamps; never accept a
        # nearest/future sample or a mapping that crosses an episode boundary.
        for rs, re, fs, fe in zip(self.rgb_starts, self.rgb_ends, self.force_starts, self.force_ends):
            rt, ft = self.rgb_times[rs:re], self.force_times[fs:fe]
            if np.any(np.diff(rt) <= 0) or np.any(np.diff(ft) <= 0):
                raise ValueError('Unordered episode timestamps')
            local = np.searchsorted(ft, rt, side='right') - 1
            expected = np.where(local >= 0, fs + local, -1)
            if not np.array_equal(expected, self.force_index[rs:re]):
                raise ValueError('Noncausal or misaligned RGB-to-force indices')
            available = local >= 0
            age = rt[available] - ft[local[available]]
            if not np.allclose(age, force_age[rs:re][available], rtol=0, atol=1e-6):
                raise ValueError('Stored force ages disagree with timestamps')
        self.usable = (self.valid & force_valid & (self.force_index >= 0)
                       & np.isfinite(force_age) & (force_age >= 0) & (force_age <= max_force_age_s))
        self.sample_weights = np.where(self.label_source == 0, rule_label_weight,
                                       manual_label_weight).astype(np.float32)
        self.indices = np.flatnonzero(self.usable)
        self._store = self._rgb = self._pid = None
        digest = hashlib.sha256(Path(self.label_path).read_bytes())
        for array in (self.rgb_times, self.rgb_ends, self.force_times, self.wrench):
            digest.update(np.ascontiguousarray(array).tobytes())
        stat = Path(self.dataset_path).stat()
        digest.update(f'{self.dataset_path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        self.fingerprint = digest.hexdigest()

    def subset(self, episodes):
        result = copy.copy(self)
        result.indices = np.flatnonzero(self.usable & np.isin(self.frame_episode, episodes))
        result._store = result._rgb = result._pid = None
        return result

    def force_statistics(self, train_episodes):
        # Native F/T rows from train episodes only. Neither validation nor test
        # measurements participate in normalization.
        if self.ft_contract['mode'] != 'raw':
            total = torch.zeros(36, dtype=torch.float64)
            square = torch.zeros_like(total)
            count = 0
            prefix = self.ft_contract['prefix_samples']
            for ep in train_episodes:
                values = self.wrench[self.force_starts[ep]:self.force_ends[ep]]
                # Reset at the physical episode boundary, never at each RGB window.
                padded = np.pad(values, ((prefix, 0), (0, 0)), mode='edge')
                history = torch.from_numpy(padded).unsqueeze(0)
                features = torch.cat([causal_ft_features(history[..., axis],
                    mode=self.ft_contract['mode'], mean_window=self.ft_contract['mean_window'],
                    delta_lag=self.ft_contract['delta_lag'])
                    for axis in (slice(0, 6), slice(6, 12))], dim=-1).double().squeeze(0)
                total += features.sum(0)
                square += features.square().sum(0)
                count += len(features)
            if count == 0:
                raise ValueError('No training F/T samples for normalization')
            mean = total / count
            std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
            return mean.float().numpy(), std.float().numpy()
        values = np.concatenate([self.wrench[self.force_starts[e]:self.force_ends[e]]
                                 for e in train_episodes]).astype(np.float64)
        return values.mean(0).astype(np.float32), values.std(0).clip(1e-4).astype(np.float32)

    def history_indices(self, frame):
        ep = self.frame_episode[frame]
        rgb = np.maximum([frame - self.rgb_stride, frame], self.rgb_starts[ep])
        latest = self.force_index[frame]
        if latest < self.force_starts[ep]:
            raise ValueError('Anchor has no causal force measurement')
        force = np.maximum(latest - np.arange(self.required_ft_history - 1, -1, -1), self.force_starts[ep])
        return rgb, force

    def _get_rgb(self):
        if self._rgb is None or self._pid != os.getpid():
            self.close()
            self._store, root, _ = open_nested_zip_group(self.dataset_path, prefix=self.prefix)
            self._rgb = root['data/camera0_rgb']
            self._pid = os.getpid()
        return self._rgb

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame = self.indices[idx]
        rgb_idx, ft_idx = self.history_indices(frame)
        rgb_array = self._get_rgb()
        rgb = np.stack([rgb_array[int(i)] for i in rgb_idx]).transpose(0, 3, 1, 2).copy()
        force = self.wrench[ft_idx]
        return dict(obs=dict(camera0_rgb=torch.from_numpy(rgb).float().div_(255),
            robot0_ft_left=torch.from_numpy(force[:, :6].copy()),
            robot0_ft_right=torch.from_numpy(force[:, 6:].copy())),
            context_label=int(self.targets[frame]), context_weight=float(self.sample_weights[frame]),
            label_source=int(self.label_source[frame]), episode_id=int(self.frame_episode[frame]),
            timestamp=float(self.rgb_times[frame]), frame_index=int(frame))

    def close(self):
        if getattr(self, '_store', None) is not None:
            self._store.close()
        self._store = self._rgb = self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_store=None, _rgb=None, _pid=None)
        return state

    def __del__(self):
        self.close()


def canonical_episode_splits(count, seed, val_ratio, test_episode_numbers):
    """Reuse the observer's fixed, one-based test episodes; split the rest by episode."""
    test = sorted(int(e) - 1 for e in test_episode_numbers)
    if not test or len(set(test)) != len(test) or min(test) < 0 or max(test) >= count:
        raise ValueError('Invalid fixed test episodes (use one-based episode numbers)')
    remaining = np.setdiff1d(np.arange(count), test)
    if not 0 < val_ratio < 1 or len(remaining) < 2:
        raise ValueError('Need nonempty train and validation episode splits')
    order = np.random.default_rng(seed).permutation(remaining)
    nv = max(1, int(len(remaining) * val_ratio))
    return dict(train=sorted(order[nv:].tolist()), validation=sorted(order[:nv].tolist()), test=test)
