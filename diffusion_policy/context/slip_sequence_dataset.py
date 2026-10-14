"""Slip review labels, synchronized RGB/F/T windows, causal context history."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion_policy.context.canonical_dataset import CanonicalRGBForceDataset
from diffusion_policy.context.slip_timing import rgb_window_spec


def split_slip_episodes(dataset, seed=260910, validation_ratio=.15, test_ratio=.15):
    """Stratify whole episodes by error presence; never reuse the old EP numbers."""
    if not 0 < validation_ratio < .5 or not 0 < test_ratio < .5:
        raise ValueError('Validation and test ratios must be between zero and 0.5')
    rng = np.random.default_rng(seed)
    groups = {}
    for ep in range(len(dataset.rgb_ends)):
        ids = dataset.indices[dataset.frame_episode[dataset.indices] == ep]
        classes = set(dataset.targets[ids].tolist())
        if not classes:
            raise ValueError(f'Episode {ep + 1} has no eligible observation window')
        groups.setdefault((1 in classes, 2 in classes), []).append(ep)
    result = dict(train=[], validation=[], test=[])
    for key in sorted(groups):
        order = rng.permutation(groups[key]).tolist()
        n = len(order)
        nv = max(1, int(round(n * validation_ratio))) if n >= 3 else 0
        nt = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        result['validation'].extend(order[:nv])
        result['test'].extend(order[nv:nv + nt])
        result['train'].extend(order[nv + nt:])
    for name, episodes in result.items():
        episodes.sort()
        ids = dataset.indices[np.isin(dataset.frame_episode[dataset.indices], episodes)]
        if np.any(np.bincount(dataset.targets[ids], minlength=4) == 0):
            raise ValueError(f'All four classes must be represented in {name}')
    return result


class SlipSequenceDataset(CanonicalRGBForceDataset):
    """Past observation anchors at ~10 Hz followed by the current anchor.

    Every RGB has its own 41 raw F/T samples. No image or force row before
    the current contiguous valid-label run is admitted, including at startup.
    Labels are returned only as loss targets, never in the observation dict.
    """
    def __init__(self, dataset_path, force_sidecar_path, label_path,
                 history_steps=20, history_stride=6, anchor_stride=6,
                 max_force_age_s=.012, rgb_horizon=4, rgb_span_s=None):
        if not 0 < max_force_age_s <= .012:
            raise ValueError('Require a positive F/T age limit of at most 12 ms')
        super().__init__(dataset_path, force_sidecar_path, label_path,
                         rgb_horizon=rgb_horizon, rgb_stride=10, ft_feature_mode='raw_mean_delta',
                         label_contract='slip', max_force_age_s=max_force_age_s)
        self.rgb_span_s, self.rgb_offsets = rgb_window_spec(self.rgb_horizon, rgb_span_s)
        if history_steps < 1 or history_stride != 6 or anchor_stride != 6:
            raise ValueError('Use positive history_steps and 6-frame (~10 Hz) cadence')
        self.history_steps, self.history_stride = int(history_steps), int(history_stride)
        self.anchor_stride = int(anchor_stride)
        n = len(self.rgb_times)
        self.run_starts = np.empty(n, np.int64)
        sensor_ok = np.zeros(n, bool)
        for rs, re, fs, fe in zip(self.rgb_starts, self.rgb_ends, self.force_starts, self.force_ends):
            frames = np.arange(rs, re)
            run = np.maximum.accumulate(np.where(self.valid[rs:re], rs, frames + 1))
            self.run_starts[rs:re] = run
            latest = self.force_index[rs:re]
            begin = latest - 40
            safe_latest = np.clip(latest, fs, fe - 1)
            safe_begin = np.clip(begin, fs, fe - 1)
            bad = np.r_[0, np.cumsum(np.diff(self.force_times[fs:fe]) > .025)]
            span = self.force_times[safe_latest] - self.force_times[safe_begin]
            sensor_ok[rs:re] = (self.valid[rs:re] & self.force_available[rs:re]
                & (begin >= fs) & (span >= .3) & (span <= .5)
                & (bad[safe_latest - fs] == bad[safe_begin - fs])
                & (self.force_times[safe_begin] >= self.rgb_times[np.minimum(run, re - 1)]))
        rgb = np.arange(n)[:, None] - self.rgb_offsets
        safe_rgb = np.maximum(rgb, 0)
        eligible = ((rgb[:, 0] >= self.run_starts) & sensor_ok[safe_rgb].all(1))
        expected = self.rgb_times[:, None] - np.linspace(self.rgb_span_s, 0, self.rgb_horizon)
        eligible &= (np.abs(self.rgb_times[safe_rgb] - expected).max(1) <= .025)
        self.window_usable = eligible
        on_grid = (np.arange(n) - self.rgb_starts[self.frame_episode]) % self.anchor_stride == 0
        self.usable = eligible & on_grid
        self.indices = np.flatnonzero(self.usable)
        self.feature_path = None
        self._features = None

    def attach_features(self, path):
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != (len(self.rgb_times), 768) or array.dtype != np.float32:
            raise ValueError('Expected CLIP cache [num_rgb_frames,768] float32')
        self.feature_path = str(Path(path).resolve())
        self._features = None

    def subset(self, episodes):
        result = super().subset(episodes)
        result._features = None
        return result

    def sequence_indices(self, frame):
        if not self.usable[frame]:
            raise ValueError('Anchor is not on the usable 10 Hz observation grid')
        anchors = frame - np.arange(self.history_steps, -1, -1) * self.history_stride
        safe = np.maximum(anchors, 0)
        valid = ((anchors >= self.run_starts[frame]) & self.window_usable[safe])
        # After a sensor/window failure, do not reuse history from before it.
        invalid = np.flatnonzero(~valid)
        if len(invalid):
            valid[:invalid[-1] + 1] = False
        safe = np.where(valid, anchors, frame)
        rgb = safe[:, None] - self.rgb_offsets
        force = self.force_index[rgb][..., None] - np.arange(40, -1, -1)
        ages = (self.rgb_times[frame] - self.rgb_times[safe]).astype(np.float32)
        ages[~valid] = 0
        return safe, rgb, force, valid, ages

    def history_indices(self, frame):
        """Current RGB indices and each RGB's own native force window."""
        _, rgb, force, _, _ = self.sequence_indices(frame)
        return rgb[-1], force[-1]

    def __getitem__(self, index):
        if self.feature_path is None:
            raise ValueError('Prepare and attach the frozen CLIP feature cache before training')
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode='r', allow_pickle=False)
        frame = int(self.indices[index])
        anchors, rgb, force, valid, ages = self.sequence_indices(frame)
        visual = np.asarray(self._features[rgb]).copy()
        wrench = self.wrench[force].copy()
        visual[~valid], wrench[~valid] = 0, 0
        targets = self.targets[anchors].copy()
        targets[~valid] = -100
        return dict(obs=dict(rgb_features=torch.from_numpy(visual),
                            robot0_ft_left=torch.from_numpy(wrench[..., :6].copy()),
                            robot0_ft_right=torch.from_numpy(wrench[..., 6:].copy()),
                            sequence_valid=torch.from_numpy(valid),
                            sequence_ages=torch.from_numpy(ages)),
                    sequence_labels=torch.from_numpy(targets),
                    context_label=int(self.targets[frame]), frame_index=frame,
                    episode_id=int(self.frame_episode[frame]), timestamp=float(self.rgb_times[frame]))

    def force_statistics(self, train_episodes):
        """Fit only sensor windows actually eligible in train episodes, excluding ignore."""
        anchors = self.indices[np.isin(self.frame_episode[self.indices], train_episodes)]
        rgb = np.unique((anchors[:, None] - self.rgb_offsets).ravel())
        total, squares, count = torch.zeros(36, dtype=torch.float64), torch.zeros(36, dtype=torch.float64), 0
        for start in range(0, len(rgb), 512):
            ids = self.force_index[rgb[start:start + 512], None] - np.arange(40, -1, -1)
            windows = torch.from_numpy(self.wrench[ids])
            from diffusion_policy.context.ft_features import causal_ft_features
            features = torch.cat([causal_ft_features(windows[..., side], mode='raw_mean_delta')
                                  for side in (slice(0, 6), slice(6, 12))], -1).double().reshape(-1, 36)
            total += features.sum(0)
            squares += features.square().sum(0)
            count += len(features)
        if not count:
            raise ValueError('No eligible training force windows')
        mean = total / count
        return mean.float().numpy(), (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4).float().numpy()

    def __getstate__(self):
        state = super().__getstate__()
        state['_features'] = None
        return state
