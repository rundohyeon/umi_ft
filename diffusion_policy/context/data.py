"""Reuse the audited UMI multirate reader; label indices are episode-local RGB rows."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from diffusion_policy.dataset.umi_dual_ft_dataset import UmiDualFTDataset
from diffusion_policy.context.labels import LabelIndex, read_rows


def load_base(config_name='train_diffusion_unet_timm_umi_dual_ft_workspace', overrides=()):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'config')):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return instantiate(cfg.task.dataset), cfg


def load_labeling_base(cfg, overrides=()):
    """Label normal datasets or a prepared clip whose SLAM pose is unavailable."""
    if cfg.get('recording_path'):
        if overrides or cfg.get('dataset_overrides'):
            raise ValueError('Dataset overrides cannot be applied to a raw labeling clip')
        inputs = cfg.get('inputs', {})
        if inputs.get('tcp_position', False) or inputs.get('tcp_rotation', True):
            raise ValueError('A raw labeling clip has no TCP pose; disable TCP inputs explicitly')
        from diffusion_policy.context.raw_recording import RawLabelRecording
        return RawLabelRecording(cfg['recording_path']), None
    return load_base(cfg['policy_config'], cfg.get('dataset_overrides', [])+list(overrides))


def split_episodes(count, seed=42, val_ratio=.15, test_ratio=.15):
    if count < 3 or val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError('Need at least three episodes and nonempty train/validation/test fractions')
    order = np.random.default_rng(seed).permutation(count)
    nv, nt = max(1, int(count * val_ratio)), max(1, int(count * test_ratio))
    if nv + nt >= count:
        raise ValueError('Split leaves no training episodes')
    return {'validation': sorted(order[:nv].tolist()), 'test': sorted(order[nv:nv+nt].tolist()),
            'train': sorted(order[nv+nt:].tolist())}


def source_fingerprint(base):
    values = []
    for name in ('dataset_path', 'force_sidecar_path'):
        source = getattr(base, name, None)
        if source is None:
            continue
        path = Path(source)
        if path.exists():
            values.append((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns))
    digest = hashlib.sha256(json.dumps(values).encode())
    for a in (base.rgb_timestamps, base.rgb_episode_ends, base.ft_left_timestamps, base.ft_right_timestamps):
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def episode_bounds(base, episode):
    ends = base.rgb_episode_ends
    if not 0 <= episode < len(ends):
        raise ValueError('Unknown episode')
    return (0 if episode == 0 else int(ends[episode-1])), int(ends[episode])


def validate_label_alignment(base, rows):
    for r in rows:
        start, end = episode_bounds(base, int(r['episode_id']))
        index = int(r['anchor_index'])
        if not 0 <= index < end-start or abs(float(r['timestamp']) - base.rgb_timestamps[start+index]) > 1e-6:
            raise ValueError('Label index/timestamp does not match RGB/state anchor')


def assert_causal(info):
    anchor = np.asarray(info['anchor_timestamp'])
    for key in ('rgb_timestamps', 'pose_timestamps', 'left_ft_timestamps', 'right_ft_timestamps'):
        t = np.asarray(info[key])
        if not np.isfinite(t).all() or np.any(t > np.expand_dims(anchor, -1)):
            raise ValueError(f'Future or invalid observations: {key}')
        if np.any(np.diff(t, axis=-1) < 0):
            raise ValueError(f'Unordered history: {key}')


def frame_view(base, episodes=None):
    """All causal anchors, including episode tails; action targets are never context inputs."""
    result = copy.copy(base)
    result.action_padding = True
    mask = np.ones(len(base.rgb_episode_ends), dtype=bool)
    if episodes is not None:
        mask[:] = False
        mask[list(episodes)] = True
    result.indices = result._build_indices(mask)
    result.split = 'validation'
    result._zip_store = result._zarr_root = result._rgb_array = result._open_pid = None
    return result


def force_history(values, times, anchor, bins):
    """Keep only force magnitudes and their native times, including brief extrema."""
    if type(bins) is not int or bins < 1:
        raise ValueError('Force history bins must be a positive integer')
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if (values.shape != (len(times), 6) or not np.isfinite(values).all()
            or not np.isfinite(times).all() or np.any(times > anchor)
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid or future force history input')
    result = dict(available=bool(len(times)), source_samples=len(times),
        selection='First, last, and force-norm minimum/maximum in each consecutive sample bin',
        columns=['offset_s', 'force_norm_N'],
        samples=[], last_sample_age_s=None)
    if not len(times):
        return result
    norms = np.linalg.norm(values[:, :3], axis=-1)
    selected = {0, len(times)-1}
    for group in np.array_split(np.arange(len(times)), min(bins, len(times))):
        selected.add(int(group[np.argmin(norms[group])]))
        selected.add(int(group[np.argmax(norms[group])]))
    selected = np.array(sorted(selected))
    result['samples'] = np.column_stack((times[selected]-anchor, norms[selected])).round(6).tolist()
    result['last_sample_age_s'] = round(float(anchor-times[-1]), 6)
    return result


def causal_summary(base, episode, local_index, history_length=32, inputs=None):
    inputs = inputs or {}
    if history_length < 1:
        raise ValueError('history_length must be positive')
    force_bins = inputs.get('force_history_bins', 0)
    if type(force_bins) is not int or force_bins < 0:
        raise ValueError('Force history bins must be a nonnegative integer')
    start, end = episode_bounds(base, episode)
    current = start + local_index
    if not start <= current < end:
        raise ValueError('Anchor outside episode')
    first = max(start, current - history_length + 1)
    ts = base.rgb_timestamps[first:current+1]
    summary = {'duration_s': float(ts[-1]-ts[0]), 'samples': len(ts), 'signals': {}}
    def statistics(name, values, times):
        x = np.asarray(values, dtype=np.float64)
        if not len(x):
            return
        if not np.isfinite(x).all() or np.any(times > ts[-1]):
            raise ValueError('Invalid or future summary input')
        dt = np.diff(times)
        entry = {key: np.round(fn(x, axis=0), 5).tolist() for key, fn in
                 [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]}
        entry.update(current=x[-1].round(5).tolist(), delta=(x[-1]-x[0]).round(5).tolist())
        if len(x) > 1 and np.all(dt > 0):
            velocity = np.diff(x, axis=0) / dt[:, None]
            entry['mean_velocity'] = velocity.mean(0).round(5).tolist()
            if len(velocity) > 1:
                entry['mean_acceleration'] = (np.diff(velocity, axis=0) / ((dt[1:]+dt[:-1])/2)[:, None]).mean(0).round(5).tolist()
        summary['signals'][name] = entry
    if inputs.get('tcp_position', False):
        statistics('eef_position_m', base.pose_mats[first:current+1, :3, 3], ts)
    if inputs.get('tcp_rotation', True):
        from scipy.spatial.transform import Rotation
        rotations = np.asarray(base.pose_mats[first:current+1, :3, :3])
        if not np.isfinite(rotations).all():
            raise ValueError('Invalid TCP rotation input')
        relative = Rotation.from_matrix(rotations[0].T @ rotations).as_rotvec()
        selected = np.unique(np.linspace(0, len(ts)-1, min(8, len(ts)), dtype=int))
        summary['signals']['tcp_rotation'] = dict(
            current_matrix=rotations[-1].round(5).tolist(),
            relative_rotvec_rad=relative[selected].round(5).tolist(),
            sample_offsets_s=(ts[selected]-ts[-1]).round(6).tolist())
        if len(ts)>1:
            increments = Rotation.from_matrix(np.swapaxes(rotations[:-1],1,2) @ rotations[1:]).as_rotvec()
            statistics('tcp_angular_velocity_rad_s', increments / np.diff(ts)[:,None], ts[1:])
    if inputs.get('gripper_width', True):
        statistics('gripper_width_m', base.gripper_width[first:current+1], ts)
    for side in (('left', 'right') if inputs.get('force_torque', True) else ()):
        times = getattr(base, f'ft_{side}_timestamps')
        ends = getattr(base, f'ft_{side}_episode_ends')
        lo, hi = (0 if episode == 0 else int(ends[episode-1])), int(ends[episode])
        selected = np.arange(lo, hi)[(times[lo:hi] >= ts[0]) & (times[lo:hi] <= ts[-1])]
        x = getattr(base, f'ft_{side}')[selected]
        # Qwen receives the force magnitude only, never signed axes or torques.
        statistics(f'{side}_force_norm_N', np.linalg.norm(x[:,:3], axis=-1)[:,None], times[selected])
        if side == 'right' and force_bins:
            summary['signals']['right_force_history'] = force_history(x, times[selected], ts[-1], force_bins)
    # Stored future pose/width/force targets are not executed command history.
    return summary, first-start, local_index


def causal_rgb(base, episode, start_index, end_index, count=4):
    """Native RGB frames in chronological order, ending at the current anchor."""
    lo,hi=episode_bounds(base,episode)
    if count<1 or not 0<=start_index<=end_index<hi-lo:
        raise ValueError('Invalid causal RGB window')
    indices=np.unique(np.linspace(start_index,end_index,min(count,end_index-start_index+1),dtype=int))
    rgb=base._get_rgb_array()
    images=[np.array(rgb[lo+int(index)],copy=True) for index in indices]
    anchor_time=base.rgb_timestamps[lo+end_index]
    metadata=[dict(anchor_index=int(index),offset_s=float(base.rgb_timestamps[lo+index]-anchor_time)) for index in indices]
    return images,metadata


class ContextUmiDataset(UmiDualFTDataset):
    def __init__(self, context_labels, split_seed=42, test_ratio=.15, context_val_ratio=.15,
                 context_split_path=None, require_known_context=False, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(context_labels)
        num_classes = cfg.get('num_classes', 5)
        auto = read_rows(cfg.pop('auto_path', 'data/context_labels/auto_labels.jsonl'), num_classes)
        reviewed = read_rows(cfg.pop('reviewed_path', 'data/context_labels/reviewed_labels.parquet'), num_classes)
        validate_label_alignment(self, auto + reviewed)
        self.label_index = LabelIndex(auto, reviewed, **cfg)
        self.require_known_context = bool(require_known_context)
        if context_split_path:
            saved = json.loads(Path(context_split_path).read_text())
            if saved['source_fingerprint'] != source_fingerprint(self):
                raise ValueError('Stage A split belongs to a different dataset')
            if saved.get('num_classes', 5) != num_classes:
                raise ValueError('Stage A split has a different class count; regenerate labels and retrain the context encoder')
            self.context_splits = saved['splits']
            if saved.get('context_definition_hash'):
                from diffusion_policy.context.labels import validate_definition_version
                validate_definition_version(auto + reviewed, saved['context_definition_hash'])
        else:
            self.context_splits = split_episodes(len(self.rgb_episode_ends), split_seed, context_val_ratio, test_ratio)
        all_ids = sum(self.context_splits.values(), [])
        if sorted(all_ids) != list(range(len(self.rgb_episode_ends))):
            raise ValueError('Episode splits overlap or omit episodes')
        self.train_mask = np.isin(np.arange(len(self.rgb_episode_ends)), self.context_splits['train'])
        self.val_mask = np.isin(np.arange(len(self.rgb_episode_ends)), self.context_splits['validation'])
        self.indices = self._build_indices(self.train_mask)

    def _build_indices(self, episode_mask):
        indices = super()._build_indices(episode_mask)
        if getattr(self, 'require_known_context', False):
            indices = [(ep, current) for ep, current in indices
                       if self.label_index.resolve(ep, current-episode_bounds(self, ep)[0])[0] >= 0]
        return indices

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        assert_causal(sample['sample_info'])
        episode, current = self.indices[idx]
        start, _ = episode_bounds(self, episode)
        label, weight, _ = self.label_index.resolve(episode, current-start)
        sample['context_label'] = torch.tensor(label, dtype=torch.long)
        sample['context_weight'] = torch.tensor(weight, dtype=torch.float32)
        return sample


class ContextFrames(torch.utils.data.Dataset):
    def __init__(self, base, episodes, labels):
        self.base = frame_view(base, episodes)
        self.labels = labels
        self.entries = []
        for i, (ep, current) in enumerate(self.base.indices):
            start, _ = episode_bounds(base, ep)
            label, weight, provenance = labels.resolve(ep, current-start)
            if label >= 0 and weight > 0:
                self.entries.append((i, label, weight, ep, current-start, provenance))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        i, label, weight, ep, anchor, _ = self.entries[index]
        obs, _, info = self.base._sample_arrays(i, load_rgb=True)
        assert_causal(info)
        return dict(obs={k: torch.from_numpy(v) for k,v in obs.items()}, context_label=torch.tensor(label),
                    context_weight=torch.tensor(weight, dtype=torch.float32), episode_id=ep, anchor_index=anchor,
                    timestamp=float(info['anchor_timestamp']))


def protect_raw_data(base, output):
    output = Path(output).resolve()
    for key in ('dataset_path', 'force_sidecar_path'):
        source = getattr(base, key, None)
        if source is None:
            continue
        source = Path(source).resolve()
        if output == source or (source.is_dir() and source in output.parents):
            raise ValueError('Derived outputs must be outside raw dataset stores')
