#!/usr/bin/env python3
"""Offline trajectory evaluation of the RGB4/paired-F/T/no-TCP slip policy.

Uses recorded observations, frozen cached context, checkpoint normalizers and
EMA weights by default. Does not connect to robot or sensor interfaces.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from diffusion_policy.dataset.slip_action_dataset import INPUT_KEYS, PHASE_NAMES
from diffusion_policy.common.pytorch_util import dict_apply
from eval_dual_ft_offline import (
    _stable_sha256, _require_file_identity, _select_state_name,
    disable_stochastic_image_transforms,
)
from umi.common.pose_util import rot6d_to_mat


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ('outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/'
                      'epoch=0008-val_loss=0.02815.ckpt')
CONTRACT = 'slip_rgb4_paired_ft_no_tcp_768_action11_v1'
METRICS = ('position_l2_mm', 'rotation_deg', 'gripper_width_abs_mm', 'force_abs_N')


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_normalizer(policy):
    expected = dict(camera0_rgb=1, robot0_ft_left=6, robot0_ft_right=6,
                    valve_context=1, action=11)
    params = policy.normalizer.params_dict
    if set(params) != set(expected):
        raise ValueError('Checkpoint normalizer fields do not match pose-free slip inputs')
    for key, dimension in expected.items():
        for field in ('scale', 'offset'):
            value = params[key][field]
            if tuple(value.shape) != (dimension,) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint normalizer {key}.{field}')
        if (params[key]['scale'] <= 0).any():
            raise ValueError(f'Nonpositive checkpoint normalizer scale: {key}')
    for key in ('camera0_rgb', 'valve_context'):
        if not torch.equal(params[key]['scale'], torch.ones_like(params[key]['scale'])) or not torch.equal(
                params[key]['offset'], torch.zeros_like(params[key]['offset'])):
            raise ValueError(f'{key} must retain the trained identity normalization')


def load_policy(checkpoint, device, weights='ema_model'):
    # Locally produced torch/dill checkpoint; no workspace, optimizer, or pretrained download.
    payload = torch.load(str(checkpoint), map_location='cpu', pickle_module=dill, mmap=True)
    cfg = OmegaConf.create(OmegaConf.to_container(payload['cfg'], resolve=True))
    if cfg.task.model_contract.version != CONTRACT or set(cfg.shape_meta.obs) != set(INPUT_KEYS):
        raise ValueError('Expected RGB4/paired-F/T/no-TCP slip action checkpoint')
    if tuple(cfg.task.model_contract.valve_context_phase_names) != PHASE_NAMES:
        raise ValueError('Checkpoint context class order is incompatible')
    targets = {
        'policy._target_': 'diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy',
        'policy.obs_encoder._target_': 'diffusion_policy.model.vision.dual_ft_obs_encoder.DualFTObsEncoder',
        'policy.noise_scheduler._target_': 'diffusers.DDIMScheduler',
        'task.dataset._target_': 'diffusion_policy.dataset.slip_action_dataset.SlipActionDataset',
    }
    for key, target in targets.items():
        if OmegaConf.select(cfg, key) != target:
            raise ValueError(f'Unsupported checkpoint implementation: {key}')
    state_name = _select_state_name(payload, weights, bool(cfg.training.use_ema))
    state = payload['state_dicts'][state_name]
    epoch = int(dill.loads(payload['pickles']['epoch'])) + 1
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy_cfg['obs_encoder'].update(pretrained=False, transforms=None,
                                      allow_frozen_without_pretrained=True)
    policy = hydra.utils.instantiate(policy_cfg)
    policy.load_state_dict(state, strict=True)
    disabled = disable_stochastic_image_transforms(policy)
    validate_normalizer(policy)
    enc = policy.obs_encoder
    if (policy.obs_feature_dim != 768 or policy.action_horizon != 16 or policy.action_dim != 11
            or enc.low_dim_output_dim != 0 or enc.num_fusion_tokens != 12
            or enc.ft_window_mode != 'per_rgb' or enc.ft_feature_mode != 'raw_mean_delta'
            or int(enc.architecture_contract_version) != 5):
        raise ValueError('Restored model does not implement the saved slip input/action contract')
    policy.requires_grad_(False).eval().to(device)
    return cfg, policy, dict(epoch_1based=epoch, weights=state_name,
                            disabled_image_transforms=disabled)


def validate_dataset_binding(expected, dataset):
    actual = dataset.action_training_metadata
    # Paths may be re-rooted; data identity, class semantics and episode split may not.
    for key in ('schema', 'phase_names', 'source_fingerprint', 'checkpoint_sha256',
                'episode_splits', 'model_config', 'observation_contract',
                'history_source', 'prediction_source', 'context_cadence_frames'):
        if key not in expected or actual.get(key) != expected[key]:
            raise ValueError(f'Action checkpoint/context cache mismatch: {key}')


def load_dataset(cfg, split, context_cache=None, context_checkpoint=None):
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    for key in ('dataset_path', 'force_sidecar_path', 'label_path',
                'context_cache_path', 'context_checkpoint_path'):
        override = context_cache if key == 'context_cache_path' else (
            context_checkpoint if key == 'context_checkpoint_path' else None)
        ds_cfg[key] = str(resolve_path(override or ds_cfg[key]))
    ds_cfg['split'] = split
    dataset = hydra.utils.instantiate(ds_cfg)
    expected = OmegaConf.to_container(cfg.task.model_contract.context_training_metadata, resolve=True)
    validate_dataset_binding(expected, dataset)
    return dataset


def select_indices(dataset, anchor_stride=6, episode=None, max_samples=None):
    ds = dataset.recording
    frames = dataset.indices
    selected = (frames - ds.rgb_starts[ds.frame_episode[frames]]) % anchor_stride == 0
    if episode is not None:
        selected &= ds.original_episode_numbers[ds.frame_episode[frames]] == episode
    ids = np.flatnonzero(selected)
    before_limit = len(ids)
    if not len(ids):
        raise ValueError('No eligible anchors: check the original episode number and selected split')
    if max_samples is not None and len(ids) > max_samples:
        # Cover the requested split instead of silently evaluating its first episode only.
        ids = ids[np.linspace(0, len(ids) - 1, max_samples).round().astype(np.int64)]
    return ids, before_limit


def action_errors(prediction, target):
    """Per-repeat/per-anchor/per-step errors, never mix physical units.

    A degenerate predicted 6D rotation is counted separately, not silently
    treated as a valid angle. Ground-truth rotations must be valid.
    """
    pred, gt = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if pred.shape != gt.shape or pred.shape[-2:] != (16, 11):
        raise ValueError('Prediction and target must have matching [...,16,11] shapes')
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite predicted or demonstrated action')
    def rotations(d6):
        first, second = d6[..., :3], d6[..., 3:]
        norm = np.linalg.norm(first, axis=-1)
        unit = first / np.maximum(norm[..., None], 1e-12)
        orthogonal = second - np.sum(second * unit, axis=-1, keepdims=True) * unit
        valid = (norm > 1e-7) & (np.linalg.norm(orthogonal, axis=-1) > 1e-7)
        # The repository's rotation helper accepts a flat batch, not arbitrary
        # [repeat, anchor, horizon] leading dimensions.
        matrices = rot6d_to_mat(d6.reshape(-1, 6)).reshape(*d6.shape[:-1], 3, 3)
        return matrices, valid
    pr, valid = rotations(pred[..., 3:9]); gr, gt_valid = rotations(gt[..., 3:9])
    if not gt_valid.all():
        raise ValueError('Invalid demonstrated rotation')
    relative = pr @ np.swapaxes(gr, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    angle = np.degrees(np.arccos(cosine))
    angle[~valid] = np.nan
    return dict(position_l2_mm=np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1) * 1000,
                rotation_deg=angle, gripper_width_abs_mm=np.abs(pred[..., 9] - gt[..., 9]) * 1000,
                force_abs_N=np.abs(pred[..., 10] - gt[..., 10]), invalid_rotation=~valid)


def summarize_errors(errors, mask=None, steps=slice(1, 16)):
    # Arrays are [repeats, anchors, 16]. Step zero is current time, not a future forecast.
    mask = np.ones(errors['position_l2_mm'].shape[1], bool) if mask is None else mask
    result = dict(anchors=int(np.count_nonzero(mask)), steps=np.arange(16)[steps].tolist(),
                  prediction_repeats=errors['position_l2_mm'].shape[0],
                  invalid_rotation_count=int(errors['invalid_rotation'][:, mask, steps].sum()))
    for key in METRICS:
        values = errors[key][:, mask, steps].reshape(-1)
        values = values[np.isfinite(values)]
        result[key] = dict(count=len(values), mean=float(np.mean(values)) if len(values) else None,
                           rmse=float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
                           median=float(np.median(values)) if len(values) else None,
                           p95=float(np.quantile(values, .95)) if len(values) else None)
    return result


@torch.inference_mode()
def evaluate_predictions(policy, loader, device, repeats=1, seed=42):
    predictions = [[] for _ in range(repeats)]
    targets, times = [], []
    loss_total, sample_count = 0., 0
    for batch_index, batch in enumerate(loader):
        batch = dict_apply(batch, lambda t: t.to(device, non_blocking=True))
        if set(batch['obs']) != set(INPUT_KEYS):
            raise ValueError('Unexpected evaluation inputs (TCP/manual labels must not enter obs)')
        target = batch['action']; targets.append(target.cpu().numpy())
        torch.manual_seed(seed + 2000003 + batch_index)
        loss = policy.compute_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite diffusion loss')
        loss_total += float(loss) * len(target); sample_count += len(target)
        for repeat in range(repeats):
            torch.manual_seed(seed + 1000003 * repeat + batch_index)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            start = time.perf_counter()
            pred = policy.predict_action(batch['obs'])['action_pred']
            if device.type == 'cuda': torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise ValueError('Invalid action prediction')
            predictions[repeat].append(pred.cpu().numpy())
        if batch_index % 20 == 0 or batch_index + 1 == len(loader):
            print(json.dumps(dict(event='eval', batch=batch_index + 1, batches=len(loader),
                                  samples=sample_count)), flush=True)
    return np.stack([np.concatenate(p) for p in predictions]), np.concatenate(targets), dict(
        diffusion_loss_no_augmentation=loss_total / sample_count,
        policy_batch_latency_ms=dict(mean=float(np.mean(times)), p95=float(np.quantile(times, .95))))


def save_reports(out, metadata, prediction, target, frames, dataset):
    ds = dataset.recording
    episodes = ds.frame_episode[frames]
    originals = ds.original_episode_numbers[episodes]
    labels = ds.targets[frames]
    gt = np.broadcast_to(target[None], prediction.shape)
    errors = action_errors(prediction, gt)
    hold = np.broadcast_to(target[None, :, :1], prediction.shape)
    baseline = action_errors(hold, gt)
    phases = {name: summarize_errors(errors, labels == i) for i, name in enumerate(PHASE_NAMES)}
    per_episode = {str(int(ep)): summarize_errors(errors, originals == ep) for ep in np.unique(originals)}
    horizons = [summarize_errors(errors, steps=slice(h, h + 1)) for h in range(16)]
    baseline_horizons = [summarize_errors(baseline, steps=slice(h, h + 1)) for h in range(16)]
    summary = dict(metadata, evaluated_samples=len(frames), evaluated_episodes=len(per_episode),
        original_episode_numbers=np.unique(originals).tolist(), phase_names=list(PHASE_NAMES),
        future_steps_1_to_15=summarize_errors(errors), current_step_0=summarize_errors(errors, steps=slice(0, 1)),
        first_future_step=summarize_errors(errors, steps=slice(1, 2)),
        hold_current_demo_baseline=summarize_errors(baseline), per_phase=phases, per_episode=per_episode,
        limits=['Offline recorded-observation replay; not robot task success or closed-loop recovery.',
                'Context predictions are cached; timing excludes classifier and sensor I/O.',
                'Hold baseline uses current demonstrated width and force; it has access to those measurements.',
                'Reported diffusion loss has augmentation disabled; it is not the logged training val_loss.',
                'Overlapping action windows are correlated; sample counts are not independent trials.'])
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'predictions.npz', predicted_actions=prediction, target_actions=target,
        frame_indices=frames, episode_indices=episodes, original_episode_numbers=originals,
        timestamps=ds.rgb_times[frames], phase_ids=labels,
        context_values=dataset.context[frames], context_source_frames=dataset.context_source[frames],
        horizon_seconds=np.arange(16) * 3 / (60000 / 1001))
    for name, mapping in (('per_phase', phases), ('per_episode', per_episode)):
        with (out / (name + '.csv')).open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow([name, 'anchors', *[k + '_mean' for k in METRICS]])
            for key, item in mapping.items(): writer.writerow([key, item['anchors'], *[item[k]['mean'] for k in METRICS]])
    with (out / 'per_horizon.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['step', 'seconds', *[k + '_mean' for k in METRICS],
                                                *['hold_' + k + '_mean' for k in METRICS]])
        for h in range(16):
            writer.writerow([h, h * 3 / (60000 / 1001), *[horizons[h][k]['mean'] for k in METRICS],
                             *[baseline_horizons[h][k]['mean'] for k in METRICS]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.flat, METRICS):
        x = np.arange(1, 16) * 3 / (60000 / 1001)
        ax.plot(x, [r[key]['mean'] for r in horizons[1:]], 'o-', label='Policy')
        ax.plot(x, [r[key]['mean'] for r in baseline_horizons[1:]], '--', color='gray', label='Hold current demo')
        ax.set(title=key, xlabel='Prediction horizon (seconds)', ylabel='Mean error')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f"Slip action: epoch {metadata['epoch_1based']} / {metadata['weights']} / {metadata['split']}")
    fig.tight_layout(); fig.savefig(out / 'errors_by_horizon.png', dpi=150); plt.close(fig)
    lines = [f"# Slip action evaluation — epoch {metadata['epoch_1based']}", '',
        f"- Split: {metadata['split']}; {len(per_episode)} episodes, {len(frames)} anchors.",
        f"- Weights: {metadata['weights']}; no image augmentation; saved normalizer only.",
        f"- Anchor stride: {metadata['anchor_stride']} source RGB frames; repeats: {prediction.shape[0]}.",
        '- Main metrics cover future steps 1–15 (~0.05–0.75 s); current step 0 is separate.', '',
        '| Anchor context | Anchors | Position L2 mm | Rotation deg | Width MAE mm | Force MAE N |',
        '|---|---:|---:|---:|---:|---:|']
    for key, row in [('all', summary['future_steps_1_to_15']), *phases.items()]:
        vals = ['n/a' if row[k]['mean'] is None else f"{row[k]['mean']:.4f}" for k in METRICS]
        lines.append(f"| {key} | {row['anchors']} | " + ' | '.join(vals) + ' |')
    lines += ['', '![Errors by horizon](errors_by_horizon.png)', '', *['- ' + s for s in summary['limits']], '']
    (out / 'report.md').write_text('\n'.join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--weights', choices=('ema_model', 'model'), default='ema_model')
    parser.add_argument('--split', choices=('test', 'validation'), default='test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', help='Must be a new directory; defaults to checkpoint run/eval_epoch9_test_timestamp')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--anchor-stride', type=int, default=6, help='6 = ~10 Hz; 1 = all eligible RGB anchors')
    parser.add_argument('--episode', type=int, help='Original reviewed episode number, e.g. 291')
    parser.add_argument('--max-samples', type=int, help='Optional evenly spread smoke subset; omitted means full selected grid')
    parser.add_argument('--prediction-repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-cache')
    parser.add_argument('--context-checkpoint')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.anchor_stride, args.prediction_repeats) < 1 or args.num_workers < 0:
        parser.error('Batch size, anchor stride and repeats must be positive; workers nonnegative')
    if args.max_samples is not None and args.max_samples < 1: parser.error('max-samples must be positive')
    checkpoint = resolve_path(args.checkpoint)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    if not OmegaConf.has_resolver('eval'): OmegaConf.register_new_resolver('eval', eval)
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    checkpoint_sha, identity = _stable_sha256(checkpoint)
    cfg, policy, metadata = load_policy(checkpoint, device, args.weights)
    dataset = load_dataset(cfg, args.split, args.context_cache, args.context_checkpoint)
    try:
        selected, grid_count = select_indices(dataset, args.anchor_stride, args.episode, args.max_samples)
        out = resolve_path(args.output) if args.output else checkpoint.parent.parent / (
            f"eval_epoch{metadata['epoch_1based']}_{args.split}_" + time.strftime('%Y%m%d_%H%M%S'))
        out.mkdir(parents=True, exist_ok=False)
        metadata.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, split=args.split,
            seed=args.seed, batch_size=args.batch_size, anchor_stride=args.anchor_stride,
            eligible_split_anchors=len(dataset), eligible_grid_anchors=grid_count,
            partial_evaluation=len(selected) < grid_count, requested_episode=args.episode,
            device=str(device), inference_steps=policy.num_inference_steps,
            context_checkpoint_sha256=dataset.action_training_metadata['checkpoint_sha256'],
            source_fingerprint=dataset.recording.fingerprint)
        (out / 'run_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
        loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        prediction, target, result = evaluate_predictions(policy, loader, device, args.prediction_repeats, args.seed)
        _require_file_identity(checkpoint, identity)
        metadata.update(result)
        summary = save_reports(out, metadata, prediction, target, dataset.indices[selected], dataset)
        print(json.dumps(dict(output=str(out), samples=summary['evaluated_samples'],
                              episodes=summary['evaluated_episodes'],
                              metrics=summary['future_steps_1_to_15']), indent=2), flush=True)
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
