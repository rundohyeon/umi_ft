"""Read-only slip context inference; consumes preprocessed RGB and native F/T."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from models.slip_temporal_context import SlipTemporalContextEncoder, PHASE_NAMES
from diffusion_policy.context.slip_timing import rgb_window_spec
from umi.real_world.rgb_force_context import ObservationUnavailable


def select_slip_window(rgb_times, ft_times, anchor, task_start=-np.inf, max_force_age_s=.012,
                       *, rgb_horizon=4, rgb_span_s=None):
    span_s, _ = rgb_window_spec(rgb_horizon, rgb_span_s)
    rt, ft = np.asarray(rgb_times, np.float64), np.asarray(ft_times, np.float64)
    if (rt.ndim != 1 or ft.ndim != 1 or not np.isfinite(rt).all() or not np.isfinite(ft).all()
            or np.any(np.diff(rt) <= 0) or np.any(np.diff(ft) <= 0) or not np.isfinite(anchor)):
        raise ValueError('Provide ordered finite sensor timestamps from one episode')
    available = np.flatnonzero((rt >= task_start) & (rt <= anchor))
    if len(available) < rgb_horizon or abs(rt[available[-1]] - anchor) > 1e-6:
        raise ObservationUnavailable('warming_up_or_missing_rgb_anchor')
    targets = anchor - np.linspace(span_s, 0, rgb_horizon)
    rgb = np.asarray([available[np.argmin(np.abs(rt[available] - t))] for t in targets])
    if np.any(np.diff(rgb) <= 0) or np.max(np.abs(rt[rgb] - targets)) > .025:
        raise ObservationUnavailable('warming_up_or_rgb_cadence_mismatch')
    end = np.searchsorted(ft, rt[rgb], side='right') - 1
    force = end[:, None] - np.arange(40, -1, -1)
    if force.min() < 0 or np.any(ft[force[:, 0]] < task_start):
        raise ObservationUnavailable('warming_up_paired_ft')
    ages = rt[rgb] - ft[end]
    span = ft[end] - ft[force[:, 0]]
    if (np.any(ages < 0) or np.any(ages > max_force_age_s)
            or np.any(np.diff(ft[force], axis=-1) > .025)
            or np.any(span < .3) or np.any(span > .5)):
        raise ObservationUnavailable('paired_ft_cadence_or_age_mismatch')
    return rgb, force


class SlipTemporalContextRuntime:
    """Call once per ~0.1 seconds, resetting at episode/task/ignored-region boundaries.

    The model's observation-only probabilities enter history; corrected outputs
    are never recursively fed back. Startup needs RGB span + ~0.4 s of sensor data.
    """
    def __init__(self, checkpoint, device='cpu'):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload.get('smoke_test'):
            raise ValueError('Smoke-test checkpoints cannot be used for live inference')
        self.model = SlipTemporalContextEncoder.from_checkpoint(payload).eval().requires_grad_(False)
        self.max_force_age_s = float(payload.get('training_config', {}).get('dataset', {}).get('max_force_age_s', .012))
        if not 0 < self.max_force_age_s <= .012:
            raise ValueError('Unsupported checkpoint force-age contract')
        self.device = torch.device(device)
        self.model.to(self.device)
        self.metadata = dict(schema=payload['schema'], phase_names=list(PHASE_NAMES),
                             model_config=self.model.config, rgb_horizon=self.model.rgb_horizon,
                             rgb_span_s=self.model.rgb_span_s, ft_samples_per_rgb=41,
                             startup_sensor_span_s=self.model.rgb_span_s + .4,
                             history_source='observation_model_probabilities',
                             recommended_hz=1 / self.model.history_period_s)
        self.reset()

    def reset(self, task_start_timestamp=-np.inf):
        self.task_start = float(task_start_timestamp)
        self.history = deque(maxlen=self.model.history_steps + 2)
        self.last_timestamp = None

    def history_at(self, timestamp):
        n, dt = self.model.history_steps, self.model.history_period_s
        probabilities, valid, ages = np.zeros((n, 4), np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
        used = set()
        for i, offset in enumerate(np.arange(n, 0, -1) * dt):
            candidates = [(abs(t - (timestamp - offset)), j, t, p)
                          for j, (t, p) in enumerate(self.history) if t < timestamp and j not in used]
            if not candidates:
                continue
            error, j, t, p = min(candidates, key=lambda row: row[0])
            if error <= .025:
                probabilities[i], valid[i], ages[i] = p, True, timestamp - t
                used.add(j)
        missing = np.flatnonzero(~valid)
        if len(missing):
            valid[:missing[-1] + 1] = False
        return probabilities, valid, ages

    def predict(self, rgb_times, rgb_frames, ft_times, wrench_12d, *, anchor_timestamp,
                now=None, active=True):
        anchor = float(anchor_timestamp)
        if not active:
            self.reset(anchor)
            raise ObservationUnavailable('inactive_or_ignored_region')
        try:
            if now is not None and (not np.isfinite(now) or now < anchor or now - anchor > .5):
                raise ObservationUnavailable('stale_or_future_rgb')
            ri, fi = select_slip_window(rgb_times, ft_times, anchor, self.task_start, self.max_force_age_s,
                                       rgb_horizon=self.model.rgb_horizon, rgb_span_s=self.model.rgb_span_s)
            frames = np.asarray(rgb_frames)
            wrench = np.asarray(wrench_12d, np.float32)
            if frames.shape != (len(rgb_times), 224, 224, 3) or frames.dtype != np.uint8:
                raise ValueError('Provide preprocessed uint8 RGB [T,224,224,3]')
            if wrench.shape != (len(ft_times), 12) or not np.isfinite(wrench).all():
                raise ValueError('Provide finite bias-removed native wrench [T,12]')
            obs = dict(camera0_rgb=torch.from_numpy(np.ascontiguousarray(frames[ri].transpose(0, 3, 1, 2))).float() / 255,
                       robot0_ft_left=torch.from_numpy(wrench[fi, :6].copy()),
                       robot0_ft_right=torch.from_numpy(wrench[fi, 6:].copy()))
            result = self.predict_observation(obs, anchor)
            result.update(rgb_timestamp_s=np.asarray(rgb_times)[ri].tolist(),
                          ft_end_timestamp_s=np.asarray(ft_times)[fi[:, -1]].tolist())
            return result
        except ObservationUnavailable:
            self.history.clear()
            raise

    @torch.inference_mode()
    def predict_observation(self, obs, timestamp):
        """Prepared single observation; caller must enforce the per-RGB sensor contract."""
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < self.task_start:
            raise ValueError('Invalid observation timestamp')
        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp
            if delta <= 0:
                raise ValueError('Context timestamps must increase; reset between episodes')
            if abs(delta - self.model.history_period_s) > .025:
                self.history.clear()
        p, valid, ages = self.history_at(timestamp)
        feature, base_logits = self.model.observe({k: v[None].to(self.device) for k, v in obs.items()})
        logits = self.model.fuse(feature, base_logits, torch.from_numpy(p)[None].to(self.device),
                                torch.from_numpy(valid)[None].to(self.device), torch.from_numpy(ages)[None].to(self.device))
        if not torch.isfinite(logits).all() or not torch.isfinite(base_logits).all():
            self.history.clear()
            raise ObservationUnavailable('nonfinite_context_prediction')
        probabilities = logits.float().softmax(-1)[0].cpu().numpy()
        raw = base_logits.float().softmax(-1)[0].cpu().numpy()
        self.history.append((timestamp, raw.copy()))
        self.last_timestamp = timestamp
        cid = int(probabilities.argmax())
        return dict(class_id=cid, class_name=PHASE_NAMES[cid], probabilities=probabilities.tolist(),
                    observation_probabilities=raw.tolist(), history_samples=int(valid.sum()), timestamp=timestamp)
