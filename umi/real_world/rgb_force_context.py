"""Inference contract for the Stage A RGB/native-F/T classifier (no robot I/O)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from models.rgb_force_context_encoder import RGBForceContextEncoder


PHASE_NAMES = ('approach', 'turning', 'recovery', 'error')
SCHEMAS = ('context_rgb_force_4state_v1', 'context_rgb_force_4state_ft_features_v2')


class ObservationUnavailable(ValueError):
    """Missing, stale, or discontinuous data: do not publish a class prediction."""


@dataclass(frozen=True)
class TimingLimits:
    # Receive timestamps must first be corrected by the measured sensor latency.
    max_rgb_age_s: float = 0.5
    min_rgb_pair_span_s: float = 0.03
    max_rgb_pair_span_s: float = 0.075
    max_ft_gap_s: float = 0.025
    ft_sample_hz: float = 100.0
    ft_span_tolerance: float = 0.25

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('Timing limits must be finite and positive')
        if self.min_rgb_pair_span_s >= self.max_rgb_pair_span_s or self.ft_span_tolerance >= 1:
            raise ValueError('Invalid timing range')


def _timestamps(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ObservationUnavailable(f'{name}: timestamps must be finite and strictly increasing')
    return values


def select_context_window(rgb_times, ft_times, *, required_ft_history=41,
                          rgb_stride=3, max_force_age_s=0.012,
                          episode_start=-np.inf, now=None, limits=None):
    """Select native samples, never interpolate, resample, or include future F/T.

    Live startup waits for a full window instead of repeating startup samples.
    Indices refer to the supplied buffers. Both buffers must use the same clock.
    """
    limits = limits or TimingLimits()
    rt, ft = _timestamps(rgb_times, 'RGB'), _timestamps(ft_times, 'F/T')
    ri = np.flatnonzero(rt >= episode_start)
    if len(ri) < rgb_stride + 1:
        raise ObservationUnavailable('warming_up_rgb')
    rgb_indices = ri[[-1 - rgb_stride, -1]]
    anchor = float(rt[rgb_indices[-1]])
    if now is not None and (now < anchor or now - anchor > limits.max_rgb_age_s):
        raise ObservationUnavailable('stale_or_future_rgb')
    pair_span = float(np.diff(rt[rgb_indices])[0])
    if not limits.min_rgb_pair_span_s <= pair_span <= limits.max_rgb_pair_span_s:
        raise ObservationUnavailable('rgb_cadence_mismatch: expected about 50 ms between t-3 and t')
    start = int(np.searchsorted(ft, episode_start, side='left'))
    end = int(np.searchsorted(ft, anchor, side='right'))
    if end - start < required_ft_history:
        raise ObservationUnavailable('warming_up_ft')
    ft_indices = np.arange(end - required_ft_history, end)
    selected_times = ft[ft_indices]
    age = anchor - float(selected_times[-1])
    if age > max_force_age_s + 1e-9:
        raise ObservationUnavailable(f'stale_ft_at_rgb_anchor: {age:.6f} s')
    if np.max(np.diff(selected_times)) > limits.max_ft_gap_s:
        raise ObservationUnavailable('gap_in_ft_history')
    span = float(selected_times[-1] - selected_times[0])
    expected_span = (required_ft_history - 1) / limits.ft_sample_hz
    if abs(span / expected_span - 1) > limits.ft_span_tolerance:
        raise ObservationUnavailable('ft_cadence_mismatch: expected native 100 Hz samples')
    return rgb_indices, ft_indices, dict(
        anchor_timestamp=anchor, rgb_pair_span_s=pair_span,
        ft_age_s=age, ft_history_span_s=span)


class TrainingImageTransform:
    """Raw BGR -> tag inpaint -> gripper mask -> resize/center crop -> RGB uint8.

    Same settings as the canonical replay-buffer generation and UmiEnv's
    training-compatible path. No mirror swap, lens warp, or image augmentation.
    """
    def __init__(self, aruco_config_path):
        import cv2
        import yaml
        from umi.common.cv_util import parse_aruco_config, _aruco_make_detector_parameters
        with Path(aruco_config_path).open() as f:
            self.aruco_dict = parse_aruco_config(yaml.safe_load(f))['aruco_dict']
        self.parameters = _aruco_make_detector_parameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def __call__(self, bgr):
        from diffusion_policy.common.cv2_util import get_image_transform
        from umi.common.cv_util import _aruco_detect_markers, inpaint_tag, draw_predefined_mask
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[-1] != 3:
            raise ValueError('Camera must supply HWC uint8 BGR')
        img = bgr.copy()
        corners, ids, _ = _aruco_detect_markers(img, self.aruco_dict, self.parameters)
        if ids is not None:
            for corner in corners:
                img = inpaint_tag(img, corner.squeeze())
        img = draw_predefined_mask(img, color=(0, 0, 0), mirror=False,
                                   gripper=True, finger=False, use_aa=False)
        transform = get_image_transform((img.shape[1], img.shape[0]), (224, 224), bgr_to_rgb=True)
        return np.ascontiguousarray(transform(img))


class RGBForceContextRuntime:
    """Restore full checkpoint, including CLIP and learned normalization buffers.

    Supply bias-corrected native-frame wrenches in N/Nm. The model constructs
    mean/delta features and normalizes RGB/F/T internally, exactly once.
    """
    def __init__(self, checkpoint, device='cpu'):
        # Only load a trusted training artifact; these checkpoints include config
        # and optimizer objects as well as tensors (PyTorch 2.6 defaults changed).
        payload = torch.load(Path(checkpoint).expanduser(), map_location='cpu', weights_only=False)
        if payload.get('schema') not in SCHEMAS:
            raise ValueError('Not an RGBForceContextEncoder checkpoint; Qwen/policy/old observer files are incompatible')
        if payload.get('smoke_test', False):
            raise ValueError('Smoke-test checkpoint cannot be used for real evaluation')
        if tuple(payload.get('manifest', {}).get('phase_names', ())) != PHASE_NAMES:
            raise ValueError('Checkpoint classes must be approach/turning/recovery/error in that order')
        dataset_cfg = payload['training_config']['dataset']
        self.rgb_stride = int(dataset_cfg['rgb_stride'])
        self.max_force_age_s = float(dataset_cfg['max_force_age_s'])
        if self.rgb_stride != 3 or not 0 < self.max_force_age_s <= 0.012:
            raise ValueError('Unsupported training time alignment contract')
        self.device = torch.device(device)
        self.model = RGBForceContextEncoder.from_checkpoint(payload).eval().requires_grad_(False)
        for name in ('ft_mean', 'ft_std', 'rgb_mean', 'rgb_std'):
            value = getattr(self.model, name)
            if not torch.isfinite(value).all() or (name.endswith('std') and (value <= 0).any()):
                raise ValueError(f'Invalid normalization buffer: {name}')
        self.model.to(self.device)
        self.required_ft_history = self.model.required_ft_history
        self.metadata = dict(
            schema=payload['schema'], phase_names=list(PHASE_NAMES),
            epoch=int(payload['epoch']) + 1,
            best_validation_macro_f1=payload.get('best_validation_macro_f1'),
            model_config=self.model.config, required_ft_history=self.required_ft_history,
            rgb_stride=self.rgb_stride, max_force_age_s=self.max_force_age_s,
            ft_channel_order=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
            device=str(self.device))

    def prepare(self, rgb_times, rgb_frames, ft_times, wrench_12d, *,
                image_transform=None, episode_start=-np.inf, now=None, limits=None):
        """Buffers contain every captured sample, not just prediction-rate samples.

        image_transform=None means frames are ALREADY preprocessed RGB uint8;
        use TrainingImageTransform for raw camera BGR. Wrenches are already tared.
        """
        ri, fi, timing = select_context_window(
            rgb_times, ft_times, required_ft_history=self.required_ft_history,
            rgb_stride=self.rgb_stride, max_force_age_s=self.max_force_age_s,
            episode_start=episode_start, now=now, limits=limits)
        wrench = np.asarray(wrench_12d)
        if len(rgb_frames) != len(rgb_times) or wrench.shape != (len(ft_times), 12):
            raise ValueError('Sensor buffers have mismatched shapes')
        images = [rgb_frames[i] for i in ri]
        if image_transform is not None:
            images = [image_transform(img) for img in images]
        obs = dict(camera0_rgb=np.stack(images),
                   robot0_ft_left=wrench[fi, :6].astype(np.float32),
                   robot0_ft_right=wrench[fi, 6:].astype(np.float32))
        timing.update(rgb_timestamps=np.asarray(rgb_times)[ri].tolist(),
                      ft_timestamps=np.asarray(ft_times)[fi].tolist())
        return obs, timing

    @torch.inference_mode()
    def predict(self, obs):
        """Two uint8 HWC RGB images + two float [41,6] histories -> class softmax.

        Raw-only v1 checkpoints use 32 F/T samples. No threshold, phase smoothing,
        or automatic unknown/finish relabeling is applied to these four classes.
        """
        rgb = np.asarray(obs['camera0_rgb'])
        if rgb.shape != (2, 224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError('camera0_rgb must be preprocessed uint8 [2,224,224,3] RGB')
        tensors = dict(camera0_rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
                       .unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0)
        for key in ('robot0_ft_left', 'robot0_ft_right'):
            value = np.asarray(obs[key], dtype=np.float32)
            if value.shape != (self.required_ft_history, 6) or not np.isfinite(value).all():
                raise ValueError(f'{key} must be finite [{self.required_ft_history},6] N/Nm')
            tensors[key] = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        logits = self.model(tensors)[0].float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError('Model returned nonfinite logits')
        probabilities = logits.softmax(-1).numpy()
        class_id = int(probabilities.argmax())
        return dict(valid=True, class_id=class_id, phase=PHASE_NAMES[class_id],
                    confidence=float(probabilities[class_id]),
                    probabilities=probabilities.tolist(), logits=logits.tolist(),
                    inference_ms=(time.perf_counter() - started) * 1000)
