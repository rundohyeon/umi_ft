#!/usr/bin/env python3
"""Self-contained inference runtime for the v4 valve-state classifier.

Only PyTorch, torchvision, and NumPy are required.  The original training
repository, replay dataset, zarr sidecar, and diffusion-policy package are not
needed at deployment time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


PHASE_NAMES = ("approach", "turning", "endpoint_reached", "task_complete", "error")
ERROR_REASON_NAMES = ("none", "turn_no_contact", "post_contact_drop", "other")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WRENCH_SCALE = np.asarray(
    [10.0, 10.0, 10.0, 0.1, 0.1, 0.1] * 2, dtype=np.float32
)


def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to the first two rotation-matrix columns."""
    vectors = np.asarray(rotvec, dtype=np.float32)
    single = vectors.ndim == 1
    if single:
        vectors = vectors[None]
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("rotation_axis_angle must have shape (3,) or (N, 3)")
    theta = np.linalg.norm(vectors, axis=1, keepdims=True)
    axis = np.divide(vectors, theta, out=np.zeros_like(vectors), where=theta > 1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    skew = np.zeros((len(vectors), 3, 3), dtype=np.float32)
    skew[:, 0, 1], skew[:, 0, 2] = -z, y
    skew[:, 1, 0], skew[:, 1, 2] = z, -x
    skew[:, 2, 0], skew[:, 2, 1] = -y, x
    eye = np.eye(3, dtype=np.float32)[None]
    theta_matrix = theta[:, None]
    rotation = eye + np.sin(theta_matrix) * skew + (1.0 - np.cos(theta_matrix)) * (skew @ skew)
    result = rotation[:, :, :2].transpose(0, 2, 1).reshape(len(vectors), 6)
    return result[0] if single else result


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return groups


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, padding=0, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(value, (self.left_padding, 0)))


class ForceHistoryEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                CausalConv1d(in_channels, out_channels, kernel_size=5, stride=2),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(),
            )

        self.network = nn.Sequential(
            block(13, base_channels),
            block(base_channels, base_channels * 2),
            block(base_channels * 2, base_channels * 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(base_channels * 4, feature_dim), nn.SiLU()
        )

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = history.transpose(1, 2)
        mask_channel = mask[:, None].to(value.dtype)
        return self.projection(
            self.network(torch.cat([value * mask_channel, mask_channel], dim=1))
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.net(value))


class ValveStateClassifier(nn.Module):
    """Architecture exactly matching the exported epoch-23 checkpoint."""

    def __init__(self, freeze_vision: bool, temporal_width: int, dropout: float) -> None:
        super().__init__()
        self.freeze_vision = bool(freeze_vision)
        self.vision = resnet18(weights=None)
        self.vision.fc = nn.Identity()
        self.vision_projection = nn.Sequential(nn.Linear(512, 128), nn.SiLU())
        self.force = ForceHistoryEncoder(feature_dim=128)
        self.lowdim = nn.Sequential(
            nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.input_projection = nn.Sequential(
            nn.Linear(128 + 128 + 64, temporal_width), nn.SiLU()
        )
        self.temporal = nn.Sequential(
            TemporalBlock(temporal_width, 1, dropout),
            TemporalBlock(temporal_width, 2, dropout),
            TemporalBlock(temporal_width, 4, dropout),
        )
        self.phase_head = nn.Linear(temporal_width, len(PHASE_NAMES))
        self.reason_head = nn.Linear(temporal_width, len(ERROR_REASON_NAMES))
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN)[None, None, :, None, None]
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD)[None, None, :, None, None]
        )
        self.register_buffer("lowdim_mean", torch.zeros(10))
        self.register_buffer("lowdim_std", torch.ones(10))

    def forward(
        self,
        image: torch.Tensor,
        lowdim: torch.Tensor,
        force: torch.Tensor,
        force_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps = image.shape[:2]
        image = (image - self.image_mean) / self.image_std
        flat_image = image.reshape(batch * steps, *image.shape[2:])
        if self.freeze_vision:
            self.vision.eval()
            with torch.no_grad():
                visual = self.vision(flat_image)
        else:
            visual = self.vision(flat_image)
        visual = self.vision_projection(visual)
        flat_force = force.reshape(batch * steps, *force.shape[2:])
        flat_mask = force_mask.reshape(batch * steps, force_mask.shape[-1])
        force_feature = self.force(flat_force, flat_mask)
        normalized_lowdim = (lowdim - self.lowdim_mean) / self.lowdim_std
        lowdim_feature = self.lowdim(normalized_lowdim.reshape(batch * steps, -1))
        feature = torch.cat([visual, force_feature, lowdim_feature], dim=-1)
        feature = self.input_projection(feature).reshape(batch, steps, -1).transpose(1, 2)
        latest = self.temporal(feature)[:, :, -1]
        return self.phase_head(latest), self.reason_head(latest)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def load_classifier(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> Tuple[ValveStateClassifier, Mapping[str, Any], torch.device]:
    """Load and strictly validate an exported checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    resolved_device = resolve_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    if checkpoint.get("schema") != "valve_state_classifier_v1":
        raise ValueError("unsupported checkpoint schema")
    config = checkpoint.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no run_config")
    if tuple(config.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError("checkpoint phase order differs from runtime")
    if tuple(config.get("error_reason_names", ())) != ERROR_REASON_NAMES:
        raise ValueError("checkpoint error-reason order differs from runtime")
    model = ValveStateClassifier(
        freeze_vision=not bool(config.get("finetune_vision", False)),
        temporal_width=int(config["temporal_width"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    return model, checkpoint, resolved_device


def _image_sequence_to_tensor(images: np.ndarray) -> torch.Tensor:
    values = np.asarray(images)
    if values.ndim != 4:
        raise ValueError("images must have shape (T,H,W,3) or (T,3,H,W)")
    if values.shape[-1] == 3:
        values = values.transpose(0, 3, 1, 2)
    elif values.shape[1] != 3:
        raise ValueError("images must have exactly three color channels")
    values = np.ascontiguousarray(values)
    tensor = torch.from_numpy(values).float()
    if values.dtype == np.uint8 or float(tensor.max()) > 1.5:
        tensor.div_(255.0)
    return tensor


@dataclass(frozen=True)
class ValvePrediction:
    timestamp_s: float
    phase_id: int
    phase: str
    confidence: float
    phase_probabilities: Dict[str, float]
    error_reason_id: int
    error_reason: str
    error_reason_probabilities: Dict[str, float]
    warmed_up: bool
    rgb_context_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def predict_window(
    model: ValveStateClassifier,
    device: torch.device,
    *,
    images: np.ndarray,
    lowdim: np.ndarray,
    wrench_history: np.ndarray,
    wrench_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict one already-sampled causal window.

    `wrench_history` must be in physical units and have shape (T,H,12).
    It is scaled exactly as during training inside this function.
    """
    image_tensor = _image_sequence_to_tensor(images)
    lowdim_values = np.asarray(lowdim, dtype=np.float32)
    wrench_values = np.asarray(wrench_history, dtype=np.float32)
    if lowdim_values.shape != (len(image_tensor), 10):
        raise ValueError("lowdim must have shape (T,10)")
    if wrench_values.ndim != 3 or wrench_values.shape[0] != len(image_tensor) or wrench_values.shape[2] != 12:
        raise ValueError("wrench_history must have shape (T,H,12)")
    if wrench_mask is None:
        mask_values = np.ones(wrench_values.shape[:2], dtype=np.float32)
    else:
        mask_values = np.asarray(wrench_mask, dtype=np.float32)
        if mask_values.shape != wrench_values.shape[:2]:
            raise ValueError("wrench_mask must have shape (T,H)")
    wrench_values = wrench_values / WRENCH_SCALE[None, None]
    phase_logits, reason_logits = model(
        image_tensor[None].to(device),
        torch.from_numpy(lowdim_values)[None].to(device),
        torch.from_numpy(wrench_values)[None].to(device),
        torch.from_numpy(mask_values)[None].to(device),
    )
    return (
        torch.softmax(phase_logits[0], dim=-1).cpu().numpy(),
        torch.softmax(reason_logits[0], dim=-1).cpu().numpy(),
    )


@dataclass
class _Observation:
    timestamp_s: float
    image: np.ndarray
    lowdim: np.ndarray
    force: np.ndarray
    force_mask: np.ndarray


class ValveStateRuntime:
    """Streaming adapter that reproduces the training-time causal histories.

    Add every 12-D wrench sample with `append_wrench`, then call `predict` for
    every RGB/robot observation in timestamp order.  At 60 Hz, predictions use
    16 RGB steps with stride 4 (about 1.001 seconds).  Startup frames are
    clamped to the first observation, exactly as in training.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        max_force_buffer: int = 2000,
    ) -> None:
        self.model, self.checkpoint, self.device = load_classifier(checkpoint_path, device)
        config = self.checkpoint["run_config"]
        self.temporal_steps = int(config["temporal_steps"])
        self.temporal_stride = int(config["temporal_stride"])
        self.force_history_samples = int(config["force_history_samples"])
        self.raw_context_frames = 1 + (self.temporal_steps - 1) * self.temporal_stride
        self.observations: Deque[_Observation] = deque(maxlen=self.raw_context_frames)
        self.force_timestamps: Deque[float] = deque(maxlen=max_force_buffer)
        self.force_values: Deque[np.ndarray] = deque(maxlen=max_force_buffer)
        self._last_rgb_timestamp: Optional[float] = None
        self._last_force_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Start a new episode; histories must never cross episode boundaries."""
        self.observations.clear()
        self.force_timestamps.clear()
        self.force_values.clear()
        self._last_rgb_timestamp = None
        self._last_force_timestamp = None

    def append_wrench(self, timestamp_s: float, wrench_12d: Sequence[float]) -> None:
        timestamp_s = float(timestamp_s)
        value = np.asarray(wrench_12d, dtype=np.float32)
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError("wrench_12d must contain 12 finite values")
        if self._last_force_timestamp is not None and timestamp_s <= self._last_force_timestamp:
            raise ValueError("wrench timestamps must be strictly increasing")
        self.force_timestamps.append(timestamp_s)
        self.force_values.append(value.copy())
        self._last_force_timestamp = timestamp_s

    def append_wrench_batch(self, timestamps_s: Sequence[float], wrench_12d: np.ndarray) -> None:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        values = np.asarray(wrench_12d, dtype=np.float32)
        if values.shape != (len(timestamps), 12):
            raise ValueError("batched wrench must have shape (N,12)")
        for timestamp, value in zip(timestamps, values):
            self.append_wrench(float(timestamp), value)

    def _force_history_at(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        history = np.zeros((self.force_history_samples, 12), dtype=np.float32)
        mask = np.zeros(self.force_history_samples, dtype=np.float32)
        if not self.force_timestamps:
            return history, mask
        times = np.fromiter(self.force_timestamps, dtype=np.float64)
        stop = int(np.searchsorted(times, timestamp_s, side="right"))
        if stop == 0:
            return history, mask
        start = max(0, stop - self.force_history_samples)
        samples = np.stack(list(self.force_values)[start:stop])
        count = len(samples)
        history[-count:] = samples
        history[:-count] = samples[0]
        mask[-count:] = 1.0
        return history, mask

    def predict(
        self,
        *,
        timestamp_s: float,
        rgb: np.ndarray,
        position_m: Sequence[float],
        rotation_axis_angle_rad: Sequence[float],
        gripper_width_m: float,
    ) -> ValvePrediction:
        timestamp_s = float(timestamp_s)
        if self._last_rgb_timestamp is not None and timestamp_s <= self._last_rgb_timestamp:
            raise ValueError("RGB timestamps must be strictly increasing")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb must have shape (H,W,3) in RGB channel order")
        position = np.asarray(position_m, dtype=np.float32)
        rotation = np.asarray(rotation_axis_angle_rad, dtype=np.float32)
        if position.shape != (3,) or rotation.shape != (3,):
            raise ValueError("position and axis-angle rotation must each have shape (3,)")
        lowdim = np.concatenate(
            [position, rotvec_to_rot6d(rotation), np.asarray([gripper_width_m], dtype=np.float32)]
        ).astype(np.float32)
        force, force_mask = self._force_history_at(timestamp_s)
        self.observations.append(
            _Observation(timestamp_s, image.copy(), lowdim, force, force_mask)
        )
        self._last_rgb_timestamp = timestamp_s

        current = len(self.observations) - 1
        indices = current - np.arange(
            self.temporal_steps - 1, -1, -1, dtype=np.int64
        ) * self.temporal_stride
        indices = np.maximum(indices, 0)
        observations = list(self.observations)
        chosen = [observations[int(index)] for index in indices]
        phase_probability, reason_probability = predict_window(
            self.model,
            self.device,
            images=np.stack([item.image for item in chosen]),
            lowdim=np.stack([item.lowdim for item in chosen]),
            wrench_history=np.stack([item.force for item in chosen]),
            wrench_mask=np.stack([item.force_mask for item in chosen]),
        )
        phase_id = int(phase_probability.argmax())
        reason_id = int(reason_probability.argmax())
        return ValvePrediction(
            timestamp_s=timestamp_s,
            phase_id=phase_id,
            phase=PHASE_NAMES[phase_id],
            confidence=float(phase_probability[phase_id]),
            phase_probabilities={
                name: float(value) for name, value in zip(PHASE_NAMES, phase_probability)
            },
            error_reason_id=reason_id,
            error_reason=ERROR_REASON_NAMES[reason_id],
            error_reason_probabilities={
                name: float(value)
                for name, value in zip(ERROR_REASON_NAMES, reason_probability)
            },
            warmed_up=len(self.observations) >= self.raw_context_frames,
            rgb_context_s=float(chosen[-1].timestamp_s - chosen[0].timestamp_s),
        )


__all__ = [
    "ERROR_REASON_NAMES",
    "PHASE_NAMES",
    "ValvePrediction",
    "ValveStateClassifier",
    "ValveStateRuntime",
    "load_classifier",
    "predict_window",
    "rotvec_to_rot6d",
]
