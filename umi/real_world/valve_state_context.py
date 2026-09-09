"""Causal bridge from live UMI observations to frozen valve-state context."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from diffusion_policy.common.valve_context_contract import (
    ValveContextRecord,
    context_from_classifier_prediction,
)


_CLASSIFIER_ROOT = Path(__file__).resolve().parents[2] / "valve_state_classifier_v4"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ValveStateContextRuntime:
    """Feed corrected causal F/T once, then predict a strict 10-D context."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"valve classifier checkpoint not found: {checkpoint_path}")
        if not _CLASSIFIER_ROOT.is_dir():
            raise FileNotFoundError(
                "valve_state_classifier_v4 package is missing next to the project: "
                f"{_CLASSIFIER_ROOT}"
            )
        package_parent = str(_CLASSIFIER_ROOT.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)
        try:
            from valve_state_classifier_v4.valve_state_classifier import (
                ValveStateRuntime,
            )
        except Exception as exc:  # pragma: no cover - hardware/container dependent
            raise RuntimeError(
                "could not import the frozen valve_state_classifier_v4 runtime"
            ) from exc
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.device = str(device)
        self._runtime_type = ValveStateRuntime
        self._runtime = ValveStateRuntime(str(checkpoint_path), device=self.device)
        self._last_wrench_timestamp = -np.inf
        self._last_prediction_timestamp = -np.inf
        self._episode_start_timestamp = -np.inf
        self.last_record: ValveContextRecord | None = None

    def reset(self, *, episode_start_timestamp_s: float | None = None) -> None:
        # The supplied v4 runtime owns only stream history here; retain its
        # already-loaded frozen weights and reset exactly at an episode boundary.
        self._runtime.reset()
        self._last_wrench_timestamp = -np.inf
        self._last_prediction_timestamp = -np.inf
        if episode_start_timestamp_s is None:
            self._episode_start_timestamp = -np.inf
        else:
            self._episode_start_timestamp = float(episode_start_timestamp_s)
            if not np.isfinite(self._episode_start_timestamp):
                raise ValueError("episode start timestamp must be finite")
        self.last_record = None

    @property
    def last_prediction_timestamp(self) -> float:
        return float(self._last_prediction_timestamp)

    def get_last_classifier_model_inputs(self) -> dict[str, np.ndarray]:
        """Return the frozen classifier's exact most recent causal window.

        RGB source frames are captured losslessly by the evaluator and are
        referenced by their timestamps. The remaining entries are the exact
        precomputed temporal low-dimensional/F-T tensors. This is an
        observation-only debug API; it does not mutate classifier state.
        """
        values = getattr(self._runtime, "last_model_inputs", None)
        if not isinstance(values, dict):
            raise RuntimeError("classifier did not expose its latest model inputs")
        required = {
            "rgb_timestamp_s",
            "lowdim",
            "wrench_history_physical",
            "wrench_history_model_scaled",
            "wrench_mask",
        }
        missing = required - set(values)
        if missing:
            raise RuntimeError(
                "classifier latest-model-input snapshot is incomplete: "
                + ", ".join(sorted(missing))
            )
        return {key: np.asarray(values[key]).copy() for key in required}

    def append_causal_wrench_history(
        self,
        timestamps,
        left_wrench,
        right_wrench,
        *,
        anchor_timestamp_s: float,
    ) -> int:
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        left_wrench = np.asarray(left_wrench, dtype=np.float32)
        right_wrench = np.asarray(right_wrench, dtype=np.float32)
        if left_wrench.shape != (len(timestamps), 6) or right_wrench.shape != (
            len(timestamps),
            6,
        ):
            raise ValueError(
                "classifier F/T histories must be matching [N,6] left/right arrays"
            )
        if not np.isfinite(timestamps).all() or not np.isfinite(left_wrench).all() or not np.isfinite(right_wrench).all():
            raise ValueError("classifier F/T history contains NaN or Inf")
        if np.any(np.diff(timestamps) < 0.0):
            raise ValueError("classifier F/T timestamps must be sorted")
        if np.any(timestamps > float(anchor_timestamp_s) + 1e-6):
            raise ValueError("classifier would consume F/T newer than its RGB anchor")
        appended = 0
        for timestamp, left, right in zip(timestamps, left_wrench, right_wrench):
            if timestamp < self._episode_start_timestamp:
                continue
            if timestamp <= self._last_wrench_timestamp:
                continue
            self._runtime.append_wrench(float(timestamp), np.concatenate([left, right]))
            self._last_wrench_timestamp = float(timestamp)
            appended += 1
        return appended

    def predict(
        self,
        *,
        timestamp_s: float,
        rgb: np.ndarray,
        position_m: np.ndarray,
        rotation_axis_angle_rad: np.ndarray,
        gripper_width_m: float,
        ft_timestamps: np.ndarray,
        ft_left: np.ndarray,
        ft_right: np.ndarray,
    ) -> ValveContextRecord:
        timestamp_s = float(timestamp_s)
        if timestamp_s < self._episode_start_timestamp:
            raise ValueError("classifier RGB timestamp predates this episode")
        if not np.isfinite(timestamp_s) or timestamp_s <= self._last_prediction_timestamp:
            raise ValueError(
                "classifier RGB timestamps must be finite and strictly increasing "
                f"(last={self._last_prediction_timestamp}, got={timestamp_s})"
            )
        rgb = np.asarray(rgb)
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                "classifier must receive the final policy RGB as uint8 [224,224,3], "
                f"got {rgb.shape} {rgb.dtype}"
            )
        self.append_causal_wrench_history(
            ft_timestamps,
            ft_left,
            ft_right,
            anchor_timestamp_s=timestamp_s,
        )
        prediction = self._runtime.predict(
            timestamp_s=timestamp_s,
            rgb=rgb,
            position_m=np.asarray(position_m, dtype=np.float32).reshape(3),
            rotation_axis_angle_rad=np.asarray(
                rotation_axis_angle_rad, dtype=np.float32
            ).reshape(3),
            gripper_width_m=float(gripper_width_m),
        )
        record = context_from_classifier_prediction(prediction)
        if not np.isclose(record.timestamp_s, timestamp_s, atol=1e-6):
            raise ValueError(
                "classifier prediction timestamp did not preserve the RGB anchor"
            )
        self._last_prediction_timestamp = timestamp_s
        self.last_record = record
        return record
