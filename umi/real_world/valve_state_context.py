"""Causal bridge from live UMI observations to frozen valve-state context."""

from __future__ import annotations

import hashlib
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.common.valve_context_contract import (
    VALVE_CONTEXT_V1_SCHEMA,
    VALVE_CONTEXT_V2_SCHEMA,
    ValveContextRecord,
    context_from_classifier_prediction,
    context_from_observer_probabilities,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.model.vision.valve_context_observer_v2 import (
    OBSERVER_FORCE_HISTORY_SAMPLES,
    OBSERVER_PHASE_NAMES,
    OBSERVER_RGB_STRIDE,
    load_frozen_context_observer,
)
from umi.common.pose_util import mat_to_pose10d, pose_to_mat


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
        self.context_schema = VALVE_CONTEXT_V1_SCHEMA
        self.context_dim = 10
        self.phase_names = (
            "approach",
            "turning",
            "endpoint_reached",
            "task_complete",
            "error",
        )
        self.observer_version = "v4"

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


class ValveStateContextRuntimeV2:
    """Streaming adapter for the frozen four-state ``best_context.pt`` observer.

    Every prediction uses the current RGB observation and the observation three
    camera frames earlier.  Both TCP poses are expressed relative to the current
    TCP, and each RGB anchor receives only its preceding 50 physical F/T samples.
    """

    _WRENCH_MODEL_SCALE = np.asarray(
        [0.1, 0.1, 0.1, 10.0, 10.0, 10.0] * 2, dtype=np.float32
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        max_force_buffer: int = 2000,
    ):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.device = str(device)
        self._model, phases, checkpoint = load_frozen_context_observer(
            checkpoint_path, self.device
        )
        self.checkpoint_metadata = {
            "schema": checkpoint.get("schema"),
            "epoch": checkpoint.get("epoch"),
            "validation": checkpoint.get("validation"),
        }
        del checkpoint
        if tuple(phases) != OBSERVER_PHASE_NAMES:
            raise ValueError("loaded observer phase order changed unexpectedly")

        self.context_schema = VALVE_CONTEXT_V2_SCHEMA
        self.context_dim = 5
        self.phase_names = OBSERVER_PHASE_NAMES
        self.observer_version = "observer_v2_4state"
        self._observations = deque(maxlen=1 + OBSERVER_RGB_STRIDE)
        self._force_timestamps = deque(maxlen=int(max_force_buffer))
        self._force_values = deque(maxlen=int(max_force_buffer))
        self._last_wrench_timestamp = -np.inf
        self._last_prediction_timestamp = -np.inf
        self._episode_start_timestamp = -np.inf
        self.last_record: ValveContextRecord | None = None
        self.last_model_inputs: dict[str, np.ndarray] | None = None

    def reset(self, *, episode_start_timestamp_s: float | None = None) -> None:
        self._observations.clear()
        self._force_timestamps.clear()
        self._force_values.clear()
        self._last_wrench_timestamp = -np.inf
        self._last_prediction_timestamp = -np.inf
        if episode_start_timestamp_s is None:
            self._episode_start_timestamp = -np.inf
        else:
            self._episode_start_timestamp = float(episode_start_timestamp_s)
            if not np.isfinite(self._episode_start_timestamp):
                raise ValueError("episode start timestamp must be finite")
        self.last_record = None
        self.last_model_inputs = None

    @property
    def last_prediction_timestamp(self) -> float:
        return float(self._last_prediction_timestamp)

    def get_last_classifier_model_inputs(self) -> dict[str, np.ndarray]:
        if not isinstance(self.last_model_inputs, dict):
            raise RuntimeError("four-state observer has not produced model inputs")
        return {
            key: np.asarray(value).copy()
            for key, value in self.last_model_inputs.items()
        }

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
            len(timestamps), 6
        ):
            raise ValueError("observer F/T must be matching [N,6] arrays")
        if (
            not np.isfinite(timestamps).all()
            or not np.isfinite(left_wrench).all()
            or not np.isfinite(right_wrench).all()
        ):
            raise ValueError("observer F/T contains NaN or Inf")
        if np.any(np.diff(timestamps) < 0.0):
            raise ValueError("observer F/T timestamps must be sorted")
        if np.any(timestamps > float(anchor_timestamp_s) + 1e-6):
            raise ValueError("observer would consume F/T newer than its RGB anchor")
        appended = 0
        for timestamp, left, right in zip(timestamps, left_wrench, right_wrench):
            timestamp = float(timestamp)
            if timestamp < self._episode_start_timestamp:
                continue
            if timestamp <= self._last_wrench_timestamp:
                continue
            self._force_timestamps.append(timestamp)
            self._force_values.append(np.concatenate([left, right]).astype(np.float32))
            self._last_wrench_timestamp = timestamp
            appended += 1
        return appended

    def _force_history_at(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros(
            (OBSERVER_FORCE_HISTORY_SAMPLES, 12), dtype=np.float32
        )
        mask = np.zeros((OBSERVER_FORCE_HISTORY_SAMPLES, 1), dtype=np.float32)
        if not self._force_timestamps:
            return history, mask
        timestamps = np.fromiter(self._force_timestamps, dtype=np.float64)
        stop = int(np.searchsorted(timestamps, float(timestamp_s), side="right"))
        if stop <= 0:
            return history, mask
        start = max(0, stop - OBSERVER_FORCE_HISTORY_SAMPLES)
        samples = np.stack(list(self._force_values)[start:stop])
        count = len(samples)
        history[-count:] = samples
        mask[-count:, 0] = 1.0
        if count < OBSERVER_FORCE_HISTORY_SAMPLES:
            history[:-count] = samples[0]
        return history, mask

    @staticmethod
    def _relative_lowdim(observations: list[dict]) -> np.ndarray:
        pose_mats = np.stack(
            [
                pose_to_mat(
                    np.concatenate(
                        [item["position_m"], item["rotation_axis_angle_rad"]]
                    )
                )
                for item in observations
            ]
        )
        relative = convert_pose_mat_rep(
            pose_mats,
            base_pose_mat=pose_mats[-1],
            pose_rep="relative",
            backward=False,
        )
        pose10d = mat_to_pose10d(relative).astype(np.float32)
        widths = np.asarray(
            [[item["gripper_width_m"]] for item in observations],
            dtype=np.float32,
        )
        return np.concatenate([pose10d, widths], axis=-1)

    @torch.inference_mode()
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
            raise ValueError("observer RGB timestamp predates this episode")
        if not np.isfinite(timestamp_s) or timestamp_s <= self._last_prediction_timestamp:
            raise ValueError(
                "observer RGB timestamps must be finite and strictly increasing "
                f"(last={self._last_prediction_timestamp}, got={timestamp_s})"
            )
        rgb = np.asarray(rgb)
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                "observer must receive final policy RGB uint8 [224,224,3], "
                f"got {rgb.shape} {rgb.dtype}"
            )
        position = np.asarray(position_m, dtype=np.float32).reshape(3)
        rotation = np.asarray(
            rotation_axis_angle_rad, dtype=np.float32
        ).reshape(3)
        if (
            not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
            or not np.isfinite(gripper_width_m)
        ):
            raise ValueError("observer TCP/gripper input contains NaN or Inf")

        self.append_causal_wrench_history(
            ft_timestamps,
            ft_left,
            ft_right,
            anchor_timestamp_s=timestamp_s,
        )
        self._observations.append(
            {
                "timestamp_s": timestamp_s,
                "rgb": rgb.copy(),
                "position_m": position.copy(),
                "rotation_axis_angle_rad": rotation.copy(),
                "gripper_width_m": float(gripper_width_m),
            }
        )
        available = list(self._observations)
        old_index = max(0, len(available) - 1 - OBSERVER_RGB_STRIDE)
        selected = [available[old_index], available[-1]]
        histories, masks = zip(
            *(self._force_history_at(item["timestamp_s"]) for item in selected)
        )
        wrench_history = np.stack(histories).astype(np.float32)
        wrench_mask = np.stack(masks).astype(np.float32)
        lowdim = self._relative_lowdim(selected)
        images = np.stack([item["rgb"] for item in selected])
        obs = {
            "camera0_rgb": torch.from_numpy(
                np.moveaxis(images, -1, 1).astype(np.float32) / 255.0
            )[None].to(self.device),
            "robot0_eef_pos": torch.from_numpy(lowdim[:, :3])[None].to(self.device),
            "robot0_eef_rot_axis_angle": torch.from_numpy(lowdim[:, 3:9])[None].to(self.device),
            "robot0_gripper_width": torch.from_numpy(lowdim[:, 9:10])[None].to(self.device),
            "robot0_ft_history": torch.from_numpy(wrench_history)[None].to(self.device),
            "robot0_ft_history_valid": torch.from_numpy(wrench_mask)[None].to(self.device),
        }
        output = self._model.predict_context(obs)
        probability = output["context_prob"][0].detach().cpu().numpy()
        # Full readiness requires a genuinely older RGB frame and 50 causal F/T
        # samples for both selected RGB anchors. Startup predictions remain
        # available, but the fifth context value tells the policy they are padded.
        context_valid = (
            len(available) >= 1 + OBSERVER_RGB_STRIDE
            and bool(np.all(wrench_mask == 1.0))
        )
        record = context_from_observer_probabilities(
            probability,
            timestamp_s=timestamp_s,
            context_valid=context_valid,
        )
        self.last_model_inputs = {
            "rgb_timestamp_s": np.asarray(
                [item["timestamp_s"] for item in selected], dtype=np.float64
            ),
            "lowdim": lowdim,
            "wrench_history_physical": wrench_history,
            "wrench_history_model_scaled": (
                wrench_history * self._WRENCH_MODEL_SCALE[None, None]
            ).astype(np.float32),
            "wrench_mask": wrench_mask[..., 0],
        }
        self._last_prediction_timestamp = timestamp_s
        self.last_record = record
        return record


__all__ = [
    "ValveStateContextRuntime",
    "ValveStateContextRuntimeV2",
    "sha256_file",
]
