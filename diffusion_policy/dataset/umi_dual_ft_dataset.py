from __future__ import annotations

import copy
import logging
import os
from typing import Dict

import numpy as np
import scipy.spatial.transform as st
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.nested_zarr import (
    detect_zarr_prefix,
    open_nested_zip_group,
)
from diffusion_policy.common.normalize_util import (
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import mat_to_pose10d, pose_to_mat


logger = logging.getLogger(__name__)
register_codecs()


REQUIRED_OBS_KEYS = {
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_ft_left",
    "robot0_ft_right",
}

ALLOWED_OBS_KEYS = REQUIRED_OBS_KEYS | {
    # Retained only so older checkpoints/configs fail gracefully. The current
    # official-parity task config uses pose-only (18-D) proprioception.
    "robot0_gripper_width",
    "robot0_eef_rot_axis_angle_wrt_start",
}

DISALLOWED_KEY_FRAGMENTS = (
    "stiffness",
    "virtual_target",
    "wrist",
    "depth",
    "camera1",
)

POSE_QUATERNION_ORDER = "xyzw"


def _cumulative_slices(ends: np.ndarray):
    starts = np.r_[0, ends[:-1]]
    return list(zip(starts.astype(np.int64), ends.astype(np.int64)))


def _positive_dt(timestamps: np.ndarray, ends: np.ndarray):
    chunks = []
    for start, end in _cumulative_slices(ends):
        chunks.append(np.diff(timestamps[start:end]))
    dt = np.concatenate(chunks)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing inside every episode")
    return dt


def _subtract_episode_bias(
    values: np.ndarray,
    episode_ends: np.ndarray,
    episode_bias: np.ndarray,
) -> np.ndarray:
    """Subtract one native-frame six-axis bias vector per episode."""

    values = np.asarray(values, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    episode_bias = np.asarray(episode_bias, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"F/T values must be [N,6], got {values.shape}")
    if episode_bias.shape != (len(episode_ends), 6):
        raise ValueError(
            "episode F/T bias must be [n_episodes,6], got "
            f"{episode_bias.shape} for {len(episode_ends)} episodes"
        )
    if len(episode_ends) == 0 or episode_ends[-1] != len(values):
        raise ValueError("episode ends do not match F/T values for bias removal")
    if np.any(~np.isfinite(episode_bias)):
        raise ValueError("episode F/T bias contains NaN or Inf")

    result = values.copy()
    for episode, (start, end) in enumerate(_cumulative_slices(episode_ends)):
        result[start:end] -= episode_bias[episode]
    return result


def _stats(array: np.ndarray) -> dict:
    x = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
    return {
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
    }


class _StatsAccumulator:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.minimum = np.full(self.dim, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dim, -np.inf, dtype=np.float64)
        self.total = np.zeros(self.dim, dtype=np.float64)
        self.total_sq = np.zeros(self.dim, dtype=np.float64)

    def update(self, array):
        x = np.asarray(array, dtype=np.float64).reshape(-1, self.dim)
        if len(x) == 0:
            return
        self.minimum = np.minimum(self.minimum, np.min(x, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(x, axis=0))
        self.total += np.sum(x, axis=0)
        self.total_sq += np.sum(np.square(x), axis=0)
        self.count += len(x)

    def result(self):
        if self.count == 0:
            raise RuntimeError("cannot compute statistics from zero samples")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
        }


class UmiDualFTDataset(BaseImageDataset):
    """Allowlisted RGB+pose+dual-native-wrench dataset.

    The legacy mode reads every stream from one multirate store.  When
    ``force_sidecar_path`` is supplied, RGB/pose/gripper come from the stock UMI
    ZIP while bias-removed native-frame wrench samples and the corrected clock
    mapping come from the force sidecar. Low-dimensional arrays are copied into
    process-local NumPy memory during construction. JPEG-XL RGB remains in the
    read-only base ZIP and is opened lazily by each DataLoader worker, so a
    ``ZipStore`` is never shared across processes.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        data_keys: dict,
        pose_quaternion_order: str,
        ft: dict,
        force_sidecar_path: str | None = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        seed: int = 42,
        val_ratio: float = 0.05,
        start_pose_noise_scale=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
    ):
        self.shape_meta = shape_meta
        self.dataset_path = str(dataset_path)
        self.force_sidecar_path = (
            None if force_sidecar_path is None else str(force_sidecar_path)
        )
        self.source_mode = (
            "multirate" if self.force_sidecar_path is None
            else "base_with_force_sidecar"
        )
        self.data_keys = dict(data_keys)
        self.pose_quaternion_order = str(pose_quaternion_order).lower()
        self.ft_cfg = dict(ft)
        self.pose_repr = dict(pose_repr)
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "relative")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "relative")
        self.action_padding = bool(action_padding)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.start_pose_noise_scale = np.asarray(
            start_pose_noise_scale, dtype=np.float64
        )
        self.threadpool_limits_is_applied = False

        self._validate_allowlist()
        self._prefix = detect_zarr_prefix(self.dataset_path).prefix
        self._zip_store = None
        self._zarr_root = None
        self._rgb_array = None
        self._open_pid = None

        store, root, detected = open_nested_zip_group(
            self.dataset_path, prefix=self._prefix
        )
        sidecar_store = None
        try:
            if detected != self._prefix:
                raise RuntimeError("nested Zarr prefix changed during open")
            if self.force_sidecar_path is None:
                raw_pose, raw_pose_format = self._load_multirate_root(root)
            else:
                self._sidecar_prefix = detect_zarr_prefix(
                    self.force_sidecar_path
                ).prefix
                sidecar_store, sidecar_root, sidecar_detected = (
                    open_nested_zip_group(
                        self.force_sidecar_path,
                        prefix=self._sidecar_prefix,
                    )
                )
                if sidecar_detected != self._sidecar_prefix:
                    raise RuntimeError("sidecar Zarr prefix changed during open")
                raw_pose, raw_pose_format = self._load_base_and_sidecar_roots(
                    root,
                    sidecar_root,
                )
        finally:
            if sidecar_store is not None:
                sidecar_store.close()
            store.close()

        if self.source_mode == "base_with_force_sidecar":
            self.grasp_force = self._derive_grasp_force()
        self._validate_raw_contract(raw_pose, raw_pose_format=raw_pose_format)
        self._apply_ft_bias()
        self._validate_grasp_force_contract()
        if raw_pose_format == "xyzw_pose7":
            self._convert_pose7(raw_pose)
        elif raw_pose_format == "axis_angle_pose6":
            self._convert_pose_axis_angle(raw_pose)
        else:
            raise AssertionError(f"unsupported raw pose format {raw_pose_format!r}")

        self.n_episodes = len(self.rgb_episode_ends)
        self.val_mask = get_val_mask(
            n_episodes=self.n_episodes,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        self.train_mask = ~self.val_mask

        self.raw_rgb_hz = 1.0 / np.median(
            _positive_dt(self.rgb_timestamps, self.rgb_episode_ends)
        )
        self.ft_left_hz = 1.0 / np.median(
            _positive_dt(self.ft_left_timestamps, self.ft_left_episode_ends)
        )
        self.ft_right_hz = 1.0 / np.median(
            _positive_dt(self.ft_right_timestamps, self.ft_right_episode_ends)
        )

        self.action_horizon = int(shape_meta["action"]["horizon"])
        self.action_down_sample_steps = int(
            shape_meta["action"]["down_sample_steps"]
        )
        self.action_frequency = self.raw_rgb_hz / self.action_down_sample_steps
        self.ft_num_steps = int(self.ft_cfg["num_steps"])
        self.ft_stride = int(self.ft_cfg["stride"])
        self.ft_history_seconds = float(self.ft_cfg["history_seconds"])
        if self.ft_cfg.get("padding", "repeat_first") != "repeat_first":
            raise ValueError("dual F/T history only supports padding=repeat_first")
        expected_history = (
            (self.ft_num_steps - 1) * self.ft_stride / self.ft_left_hz
        )
        if not np.isclose(
            expected_history,
            self.ft_history_seconds,
            atol=max(0.5 / self.ft_left_hz, 1e-6),
        ):
            raise ValueError(
                "ft.history_seconds is inconsistent with num_steps, stride, "
                f"and actual Hz: configured={self.ft_history_seconds:.6f}, "
                f"actual span={expected_history:.6f}"
            )

        self.causal_drop_report = self._build_causal_drop_report()
        self.indices = self._build_indices(self.train_mask)
        self.split = "train"
        self.normalizer_input_stats = None

        logger.info(
            "UmiDualFTDataset source_mode=%s prefix=%r episodes=%d train_samples=%d "
            "rgb_hz=%.6f ft_left_hz=%.6f ft_right_hz=%.6f action_hz=%.6f",
            self.source_mode,
            self._prefix,
            self.n_episodes,
            len(self.indices),
            self.raw_rgb_hz,
            self.ft_left_hz,
            self.ft_right_hz,
            self.action_frequency,
        )

    def _load_multirate_root(self, root):
        attrs = dict(root.attrs)
        data = root["data"]
        meta = root["meta"]

        self.dataset_attrs = attrs
        self.sidecar_attrs = None
        self.rgb_shape = tuple(data[self.data_keys["rgb"]].shape)
        self.rgb_dtype = np.dtype(data[self.data_keys["rgb"]].dtype)
        self.rgb_chunks = tuple(data[self.data_keys["rgb"]].chunks)

        self.rgb_timestamps = np.asarray(
            data[self.data_keys["rgb_timestamp"]][:], dtype=np.float64
        ).reshape(-1)
        self.robot_timestamps = np.asarray(
            data[self.data_keys["pose_timestamp"]][:], dtype=np.float64
        ).reshape(-1)
        self.gripper_timestamps = np.asarray(
            data[self.data_keys["gripper_timestamp"]][:], dtype=np.float64
        ).reshape(-1)
        raw_pose = np.asarray(data[self.data_keys["pose"]][:], dtype=np.float32)
        self.gripper_width = np.asarray(
            data[self.data_keys["gripper"]][:], dtype=np.float32
        )
        self.grasp_force = np.asarray(
            data[self.data_keys["grasp_force"]][:], dtype=np.float32
        )

        self.ft_left = np.asarray(
            data[self.ft_cfg["left_key"]][:], dtype=np.float32
        )
        self.ft_right = np.asarray(
            data[self.ft_cfg["right_key"]][:], dtype=np.float32
        )
        self.ft_left_timestamps = np.asarray(
            data[self.ft_cfg["left_timestamp_key"]][:], dtype=np.float64
        ).reshape(-1)
        self.ft_right_timestamps = np.asarray(
            data[self.ft_cfg["right_timestamp_key"]][:], dtype=np.float64
        ).reshape(-1)

        self.rgb_episode_ends = np.asarray(
            meta[self.data_keys["rgb_episode_ends"]][:], dtype=np.int64
        ).reshape(-1)
        self.robot_episode_ends = np.asarray(
            meta[self.data_keys["pose_episode_ends"]][:], dtype=np.int64
        ).reshape(-1)
        self.gripper_episode_ends = np.asarray(
            meta[self.data_keys["gripper_episode_ends"]][:], dtype=np.int64
        ).reshape(-1)
        self.ft_left_episode_ends = np.asarray(
            meta[self.ft_cfg["left_episode_ends_key"]][:], dtype=np.int64
        ).reshape(-1)
        self.ft_right_episode_ends = np.asarray(
            meta[self.ft_cfg["right_episode_ends_key"]][:], dtype=np.int64
        ).reshape(-1)
        bias_key = self.ft_cfg.get("bias_key")
        if bias_key is None:
            self.ft_episode_bias_12d = None
        else:
            if str(self.ft_cfg.get("bias_removal", "")) != "per_episode_metadata":
                raise ValueError(
                    "ft.bias_key requires bias_removal=per_episode_metadata"
                )
            self.ft_episode_bias_12d = np.asarray(
                meta[bias_key][:], dtype=np.float32
            )
        self._ft_values_are_bias_removed = False
        self.rgb_to_wrench_end_idx = None
        self.rgb_wrench_age_s = None
        self.rgb_wrench_valid = None
        return raw_pose, "xyzw_pose7"

    def _load_base_and_sidecar_roots(self, base_root, sidecar_root):
        base_data = base_root["data"]
        base_meta = base_root["meta"]
        sidecar_data = sidecar_root["data"]
        sidecar_meta = sidecar_root["meta"]
        self.dataset_attrs = dict(base_root.attrs)
        self.sidecar_attrs = dict(sidecar_root.attrs)

        schema = str(self.sidecar_attrs.get("schema", ""))
        if schema != "umi_force_sidecar_v1":
            raise ValueError(
                "force sidecar schema must be 'umi_force_sidecar_v1', got "
                f"{schema!r}"
            )
        wrench_key = str(self.ft_cfg.get("wrench_key", ""))
        if wrench_key != "wrench_12d":
            raise ValueError(
                "base+sidecar training only permits bias-removed native "
                f"data/wrench_12d, got {wrench_key!r}"
            )
        declared_train_key = str(self.sidecar_attrs.get("train_wrench_key", ""))
        if declared_train_key != wrench_key:
            raise ValueError(
                "sidecar train_wrench_key does not match configured wrench key: "
                f"{declared_train_key!r} != {wrench_key!r}"
            )
        expected_channels = [
            "fx_l", "fy_l", "fz_l", "tx_l", "ty_l", "tz_l",
            "fx_r", "fy_r", "fz_r", "tx_r", "ty_r", "tz_r",
        ]
        if list(self.sidecar_attrs.get("wrench_channel_order", ())) != expected_channels:
            raise ValueError("sidecar native 12-D wrench channel order is invalid")

        self.rgb_shape = tuple(base_data[self.data_keys["rgb"]].shape)
        self.rgb_dtype = np.dtype(base_data[self.data_keys["rgb"]].dtype)
        self.rgb_chunks = tuple(base_data[self.data_keys["rgb"]].chunks)
        eef_pos = np.asarray(
            base_data[self.data_keys["pose_position"]][:], dtype=np.float32
        )
        eef_rot_axis_angle = np.asarray(
            base_data[self.data_keys["pose_rotation_axis_angle"]][:],
            dtype=np.float32,
        )
        raw_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=-1)
        self.gripper_width = np.asarray(
            base_data[self.data_keys["gripper"]][:], dtype=np.float32
        )

        self.rgb_episode_ends = np.asarray(
            base_meta[self.data_keys["rgb_episode_ends"]][:], dtype=np.int64
        ).reshape(-1)
        sidecar_rgb_episode_ends = np.asarray(
            sidecar_meta[self.ft_cfg["rgb_episode_ends_key"]][:], dtype=np.int64
        ).reshape(-1)
        if not np.array_equal(self.rgb_episode_ends, sidecar_rgb_episode_ends):
            raise ValueError("base ZIP and force sidecar RGB episode ends differ")
        self.robot_episode_ends = self.rgb_episode_ends.copy()
        self.gripper_episode_ends = self.rgb_episode_ends.copy()

        self.rgb_timestamps = np.asarray(
            sidecar_data[self.ft_cfg["rgb_timestamp_key"]][:], dtype=np.float64
        ).reshape(-1)
        self.robot_timestamps = self.rgb_timestamps.copy()
        self.gripper_timestamps = self.rgb_timestamps.copy()

        wrench_12d = np.asarray(
            sidecar_data[wrench_key][:], dtype=np.float32
        )
        if wrench_12d.ndim != 2 or wrench_12d.shape[1] != 12:
            raise ValueError(f"sidecar wrench_12d must be [N,12], got {wrench_12d.shape}")
        self.ft_left = wrench_12d[:, :6].copy()
        self.ft_right = wrench_12d[:, 6:].copy()
        wrench_timestamps = np.asarray(
            sidecar_data[self.ft_cfg["timestamp_key"]][:], dtype=np.float64
        ).reshape(-1)
        self.ft_left_timestamps = wrench_timestamps
        self.ft_right_timestamps = wrench_timestamps.copy()
        wrench_episode_ends = np.asarray(
            sidecar_meta[self.ft_cfg["episode_ends_key"]][:], dtype=np.int64
        ).reshape(-1)
        self.ft_left_episode_ends = wrench_episode_ends
        self.ft_right_episode_ends = wrench_episode_ends.copy()

        bias_key = self.ft_cfg.get("bias_key")
        self.ft_episode_bias_12d = (
            None if bias_key is None else np.asarray(
                sidecar_meta[bias_key][:], dtype=np.float32
            )
        )
        if str(self.ft_cfg.get("bias_removal", "")) != "precomputed_in_sidecar":
            raise ValueError(
                "wrench_12d requires ft.bias_removal=precomputed_in_sidecar"
            )
        self._ft_values_are_bias_removed = True

        self.rgb_to_wrench_end_idx = np.asarray(
            sidecar_data[self.ft_cfg["rgb_to_wrench_end_idx_key"]][:],
            dtype=np.int64,
        ).reshape(-1)
        self.rgb_wrench_age_s = np.asarray(
            sidecar_data[self.ft_cfg["rgb_wrench_age_key"]][:], dtype=np.float64
        ).reshape(-1)
        self.rgb_wrench_valid = np.asarray(
            sidecar_data[self.ft_cfg["rgb_wrench_valid_key"]][:], dtype=bool
        ).reshape(-1)
        self._validate_sidecar_causal_mapping()
        return raw_pose, "axis_angle_pose6"

    @property
    def detected_prefix(self):
        return self._prefix

    def _validate_allowlist(self):
        obs_keys = set(self.shape_meta["obs"].keys())
        missing = REQUIRED_OBS_KEYS - obs_keys
        if missing:
            raise ValueError(f"dual-F/T shape_meta is missing required keys: {missing}")
        unexpected = obs_keys - ALLOWED_OBS_KEYS
        if unexpected:
            raise ValueError(
                "dual-F/T observation allowlist rejected keys: "
                f"{sorted(unexpected)}"
            )
        bad = sorted(
            key
            for key in obs_keys | set(self.shape_meta["action"].keys())
            if any(fragment in key.lower() for fragment in DISALLOWED_KEY_FRAGMENTS)
        )
        if bad:
            raise ValueError(f"disallowed observation/action keys: {bad}")
        action_shape = tuple(self.shape_meta["action"]["shape"])
        if action_shape != (11,):
            raise ValueError(
                "dual-F/T action must be 11D "
                "[xyz, rotation_6d, gripper_width, grasp_force], got "
                f"{action_shape}"
            )
        if self.pose_quaternion_order != POSE_QUATERNION_ORDER:
            raise ValueError(
                "dual-F/T pose quaternion order must be explicitly declared "
                f"as {POSE_QUATERNION_ORDER!r}, got "
                f"{self.pose_quaternion_order!r}"
            )

    def _validate_raw_contract(self, raw_pose, *, raw_pose_format):
        n_rgb = len(self.rgb_timestamps)
        for name, array in (
            ("pose", raw_pose),
            ("robot timestamps", self.robot_timestamps),
            ("gripper", self.gripper_width),
            ("gripper timestamps", self.gripper_timestamps),
            ("grasp force", self.grasp_force),
        ):
            if len(array) != n_rgb:
                raise ValueError(f"{name} length {len(array)} != RGB length {n_rgb}")
        if self.rgb_shape != (n_rgb, 224, 224, 3):
            raise ValueError(f"unexpected RGB shape {self.rgb_shape}")
        if self.rgb_dtype != np.dtype(np.uint8):
            raise ValueError(f"unexpected RGB dtype {self.rgb_dtype}")
        if raw_pose_format == "xyzw_pose7":
            if raw_pose.shape != (n_rgb, 7):
                raise ValueError(
                    "pose must be [N,7] [x,y,z,qx,qy,qz,qw], "
                    f"got {raw_pose.shape}"
                )
        elif raw_pose_format == "axis_angle_pose6":
            if raw_pose.shape != (n_rgb, 6):
                raise ValueError(
                    "base pose must be [N,6] [x,y,z,rx,ry,rz], "
                    f"got {raw_pose.shape}"
                )
        else:
            raise ValueError(f"unknown raw pose format {raw_pose_format!r}")
        if self.gripper_width.shape != (n_rgb, 1):
            raise ValueError(f"gripper must be [N,1], got {self.gripper_width.shape}")
        if self.grasp_force.shape != (n_rgb, 1):
            raise ValueError(
                f"grasp force must be [N,1], got {self.grasp_force.shape}"
            )
        if self.ft_left.ndim != 2 or self.ft_left.shape[1] != 6:
            raise ValueError(f"left F/T must be [N,6], got {self.ft_left.shape}")
        if self.ft_right.ndim != 2 or self.ft_right.shape[1] != 6:
            raise ValueError(f"right F/T must be [N,6], got {self.ft_right.shape}")
        if len(self.ft_left) != len(self.ft_left_timestamps):
            raise ValueError("left F/T value/timestamp length mismatch")
        if len(self.ft_right) != len(self.ft_right_timestamps):
            raise ValueError("right F/T value/timestamp length mismatch")

        episode_arrays = (
            self.rgb_episode_ends,
            self.robot_episode_ends,
            self.gripper_episode_ends,
            self.ft_left_episode_ends,
            self.ft_right_episode_ends,
        )
        if len({len(x) for x in episode_arrays}) != 1:
            raise ValueError("modality episode counts do not match")
        if self.rgb_episode_ends[-1] != n_rgb:
            raise ValueError("RGB cumulative episode ends do not match RGB length")
        if not np.array_equal(self.robot_episode_ends, self.rgb_episode_ends):
            raise ValueError("robot and RGB episode timelines differ")
        if not np.array_equal(self.gripper_episode_ends, self.rgb_episode_ends):
            raise ValueError("gripper and RGB episode timelines differ")
        if self.ft_left_episode_ends[-1] != len(self.ft_left):
            raise ValueError("left F/T episode ends do not match value length")
        if self.ft_right_episode_ends[-1] != len(self.ft_right):
            raise ValueError("right F/T episode ends do not match value length")

        permutation = tuple(int(x) for x in self.ft_cfg.get(
            "axis_permutation", range(6)
        ))
        signs = tuple(float(x) for x in self.ft_cfg.get("axis_sign", [1] * 6))
        if permutation != tuple(range(6)) or signs != (1.0,) * 6:
            raise ValueError(
                "dual-F/T native-frame contract forbids coordinate transforms: "
                f"axis_permutation={permutation}, axis_sign={signs}"
            )
        for side in ("left", "right"):
            frame = str(self.ft_cfg.get(f"{side}_frame", ""))
            if frame != f"{side}_native_sensor":
                raise ValueError(
                    f"{side} F/T frame must be {side}_native_sensor, got {frame!r}"
                )

        if self.ft_episode_bias_12d is not None:
            expected = (len(self.ft_left_episode_ends), 12)
            if self.ft_episode_bias_12d.shape != expected:
                raise ValueError(
                    f"episode 12-D F/T bias must be {expected}, got "
                    f"{self.ft_episode_bias_12d.shape}"
                )
            if np.any(~np.isfinite(self.ft_episode_bias_12d)):
                raise ValueError("episode 12-D F/T bias contains NaN or Inf")

        for name, a, b in (
            ("robot", self.robot_timestamps, self.rgb_timestamps),
            ("gripper", self.gripper_timestamps, self.rgb_timestamps),
        ):
            if not np.array_equal(a, b):
                raise ValueError(f"{name} timestamp grid differs from RGB anchor grid")
        for name, array in (
            ("pose", raw_pose),
            ("gripper", self.gripper_width),
            ("grasp force", self.grasp_force),
            ("left F/T", self.ft_left),
            ("right F/T", self.ft_right),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")

    def _apply_ft_bias(self):
        if self._ft_values_are_bias_removed:
            self.ft_bias_removed = True
            return
        self.ft_bias_removed = self.ft_episode_bias_12d is not None
        if not self.ft_bias_removed:
            return
        self.ft_left = _subtract_episode_bias(
            self.ft_left,
            self.ft_left_episode_ends,
            self.ft_episode_bias_12d[:, :6],
        )
        self.ft_right = _subtract_episode_bias(
            self.ft_right,
            self.ft_right_episode_ends,
            self.ft_episode_bias_12d[:, 6:],
        )

    def _validate_sidecar_causal_mapping(self):
        n_rgb = len(self.rgb_timestamps)
        for name, value in (
            ("rgb_to_wrench_end_idx", self.rgb_to_wrench_end_idx),
            ("rgb_wrench_age_s", self.rgb_wrench_age_s),
            ("rgb_wrench_valid", self.rgb_wrench_valid),
        ):
            if len(value) != n_rgb:
                raise ValueError(f"sidecar {name} length {len(value)} != RGB length {n_rgb}")
        if not np.array_equal(self.rgb_wrench_valid, self.rgb_to_wrench_end_idx >= 0):
            raise ValueError("sidecar valid mask and causal wrench index disagree")

        rgb_slices = _cumulative_slices(self.rgb_episode_ends)
        wrench_slices = _cumulative_slices(self.ft_left_episode_ends)
        for episode, ((rgb_start, rgb_end), (wrench_start, wrench_end)) in enumerate(
            zip(rgb_slices, wrench_slices)
        ):
            anchor_t = self.rgb_timestamps[rgb_start:rgb_end]
            wrench_t = self.ft_left_timestamps[wrench_start:wrench_end]
            expected_local = np.searchsorted(wrench_t, anchor_t, side="right") - 1
            expected_global = np.full(expected_local.shape, -1, dtype=np.int64)
            valid = expected_local >= 0
            expected_global[valid] = wrench_start + expected_local[valid]
            stored_global = self.rgb_to_wrench_end_idx[rgb_start:rgb_end]
            if not np.array_equal(stored_global, expected_global):
                mismatch = int(np.flatnonzero(stored_global != expected_global)[0])
                raise ValueError(
                    "sidecar causal mapping does not match its timestamps in "
                    f"episode {episode}, local RGB index {mismatch}"
                )
            stored_age = self.rgb_wrench_age_s[rgb_start:rgb_end]
            if np.any(~np.isnan(stored_age[~valid])):
                raise ValueError("invalid sidecar RGB anchors must have NaN wrench age")
            expected_age = anchor_t[valid] - wrench_t[expected_local[valid]]
            if np.any(expected_age < 0) or not np.allclose(
                stored_age[valid], expected_age, rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"sidecar causal wrench ages are invalid in episode {episode}"
                )

    def _derive_grasp_force(self):
        """Create the signed force action label from bias-removed native Fz."""

        result = np.empty((len(self.rgb_timestamps), 1), dtype=np.float32)
        rgb_slices = _cumulative_slices(self.rgb_episode_ends)
        left_slices = _cumulative_slices(self.ft_left_episode_ends)
        right_slices = _cumulative_slices(self.ft_right_episode_ends)
        for (rgb_start, rgb_end), (left_start, left_end), (
            right_start,
            right_end,
        ) in zip(rgb_slices, left_slices, right_slices):
            anchor_t = self.rgb_timestamps[rgb_start:rgb_end]
            left_fz = np.interp(
                anchor_t,
                self.ft_left_timestamps[left_start:left_end],
                self.ft_left[left_start:left_end, 2],
            )
            right_fz = np.interp(
                anchor_t,
                self.ft_right_timestamps[right_start:right_end],
                self.ft_right[right_start:right_end, 2],
            )
            result[rgb_start:rgb_end, 0] = 0.5 * (right_fz - left_fz)
        return result

    def _validate_grasp_force_contract(self):
        """Verify the 11th action label against effective native Fz streams.

        The supervised scalar is a signed measurement, not a gripper command:
        half of right-minus-left native Fz after the configured episode bias,
        linearly interpolated onto each RGB/action anchor.
        """

        maximum_error = 0.0
        rgb_slices = _cumulative_slices(self.rgb_episode_ends)
        left_slices = _cumulative_slices(self.ft_left_episode_ends)
        right_slices = _cumulative_slices(self.ft_right_episode_ends)
        for episode, ((rgb_start, rgb_end), (left_start, left_end), (
            right_start,
            right_end,
        )) in enumerate(zip(rgb_slices, left_slices, right_slices)):
            anchor_t = self.rgb_timestamps[rgb_start:rgb_end]
            left_interp = np.interp(
                anchor_t,
                self.ft_left_timestamps[left_start:left_end],
                self.ft_left[left_start:left_end, 2],
            )
            right_interp = np.interp(
                anchor_t,
                self.ft_right_timestamps[right_start:right_end],
                self.ft_right[right_start:right_end, 2],
            )
            expected = 0.5 * (right_interp - left_interp)
            stored = self.grasp_force[rgb_start:rgb_end, 0]
            error = np.abs(expected - stored)
            this_max = float(np.max(error)) if len(error) else 0.0
            maximum_error = max(maximum_error, this_max)
            if not np.allclose(expected, stored, rtol=1e-5, atol=5e-5):
                worst = int(np.argmax(error))
                raise ValueError(
                    "grasp_force_0 violates the signed native-Fz measurement "
                    f"contract in episode {episode}, local index {worst}: "
                    f"stored={float(stored[worst]):.9g} "
                    f"expected={float(expected[worst]):.9g} "
                    f"error={float(error[worst]):.9g}"
                )
        self.grasp_force_contract_max_abs_error = maximum_error

    def _convert_pose7(self, pose7):
        # ts_pose_fb_0 stores [x,y,z,qx,qy,qz,qw], exactly SciPy's scalar-last
        # convention. Reordering these components would change the rotation.
        quat_xyzw = pose7[:, 3:].astype(np.float64)
        quat_norm = np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
        if np.any(quat_norm <= np.finfo(np.float64).eps):
            raise ValueError("pose contains a zero-norm quaternion")
        quat_xyzw /= quat_norm
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        pose_axis_angle = np.concatenate([pose7[:, :3], rotvec], axis=-1)
        self._convert_pose_axis_angle(pose_axis_angle)

    def _convert_pose_axis_angle(self, pose_axis_angle):
        """Use the stock UMI base pose without a quaternion round trip."""

        pose_axis_angle = np.asarray(pose_axis_angle, dtype=np.float32)
        if pose_axis_angle.ndim != 2 or pose_axis_angle.shape[1] != 6:
            raise ValueError(
                f"axis-angle pose must be [N,6], got {pose_axis_angle.shape}"
            )
        self.eef_pos = pose_axis_angle[:, :3].copy()
        self.eef_rot_axis_angle = pose_axis_angle[:, 3:].copy()
        self.pose_mats = pose_to_mat(
            np.concatenate([self.eef_pos, self.eef_rot_axis_angle], axis=-1)
        )
        self.episode_start_pose = np.stack(
            [
                np.concatenate([self.eef_pos[start], self.eef_rot_axis_angle[start]])
                for start, _ in _cumulative_slices(self.rgb_episode_ends)
            ]
        )

    def _build_causal_drop_report(self):
        report = []
        rgb_slices = _cumulative_slices(self.rgb_episode_ends)
        left_slices = _cumulative_slices(self.ft_left_episode_ends)
        right_slices = _cumulative_slices(self.ft_right_episode_ends)
        for episode, ((rs, re), (ls, le), (xs, xe)) in enumerate(
            zip(rgb_slices, left_slices, right_slices)
        ):
            anchors = self.rgb_timestamps[rs:re]
            if self.rgb_wrench_valid is not None:
                has_left = self.rgb_wrench_valid[rs:re]
                has_right = self.rgb_wrench_valid[rs:re]
            else:
                left_t = self.ft_left_timestamps[ls:le]
                right_t = self.ft_right_timestamps[xs:xe]
                has_left = np.searchsorted(left_t, anchors, side="right") > 0
                has_right = np.searchsorted(right_t, anchors, side="right") > 0
            valid = has_left & has_right
            dropped = int(np.sum(~valid))
            report.append(
                {
                    "episode": episode,
                    "anchors": int(len(anchors)),
                    "dropped": dropped,
                    "ratio": float(dropped / len(anchors)),
                    "left_missing": int(np.sum(~has_left)),
                    "right_missing": int(np.sum(~has_right)),
                }
            )
        return report

    def _build_indices(self, episode_mask):
        indices = []
        rgb_slices = _cumulative_slices(self.rgb_episode_ends)
        left_slices = _cumulative_slices(self.ft_left_episode_ends)
        right_slices = _cumulative_slices(self.ft_right_episode_ends)
        for episode, ((start, end), (ls, le), (rs, re)) in enumerate(
            zip(rgb_slices, left_slices, right_slices)
        ):
            if not episode_mask[episode]:
                continue
            left_t = self.ft_left_timestamps[ls:le]
            right_t = self.ft_right_timestamps[rs:re]
            for current in range(int(start), int(end)):
                anchor = self.rgb_timestamps[current]
                if self.rgb_wrench_valid is not None:
                    if not self.rgb_wrench_valid[current]:
                        continue
                else:
                    if np.searchsorted(left_t, anchor, side="right") == 0:
                        continue
                    if np.searchsorted(right_t, anchor, side="right") == 0:
                        continue
                required_end = (
                    current
                    + (self.action_horizon - 1) * self.action_down_sample_steps
                    + 1
                )
                if not self.action_padding and required_end > end:
                    continue
                indices.append((episode, current))
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.close()
        val_set._zip_store = None
        val_set._zarr_root = None
        val_set._rgb_array = None
        val_set._open_pid = None
        val_set.indices = val_set._build_indices(self.val_mask)
        val_set.split = "validation"
        val_set.train_mask = self.val_mask.copy()
        val_set.val_mask = ~self.val_mask
        return val_set

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_store"] = None
        state["_zarr_root"] = None
        state["_rgb_array"] = None
        state["_open_pid"] = None
        return state

    def close(self):
        if self._zip_store is not None:
            self._zip_store.close()
        self._zip_store = None
        self._zarr_root = None
        self._rgb_array = None
        self._open_pid = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_rgb_array(self):
        pid = os.getpid()
        if self._rgb_array is None or self._open_pid != pid:
            self.close()
            store, root, prefix = open_nested_zip_group(
                self.dataset_path, prefix=self._prefix
            )
            if prefix != self._prefix:
                store.close()
                raise RuntimeError("detected Zarr prefix changed")
            self._zip_store = store
            self._zarr_root = root
            self._rgb_array = root["data"][self.data_keys["rgb"]]
            self._open_pid = pid
            logger.info("Worker pid=%d lazily opened RGB ZipStore", pid)
        return self._rgb_array

    @staticmethod
    def _history_indices(current, episode_start, horizon, down_sample_steps):
        idx = current - np.arange(horizon - 1, -1, -1) * down_sample_steps
        return np.maximum(idx, episode_start).astype(np.int64)

    def _ft_history(
        self,
        *,
        values,
        timestamps,
        episode_ends,
        episode,
        anchor,
        actual_hz,
        latest_global_idx=None,
    ):
        start, end = _cumulative_slices(episode_ends)[episode]
        episode_t = timestamps[start:end]
        expected_latest = np.searchsorted(episode_t, anchor, side="right") - 1
        if latest_global_idx is None:
            latest = expected_latest
        else:
            latest = int(latest_global_idx) - int(start)
            if latest != expected_latest:
                raise RuntimeError(
                    "precomputed causal F/T index disagrees with timestamp search"
                )
        if latest < 0:
            raise RuntimeError("anchor has no causal F/T sample")
        target_t = anchor - (
            np.arange(self.ft_num_steps - 1, -1, -1, dtype=np.float64)
            * self.ft_stride
            / actual_hz
        )
        local_idx = np.searchsorted(episode_t, target_t, side="right") - 1
        # Once one causal sample exists at the anchor, history slots preceding
        # the episode's first F/T use that earliest causal value.
        local_idx = np.maximum(local_idx, 0)
        selected_t = episode_t[local_idx]
        if np.any(selected_t > anchor):
            raise AssertionError("future F/T sample selected")
        global_idx = start + local_idx
        return values[global_idx].astype(np.float32), selected_t

    def _sample_arrays(self, idx, *, load_rgb):
        episode, current = self.indices[idx]
        episode_start, episode_end = _cumulative_slices(self.rgb_episode_ends)[episode]
        anchor = float(self.rgb_timestamps[current])

        obs_meta = self.shape_meta["obs"]
        image_attr = obs_meta["camera0_rgb"]
        pose_attr = obs_meta["robot0_eef_pos"]
        image_idx = self._history_indices(
            current,
            episode_start,
            int(image_attr["horizon"]),
            int(image_attr["down_sample_steps"]),
        )
        pose_idx = self._history_indices(
            current,
            episode_start,
            int(pose_attr["horizon"]),
            int(pose_attr["down_sample_steps"]),
        )

        pose_mat = self.pose_mats[pose_idx]
        relative_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=pose_mat[-1],
            pose_rep=self.obs_pose_repr,
            backward=False,
        )
        relative_pose10d = mat_to_pose10d(relative_pose_mat)

        obs = {
            "robot0_eef_pos": relative_pose10d[:, :3].astype(np.float32),
            "robot0_eef_rot_axis_angle": relative_pose10d[:, 3:].astype(np.float32),
        }

        gripper_idx = None
        if "robot0_gripper_width" in obs_meta:
            gripper_attr = obs_meta["robot0_gripper_width"]
            gripper_idx = self._history_indices(
                current,
                episode_start,
                int(gripper_attr["horizon"]),
                int(gripper_attr["down_sample_steps"]),
            )
            obs["robot0_gripper_width"] = self.gripper_width[
                gripper_idx
            ].astype(np.float32)

        if "robot0_eef_rot_axis_angle_wrt_start" in obs_meta:
            start_pose = self.episode_start_pose[episode].astype(np.float64).copy()
            if self.split == "train" and np.any(self.start_pose_noise_scale):
                start_pose += np.random.normal(scale=self.start_pose_noise_scale)
            start_pose_mat = pose_to_mat(start_pose)
            wrt_start_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep="relative",
                backward=False,
            )
            obs["robot0_eef_rot_axis_angle_wrt_start"] = mat_to_pose10d(
                wrt_start_mat
            )[:, 3:].astype(np.float32)

        left, left_t = self._ft_history(
            values=self.ft_left,
            timestamps=self.ft_left_timestamps,
            episode_ends=self.ft_left_episode_ends,
            episode=episode,
            anchor=anchor,
            actual_hz=self.ft_left_hz,
            latest_global_idx=(
                None if self.rgb_to_wrench_end_idx is None
                else self.rgb_to_wrench_end_idx[current]
            ),
        )
        right, right_t = self._ft_history(
            values=self.ft_right,
            timestamps=self.ft_right_timestamps,
            episode_ends=self.ft_right_episode_ends,
            episode=episode,
            anchor=anchor,
            actual_hz=self.ft_right_hz,
            latest_global_idx=(
                None if self.rgb_to_wrench_end_idx is None
                else self.rgb_to_wrench_end_idx[current]
            ),
        )
        obs["robot0_ft_left"] = left
        obs["robot0_ft_right"] = right

        if load_rgb:
            rgb_array = self._get_rgb_array()
            rgb = np.stack([rgb_array[int(i)] for i in image_idx], axis=0)
            expected = (len(image_idx),) + self.rgb_shape[1:]
            if rgb.shape != expected or rgb.dtype != self.rgb_dtype:
                raise ValueError(
                    f"decoded RGB contract mismatch: got {rgb.shape}/{rgb.dtype}, "
                    f"expected {expected}/{self.rgb_dtype}"
                )
            obs["camera0_rgb"] = np.moveaxis(rgb, -1, 1).astype(np.float32) / 255.0

        action_end = min(
            episode_end,
            current + (self.action_horizon - 1) * self.action_down_sample_steps + 1,
        )
        action_idx = np.arange(
            current, action_end, self.action_down_sample_steps, dtype=np.int64
        )
        if len(action_idx) < self.action_horizon:
            if not self.action_padding:
                raise AssertionError("short action without action padding")
            action_idx = np.r_[
                action_idx,
                np.repeat(action_idx[-1], self.action_horizon - len(action_idx)),
            ]
        action_mat = self.pose_mats[action_idx]
        action_relative_mat = convert_pose_mat_rep(
            action_mat,
            base_pose_mat=pose_mat[-1],
            pose_rep=self.action_pose_repr,
            backward=False,
        )
        action_pose10d = mat_to_pose10d(action_relative_mat)
        action = np.concatenate(
            [
                action_pose10d,
                self.gripper_width[action_idx],
                self.grasp_force[action_idx],
            ],
            axis=-1,
        ).astype(np.float32)

        observation_timestamp_parts = [
            self.rgb_timestamps[image_idx],
            self.robot_timestamps[pose_idx],
            left_t,
            right_t,
        ]
        if gripper_idx is not None:
            observation_timestamp_parts.append(self.gripper_timestamps[gripper_idx])
        observation_timestamps = np.concatenate(observation_timestamp_parts)
        if np.max(observation_timestamps) > anchor:
            raise AssertionError("observation timestamp exceeds policy anchor")

        sample_info = {
            "episode_index": np.asarray(episode, dtype=np.int64),
            "anchor_timestamp": np.asarray(anchor, dtype=np.float64),
            "rgb_timestamps": self.rgb_timestamps[image_idx].astype(np.float64),
            "pose_timestamps": self.robot_timestamps[pose_idx].astype(np.float64),
            "left_ft_timestamps": left_t.astype(np.float64),
            "right_ft_timestamps": right_t.astype(np.float64),
            "action_timestamps": self.robot_timestamps[action_idx].astype(np.float64),
            "left_ft_age": np.asarray(anchor - left_t[-1], dtype=np.float64),
            "right_ft_age": np.asarray(anchor - right_t[-1], dtype=np.float64),
        }
        return obs, action, sample_info

    def get_normalizer(self, **kwargs):
        # F/T statistics are computed only from episodes in this dataset's
        # train mask. Left/right and all six channels remain independent.
        left_train = []
        right_train = []
        gripper_train = []
        for episode, selected in enumerate(self.train_mask):
            if not selected:
                continue
            ls, le = _cumulative_slices(self.ft_left_episode_ends)[episode]
            rs, re = _cumulative_slices(self.ft_right_episode_ends)[episode]
            gs, ge = _cumulative_slices(self.gripper_episode_ends)[episode]
            left_train.append(self.ft_left[ls:le])
            right_train.append(self.ft_right[rs:re])
            gripper_train.append(self.gripper_width[gs:ge])
        left_stat = _stats(np.concatenate(left_train, axis=0))
        right_stat = _stats(np.concatenate(right_train, axis=0))
        gripper_stat = _stats(np.concatenate(gripper_train, axis=0))

        obs_pos = _StatsAccumulator(3)
        obs_rot = _StatsAccumulator(6)
        obs_start_rot = _StatsAccumulator(6)
        action_pos = _StatsAccumulator(3)
        action_rot = _StatsAccumulator(6)
        action_gripper = _StatsAccumulator(1)
        action_grasp_force = _StatsAccumulator(1)
        for idx in range(len(self.indices)):
            obs, action, _ = self._sample_arrays(idx, load_rgb=False)
            obs_pos.update(obs["robot0_eef_pos"])
            obs_rot.update(obs["robot0_eef_rot_axis_angle"])
            if "robot0_eef_rot_axis_angle_wrt_start" in obs:
                obs_start_rot.update(obs["robot0_eef_rot_axis_angle_wrt_start"])
            action_pos.update(action[:, :3])
            action_rot.update(action[:, 3:9])
            action_gripper.update(action[:, 9:10])
            action_grasp_force.update(action[:, 10:11])

        normalizer = LinearNormalizer()
        normalizer["camera0_rgb"] = get_image_identity_normalizer()
        normalizer["robot0_eef_pos"] = get_range_normalizer_from_stat(
            obs_pos.result()
        )
        normalizer["robot0_eef_rot_axis_angle"] = get_identity_normalizer_from_stat(
            obs_rot.result()
        )
        if "robot0_gripper_width" in self.shape_meta["obs"]:
            normalizer["robot0_gripper_width"] = get_range_normalizer_from_stat(
                gripper_stat
            )
        if "robot0_eef_rot_axis_angle_wrt_start" in self.shape_meta["obs"]:
            normalizer["robot0_eef_rot_axis_angle_wrt_start"] = (
                get_identity_normalizer_from_stat(obs_start_rot.result())
            )
        normalizer["robot0_ft_left"] = get_range_normalizer_from_stat(left_stat)
        normalizer["robot0_ft_right"] = get_range_normalizer_from_stat(right_stat)
        normalizer["action"] = concatenate_normalizer(
            [
                get_range_normalizer_from_stat(action_pos.result()),
                get_identity_normalizer_from_stat(action_rot.result()),
                get_range_normalizer_from_stat(action_gripper.result()),
                get_range_normalizer_from_stat(action_grasp_force.result()),
            ]
        )
        self.normalizer_input_stats = {
            "robot0_ft_left": left_stat,
            "robot0_ft_right": right_stat,
        }
        return normalizer

    def describe_sample(self, idx=0):
        obs, action, info = self._sample_arrays(idx, load_rgb=True)
        return {
            "rgb": {
                "shape": tuple(obs["camera0_rgb"].shape),
                "timestamps": info["rgb_timestamps"].tolist(),
            },
            "pose": {
                "position_shape": tuple(obs["robot0_eef_pos"].shape),
                "rotation_shape": tuple(obs["robot0_eef_rot_axis_angle"].shape),
                "timestamps": info["pose_timestamps"].tolist(),
            },
            "left_ft": {
                "shape": tuple(obs["robot0_ft_left"].shape),
                "first_timestamp": float(info["left_ft_timestamps"][0]),
                "last_timestamp": float(info["left_ft_timestamps"][-1]),
            },
            "right_ft": {
                "shape": tuple(obs["robot0_ft_right"].shape),
                "first_timestamp": float(info["right_ft_timestamps"][0]),
                "last_timestamp": float(info["right_ft_timestamps"][-1]),
            },
            "action": {
                "shape": tuple(action.shape),
                "start_timestamp": float(info["action_timestamps"][0]),
                "end_timestamp": float(info["action_timestamps"][-1]),
            },
            "anchor_timestamp": float(info["anchor_timestamp"]),
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs, action, info = self._sample_arrays(idx, load_rgb=True)
        return {
            "obs": dict_apply(obs, torch.from_numpy),
            "action": torch.from_numpy(action),
            "sample_info": dict_apply(info, torch.from_numpy),
        }
