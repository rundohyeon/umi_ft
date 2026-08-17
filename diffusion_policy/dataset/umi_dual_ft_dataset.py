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
    "robot0_gripper_width",
    "robot0_ft_left",
    "robot0_ft_right",
}

ALLOWED_OBS_KEYS = REQUIRED_OBS_KEYS | {
    "robot0_eef_rot_axis_angle_wrt_start",
}

DISALLOWED_KEY_FRAGMENTS = (
    "stiffness",
    "virtual_target",
    "grasp_force",
    "wrist",
    "depth",
    "camera1",
)


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
    """Allowlisted RGB+pose+dual-native-wrench multirate dataset.

    Low-dimensional arrays are copied into process-local NumPy memory during
    construction. JPEG-XL RGB remains in the read-only ZIP and is opened
    lazily by each DataLoader worker, so a ``ZipStore`` is never shared across
    processes.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        data_keys: dict,
        ft: dict,
        pose_repr: dict = {},
        action_padding: bool = False,
        seed: int = 42,
        val_ratio: float = 0.05,
        start_pose_noise_scale=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
    ):
        self.shape_meta = shape_meta
        self.dataset_path = str(dataset_path)
        self.data_keys = dict(data_keys)
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
        try:
            if detected != self._prefix:
                raise RuntimeError("nested Zarr prefix changed during open")
            attrs = dict(root.attrs)
            data = root["data"]
            meta = root["meta"]

            self.dataset_attrs = attrs
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
            pose7 = np.asarray(data[self.data_keys["pose"]][:], dtype=np.float32)
            self.gripper_width = np.asarray(
                data[self.data_keys["gripper"]][:], dtype=np.float32
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
        finally:
            store.close()

        self._validate_raw_contract(pose7)
        self._convert_pose7(pose7)

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
            "UmiDualFTDataset prefix=%r episodes=%d train_samples=%d "
            "rgb_hz=%.6f ft_left_hz=%.6f ft_right_hz=%.6f action_hz=%.6f",
            self._prefix,
            self.n_episodes,
            len(self.indices),
            self.raw_rgb_hz,
            self.ft_left_hz,
            self.ft_right_hz,
            self.action_frequency,
        )

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
        if action_shape != (10,):
            raise ValueError(
                f"TARGET coordinate action must remain 10D, got {action_shape}"
            )

    def _validate_raw_contract(self, pose7):
        n_rgb = len(self.rgb_timestamps)
        for name, array in (
            ("pose", pose7),
            ("robot timestamps", self.robot_timestamps),
            ("gripper", self.gripper_width),
            ("gripper timestamps", self.gripper_timestamps),
        ):
            if len(array) != n_rgb:
                raise ValueError(f"{name} length {len(array)} != RGB length {n_rgb}")
        if self.rgb_shape != (n_rgb, 224, 224, 3):
            raise ValueError(f"unexpected RGB shape {self.rgb_shape}")
        if self.rgb_dtype != np.dtype(np.uint8):
            raise ValueError(f"unexpected RGB dtype {self.rgb_dtype}")
        if pose7.shape != (n_rgb, 7):
            raise ValueError(
                "pose must be [N,7] [x,y,z,qw,qx,qy,qz], "
                f"got {pose7.shape}"
            )
        if self.gripper_width.shape != (n_rgb, 1):
            raise ValueError(f"gripper must be [N,1], got {self.gripper_width.shape}")
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

        for name, a, b in (
            ("robot", self.robot_timestamps, self.rgb_timestamps),
            ("gripper", self.gripper_timestamps, self.rgb_timestamps),
        ):
            if not np.array_equal(a, b):
                raise ValueError(f"{name} timestamp grid differs from RGB anchor grid")
        for name, array in (
            ("pose", pose7),
            ("gripper", self.gripper_width),
            ("left F/T", self.ft_left),
            ("right F/T", self.ft_right),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")

    def _convert_pose7(self, pose7):
        # Dataset pose7 contract is [x,y,z,qw,qx,qy,qz]. SciPy consumes xyzw.
        quat_wxyz = pose7[:, 3:].astype(np.float64)
        quat_wxyz /= np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
        quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=-1)
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        self.eef_pos = pose7[:, :3].astype(np.float32)
        self.eef_rot_axis_angle = rotvec.astype(np.float32)
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
    ):
        start, end = _cumulative_slices(episode_ends)[episode]
        episode_t = timestamps[start:end]
        latest = np.searchsorted(episode_t, anchor, side="right") - 1
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
        gripper_attr = obs_meta["robot0_gripper_width"]
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
        gripper_idx = self._history_indices(
            current,
            episode_start,
            int(gripper_attr["horizon"]),
            int(gripper_attr["down_sample_steps"]),
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
            "robot0_gripper_width": self.gripper_width[gripper_idx].astype(np.float32),
        }

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
        )
        right, right_t = self._ft_history(
            values=self.ft_right,
            timestamps=self.ft_right_timestamps,
            episode_ends=self.ft_right_episode_ends,
            episode=episode,
            anchor=anchor,
            actual_hz=self.ft_right_hz,
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
            [action_pose10d, self.gripper_width[action_idx]], axis=-1
        ).astype(np.float32)

        observation_timestamps = np.concatenate(
            [
                self.rgb_timestamps[image_idx],
                self.robot_timestamps[pose_idx],
                self.gripper_timestamps[gripper_idx],
                left_t,
                right_t,
            ]
        )
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
        for idx in range(len(self.indices)):
            obs, action, _ = self._sample_arrays(idx, load_rgb=False)
            obs_pos.update(obs["robot0_eef_pos"])
            obs_rot.update(obs["robot0_eef_rot_axis_angle"])
            if "robot0_eef_rot_axis_angle_wrt_start" in obs:
                obs_start_rot.update(obs["robot0_eef_rot_axis_angle_wrt_start"])
            action_pos.update(action[:, :3])
            action_rot.update(action[:, 3:9])
            action_gripper.update(action[:, 9:10])

        normalizer = LinearNormalizer()
        normalizer["camera0_rgb"] = get_image_identity_normalizer()
        normalizer["robot0_eef_pos"] = get_range_normalizer_from_stat(
            obs_pos.result()
        )
        normalizer["robot0_eef_rot_axis_angle"] = get_identity_normalizer_from_stat(
            obs_rot.result()
        )
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
