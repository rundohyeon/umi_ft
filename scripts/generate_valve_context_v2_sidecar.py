#!/usr/bin/env python3
"""Generate a frozen four-state valve-context sidecar for action training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.nested_zarr import open_nested_zip_group
from diffusion_policy.common.valve_context_contract import (
    VALVE_CONTEXT_V2_DIM,
    VALVE_CONTEXT_V2_PHASE_NAMES,
    VALVE_CONTEXT_V2_SCHEMA,
    validate_valve_context_v2,
)
from diffusion_policy.model.vision.valve_context_observer_v2 import (
    OBSERVER_FORCE_HISTORY_SAMPLES,
    OBSERVER_RGB_STRIDE,
    load_frozen_context_observer,
)
from umi.common.pose_util import mat_to_pose10d, pose_to_mat
from umi.real_world.valve_state_context import sha256_file


register_codecs()


def _starts(ends: np.ndarray) -> np.ndarray:
    return np.r_[0, np.asarray(ends, dtype=np.int64)[:-1]]


def _episode_for_indices(indices: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.searchsorted(ends, indices, side="right").astype(np.int64)


def _force_history(
    rgb_index: int,
    *,
    rgb_episode_ends: np.ndarray,
    wrench_episode_starts: np.ndarray,
    wrench_episode_ends: np.ndarray,
    rgb_to_wrench_end_idx: np.ndarray,
    wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    episode = int(np.searchsorted(rgb_episode_ends, rgb_index, side="right"))
    wrench_start = int(wrench_episode_starts[episode])
    wrench_end = int(wrench_episode_ends[episode])
    end_idx = int(rgb_to_wrench_end_idx[rgb_index])
    history = np.zeros((OBSERVER_FORCE_HISTORY_SAMPLES, 12), dtype=np.float32)
    mask = np.zeros((OBSERVER_FORCE_HISTORY_SAMPLES, 1), dtype=np.float32)
    if end_idx < 0:
        return history, mask
    if not wrench_start <= end_idx < wrench_end:
        raise ValueError("force-sidecar causal index crosses an episode boundary")
    first = max(wrench_start, end_idx - OBSERVER_FORCE_HISTORY_SAMPLES + 1)
    samples = wrench[first:end_idx + 1]
    count = len(samples)
    history[-count:] = samples
    mask[-count:, 0] = 1.0
    if count < OBSERVER_FORCE_HISTORY_SAMPLES:
        history[:-count] = samples[0]
    return history, mask


def generate(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).expanduser().resolve()
    force_sidecar_path = Path(args.force_sidecar).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in (
        (dataset_path, "dataset"),
        (force_sidecar_path, "force sidecar"),
        (checkpoint_path, "observer checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; choose a new path"
        )
    if output_path in (dataset_path, force_sidecar_path, checkpoint_path):
        raise ValueError("output must differ from every input")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, phase_names, checkpoint = load_frozen_context_observer(
        checkpoint_path, device
    )
    del checkpoint
    if tuple(phase_names) != VALVE_CONTEXT_V2_PHASE_NAMES:
        raise ValueError("observer and policy context phase orders differ")

    base_store, base_root, _ = open_nested_zip_group(dataset_path)
    force_store, force_root, _ = open_nested_zip_group(force_sidecar_path)
    try:
        base_data = base_root["data"]
        base_meta = base_root["meta"]
        force_data = force_root["data"]
        force_meta = force_root["meta"]
        if str(force_root.attrs.get("schema", "")) != "umi_force_sidecar_v1":
            raise ValueError("force sidecar must use umi_force_sidecar_v1")
        if str(force_root.attrs.get("train_wrench_key", "")) != "wrench_12d":
            raise ValueError("observer requires bias-removed data/wrench_12d")

        rgb = base_data["camera0_rgb"]
        position = np.asarray(base_data["robot0_eef_pos"][:], dtype=np.float32)
        rotation = np.asarray(
            base_data["robot0_eef_rot_axis_angle"][:], dtype=np.float32
        )
        gripper = np.asarray(
            base_data["robot0_gripper_width"][:], dtype=np.float32
        )
        rgb_episode_ends = np.asarray(
            base_meta["episode_ends"][:], dtype=np.int64
        ).reshape(-1)
        force_rgb_episode_ends = np.asarray(
            force_meta["rgb_episode_ends"][:], dtype=np.int64
        ).reshape(-1)
        if not np.array_equal(rgb_episode_ends, force_rgb_episode_ends):
            raise ValueError("base dataset and force sidecar episode ends differ")
        n_rgb = int(rgb_episode_ends[-1])
        if (
            tuple(rgb.shape) != (n_rgb, 224, 224, 3)
            or position.shape != (n_rgb, 3)
            or rotation.shape != (n_rgb, 3)
            or gripper.shape != (n_rgb, 1)
        ):
            raise ValueError("base observation arrays do not match observer contract")

        rgb_timestamp = np.asarray(
            force_data["rgb_timestamp_s"][:], dtype=np.float64
        ).reshape(-1)
        wrench_timestamp = np.asarray(
            force_data["wrench_timestamp_s"][:], dtype=np.float64
        ).reshape(-1)
        wrench = np.asarray(force_data["wrench_12d"][:], dtype=np.float32)
        rgb_to_wrench = np.asarray(
            force_data["rgb_to_wrench_end_idx"][:], dtype=np.int64
        ).reshape(-1)
        wrench_episode_ends = np.asarray(
            force_meta["wrench_episode_ends"][:], dtype=np.int64
        ).reshape(-1)
        if rgb_timestamp.shape != (n_rgb,) or rgb_to_wrench.shape != (n_rgb,):
            raise ValueError("force sidecar RGB mapping length mismatch")
        if wrench.shape != (len(wrench_timestamp), 12):
            raise ValueError("force sidecar wrench/timestamp shape mismatch")
        valid_map = rgb_to_wrench >= 0
        if valid_map.any() and np.any(
            wrench_timestamp[rgb_to_wrench[valid_map]]
            > rgb_timestamp[valid_map] + 1e-6
        ):
            raise ValueError("force sidecar contains future wrench mappings")

        pose_mats = pose_to_mat(np.concatenate([position, rotation], axis=-1))
        rgb_episode_starts = _starts(rgb_episode_ends)
        wrench_episode_starts = _starts(wrench_episode_ends)
        context = np.empty((n_rgb, VALVE_CONTEXT_V2_DIM), dtype=np.float32)

        with torch.inference_mode():
            for batch_start in tqdm(
                range(0, n_rgb, int(args.batch_size)),
                desc="four-state context",
            ):
                current = np.arange(
                    batch_start,
                    min(batch_start + int(args.batch_size), n_rgb),
                    dtype=np.int64,
                )
                episodes = _episode_for_indices(current, rgb_episode_ends)
                old = np.maximum(
                    current - OBSERVER_RGB_STRIDE,
                    rgb_episode_starts[episodes],
                )
                pair_idx = np.stack([old, current], axis=1)
                pair_pose = pose_mats[pair_idx]
                relative_pose = np.linalg.inv(pair_pose[:, -1])[:, None] @ pair_pose
                pose10d = mat_to_pose10d(relative_pose).astype(np.float32)
                pair_gripper = gripper[pair_idx].astype(np.float32)
                pair_rgb = np.stack(
                    [
                        np.stack([rgb[int(old_idx)], rgb[int(now_idx)]])
                        for old_idx, now_idx in pair_idx
                    ]
                )

                batch_history = []
                batch_mask = []
                for indices in pair_idx:
                    histories, masks = zip(
                        *(
                            _force_history(
                                int(rgb_idx),
                                rgb_episode_ends=rgb_episode_ends,
                                wrench_episode_starts=wrench_episode_starts,
                                wrench_episode_ends=wrench_episode_ends,
                                rgb_to_wrench_end_idx=rgb_to_wrench,
                                wrench=wrench,
                            )
                            for rgb_idx in indices
                        )
                    )
                    batch_history.append(np.stack(histories))
                    batch_mask.append(np.stack(masks))
                batch_history = np.stack(batch_history).astype(np.float32)
                batch_mask = np.stack(batch_mask).astype(np.float32)
                model_obs = {
                    "camera0_rgb": torch.from_numpy(
                        np.moveaxis(pair_rgb, -1, 2).astype(np.float32) / 255.0
                    ).to(device),
                    "robot0_eef_pos": torch.from_numpy(pose10d[..., :3]).to(device),
                    "robot0_eef_rot_axis_angle": torch.from_numpy(
                        pose10d[..., 3:]
                    ).to(device),
                    "robot0_gripper_width": torch.from_numpy(pair_gripper).to(device),
                    "robot0_ft_history": torch.from_numpy(batch_history).to(device),
                    "robot0_ft_history_valid": torch.from_numpy(batch_mask).to(device),
                }
                probability = (
                    model.predict_context(model_obs)["context_prob"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                ready = (
                    (old != current)
                    & np.all(batch_mask == 1.0, axis=(1, 2, 3))
                ).astype(np.float32)
                values = np.concatenate([probability, ready[:, None]], axis=-1)
                for row in values:
                    validate_valve_context_v2(row)
                context[current] = values
    finally:
        base_store.close()
        force_store.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_store = zarr.DirectoryStore(str(output_path))
    try:
        root = zarr.group(store=output_store, overwrite=False)
        root.attrs.update(
            {
                "schema": VALVE_CONTEXT_V2_SCHEMA,
                "context_key": "valve_context_5d",
                "timestamp_key": "valve_context_timestamp_s",
                "episode_ends_key": "rgb_episode_ends",
                "context_dim": VALVE_CONTEXT_V2_DIM,
                "phase_names": list(VALVE_CONTEXT_V2_PHASE_NAMES),
                "context_valid_semantics": (
                    "two distinct stride-3 RGB frames and 50 real causal F/T "
                    "samples for each frame"
                ),
                "observer_checkpoint_sha256": sha256_file(checkpoint_path),
                "base_episode_ends_sha256": hashlib.sha256(
                    rgb_episode_ends.tobytes()
                ).hexdigest(),
                "force_coordinate_transform": "none",
                "force_units": "N,N,N,Nm,Nm,Nm per finger",
            }
        )
        data = root.create_group("data")
        meta = root.create_group("meta")
        data.create_dataset(
            "valve_context_5d",
            data=context,
            chunks=(min(4096, len(context)), VALVE_CONTEXT_V2_DIM),
        )
        data.create_dataset(
            "valve_context_timestamp_s",
            data=rgb_timestamp,
            chunks=(min(4096, len(rgb_timestamp)),),
        )
        meta.create_dataset("rgb_episode_ends", data=rgb_episode_ends)
    finally:
        output_store.close()
    print(f"wrote {len(context)} contexts to {output_path}")
    print(f"observer_sha256={sha256_file(checkpoint_path)}")
    print(f"valid_ratio={float(np.mean(context[:, -1])):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--force-sidecar", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
