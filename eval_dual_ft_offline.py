#!/usr/bin/env python3
"""Offline validation for the 786-D-condition, 11-D-action dual-F/T policy.

This evaluator never opens robot, camera, or Modbus interfaces. It restores a
training checkpoint, uses the checkpoint's own normalizer, samples the
checkpoint-defined train/validation dataset, and reports denoising loss plus
physical-unit action errors. The stochastic diffusion path is made
reproducible by deriving every batch/repeat seed from ``--seed``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.dual_ft_contract import (
    inspect_dual_ft_checkpoint_payload,
)
from diffusion_policy.common.nested_zarr import open_nested_zip_group
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from umi.common.pose_util import rot6d_to_mat


ROOT_DIR = Path(__file__).resolve().parent
EXPECTED_CONDITION_DIM = 786
EXPECTED_ACTION_HORIZON = 16
EXPECTED_ACTION_DIM = 11
IMPLEMENTATION_FILES = (
    "eval_dual_ft_offline.py",
    "diffusion_policy/common/dual_ft_contract.py",
    "diffusion_policy/config/task/umi_dual_ft.yaml",
    "diffusion_policy/config/task/umi_dual_ft_260827_bias_only.yaml",
    "diffusion_policy/config/train_diffusion_unet_timm_umi_dual_ft_workspace.yaml",
    "diffusion_policy/dataset/umi_dual_ft_dataset.py",
    "diffusion_policy/model/vision/dual_ft_obs_encoder.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy.py",
)


@dataclass
class LoadedPolicy:
    cfg: Any
    policy: torch.nn.Module
    state_name: str
    device: torch.device
    disabled_image_transforms: list[str]


class ErrorAccumulator:
    """Streaming absolute/squared error statistics without retaining tensors."""

    def __init__(self) -> None:
        self.count = 0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.max_abs = 0.0

    def update(self, error) -> None:
        values = np.asarray(error, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("metric input contains NaN or Inf")
        absolute = np.abs(values)
        self.count += int(values.size)
        self.abs_sum += float(np.sum(absolute, dtype=np.float64))
        self.square_sum += float(np.sum(np.square(values), dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(absolute)))

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mae": None,
                "mse": None,
                "rmse": None,
                "max_abs": None,
            }
        mse = self.square_sum / self.count
        return {
            "count": self.count,
            "mae": self.abs_sum / self.count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "max_abs": self.max_abs,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_sha256(path: Path) -> tuple[str, dict[str, int]]:
    before = _file_identity(path)
    digest = _sha256(path)
    after = _file_identity(path)
    if after != before:
        raise RuntimeError(
            f"checkpoint changed while hashing: {path}. Evaluate an immutable copy."
        )
    return digest, after


def _require_file_identity(path: Path, expected: dict[str, int]) -> None:
    actual = _file_identity(path)
    if actual != expected:
        raise RuntimeError(
            f"checkpoint changed during evaluation: {path}. "
            "Evaluate a copied/immutable checkpoint instead of a live latest.ckpt."
        )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _implementation_sha256() -> dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_FILES:
        path = ROOT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"evaluation implementation file is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _resolve_checkpoint(path: str | os.PathLike) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.suffix != ".ckpt":
        checkpoint = checkpoint / "checkpoints" / "latest.ckpt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def _resolve_dataset(
    requested: str | os.PathLike | None,
    embedded: str,
    checkpoint: Path,
    *,
    option_name: str = "--dataset",
) -> Path:
    raw = Path(requested if requested is not None else embedded).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        ROOT_DIR / raw,
        checkpoint.parent / raw,
        checkpoint.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    source = option_name if requested is not None else "checkpoint cfg"
    raise FileNotFoundError(
        f"dataset from {source} does not exist: {raw}. "
        f"Pass the transferred zarr path with {option_name}."
    )


def _load_payload(checkpoint: Path) -> dict:
    # dill checkpoints can execute Python while loading. This CLI is for
    # checkpoints produced locally by this repository, never untrusted files.
    with checkpoint.open("rb") as stream:
        payload = torch.load(
            stream,
            map_location="cpu",
            pickle_module=dill,
        )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def _select_state_name(payload: dict, requested: str, use_ema: bool) -> str:
    states = payload.get("state_dicts", {})
    if requested == "auto":
        # A missing EMA in an EMA-configured run is a corrupt checkpoint, not
        # a reason to silently evaluate a different family of weights.
        candidates = ("ema_model",) if use_ema else ("model",)
    else:
        candidates = (requested,)
    for name in candidates:
        if isinstance(states.get(name), dict) and states[name]:
            return name
    raise ValueError(
        f"checkpoint has no usable {requested!r} policy state; "
        f"available states: {sorted(states.keys())}"
    )


def disable_stochastic_image_transforms(policy) -> list[str]:
    """Disable augmentation in both legacy and nested dual-F/T encoders."""

    root = getattr(policy, "obs_encoder", None)
    if root is None:
        return []
    candidates = [("obs_encoder", root)]
    nested = getattr(root, "vision_pose_encoder", None)
    if nested is not None:
        candidates.append(("obs_encoder.vision_pose_encoder", nested))

    disabled: list[str] = []
    seen_maps: set[int] = set()
    for prefix, encoder in candidates:
        transform_map = getattr(encoder, "key_transform_map", None)
        if transform_map is None or id(transform_map) in seen_maps:
            continue
        seen_maps.add(id(transform_map))
        for key in list(transform_map.keys()):
            transform_map[key] = torch.nn.Identity()
            disabled.append(f"{prefix}.{key}")
    return disabled


def _normalizer_summary(policy) -> dict[str, Any]:
    params = policy.normalizer.params_dict
    expected_dimensions = {
        "camera0_rgb": 1,
        "robot0_eef_pos": 3,
        "robot0_eef_rot_axis_angle": 6,
        "robot0_ft_left": 6,
        "robot0_ft_right": 6,
        "action": EXPECTED_ACTION_DIM,
    }
    missing = [key for key in expected_dimensions if key not in params]
    if missing:
        raise ValueError(f"checkpoint normalizer is missing keys: {missing}")
    unexpected = sorted(set(params.keys()) - set(expected_dimensions))
    if unexpected:
        raise ValueError(f"checkpoint normalizer has unexpected keys: {unexpected}")

    fields: dict[str, Any] = {}
    for key, dimension in expected_dimensions.items():
        group = params[key]
        for required_name in ("scale", "offset", "input_stats"):
            if required_name not in group:
                raise ValueError(
                    f"checkpoint normalizer[{key!r}] is missing {required_name!r}"
                )
        scale = group["scale"].detach().cpu().numpy()
        offset = group["offset"].detach().cpu().numpy()
        expected_shape = (dimension,)
        if scale.shape != expected_shape or offset.shape != expected_shape:
            raise ValueError(
                f"checkpoint normalizer[{key!r}] scale/offset must be "
                f"{expected_shape}, got {scale.shape}/{offset.shape}"
            )
        if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(offset)):
            raise ValueError(
                f"checkpoint normalizer[{key!r}] scale/offset contains NaN or Inf"
            )
        if np.any(scale == 0):
            raise ValueError(f"checkpoint normalizer[{key!r}] has zero scale")

        stats = group["input_stats"]
        if set(stats.keys()) != {"min", "max", "mean", "std"}:
            raise ValueError(
                f"checkpoint normalizer[{key!r}] input_stats keys must be "
                "min/max/mean/std"
            )
        field_stats = {}
        for name in ("min", "max", "mean", "std"):
            value = stats[name].detach().cpu().numpy()
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"checkpoint normalizer[{key!r}] {name} must be finite "
                    f"{expected_shape}, got {value.shape}"
                )
            field_stats[name] = value.tolist()
        if np.any(np.asarray(field_stats["min"]) > np.asarray(field_stats["max"])):
            raise ValueError(f"checkpoint normalizer[{key!r}] has min > max")
        if np.any(np.asarray(field_stats["std"]) < 0):
            raise ValueError(f"checkpoint normalizer[{key!r}] has negative std")
        fields[key] = {
            "scale": scale.tolist(),
            "offset": offset.tolist(),
            "input_stats": field_stats,
        }

    return {
        "keys": list(params.keys()),
        "fields": fields,
        "grasp_force": {
            name: float(values[10])
            for name, values in fields["action"]["input_stats"].items()
        },
    }


def load_policy(
    payload: dict,
    *,
    dataset_path: Path,
    force_sidecar_path: Path,
    device_spec: str,
    weights: str,
    num_inference_steps: int | None,
) -> LoadedPolicy:
    contract = inspect_dual_ft_checkpoint_payload(payload)
    cfg = copy.deepcopy(contract["cfg"])
    expected_targets = {
        "policy._target_": (
            "diffusion_policy.policy.diffusion_unet_timm_policy."
            "DiffusionUnetTimmPolicy"
        ),
        "policy.obs_encoder._target_": (
            "diffusion_policy.model.vision.dual_ft_obs_encoder."
            "DualFTObsEncoder"
        ),
        "policy.noise_scheduler._target_": "diffusers.DDIMScheduler",
        "task.dataset._target_": (
            "diffusion_policy.dataset.umi_dual_ft_dataset.UmiDualFTDataset"
        ),
    }
    for path, expected in expected_targets.items():
        actual = str(OmegaConf.select(cfg, path, default=""))
        if actual != expected:
            raise ValueError(
                f"checkpoint target {path} must be {expected!r}, got {actual!r}"
            )
    cfg.task.dataset_path = str(dataset_path)
    cfg.task.dataset.dataset_path = str(dataset_path)
    cfg.task.force_sidecar_path = str(force_sidecar_path)
    cfg.task.dataset.force_sidecar_path = str(force_sidecar_path)
    if OmegaConf.select(cfg, "policy.obs_encoder.pretrained", default=None) is not None:
        cfg.policy.obs_encoder.pretrained = False
    if OmegaConf.select(cfg, "policy.obs_encoder.transforms", default=None) is not None:
        cfg.policy.obs_encoder.transforms = None

    state_name = _select_state_name(
        payload,
        requested=weights,
        use_ema=bool(OmegaConf.select(cfg, "training.use_ema", default=False)),
    )
    selected_state = payload["state_dicts"][state_name]
    # The payload can also contain the other policy, EMA, and optimizer states.
    # Keep only the selected mapping before allocating the single evaluation
    # policy so CPU memory does not peak at workspace-training levels.
    payload.clear()

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(selected_state, strict=True)
    del selected_state
    if num_inference_steps is not None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        policy.num_inference_steps = int(num_inference_steps)

    disabled = disable_stochastic_image_transforms(policy)
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)
    policy.eval().to(device)

    if int(policy.obs_feature_dim) != EXPECTED_CONDITION_DIM:
        raise ValueError(
            f"restored policy condition must be 786-D, got {policy.obs_feature_dim}"
        )
    if (int(policy.action_horizon), int(policy.action_dim)) != (
        EXPECTED_ACTION_HORIZON,
        EXPECTED_ACTION_DIM,
    ):
        raise ValueError(
            "restored policy action must be [16,11], got "
            f"[{policy.action_horizon},{policy.action_dim}]"
        )
    marker_values = {
        "architecture": int(policy.obs_encoder.architecture_contract_version),
        "left_temporal": int(
            policy.obs_encoder.left_ft_encoder.temporal_contract_version
        ),
        "right_temporal": int(
            policy.obs_encoder.right_ft_encoder.temporal_contract_version
        ),
    }
    if marker_values != {
        "architecture": 2,
        "left_temporal": 1,
        "right_temporal": 1,
    }:
        raise ValueError(f"restored policy contract buffers mismatch: {marker_values}")
    _normalizer_summary(policy)
    return LoadedPolicy(
        cfg=cfg,
        policy=policy,
        state_name=state_name,
        device=device,
        disabled_image_transforms=disabled,
    )


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        bad = torch.nonzero(~torch.isfinite(value), as_tuple=False)
        raise ValueError(
            f"{name} contains NaN or Inf; first indices={bad[:10].cpu().tolist()}"
        )


def _rotation_error_rad(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    # umi.common.pose_util.normalize uses a 2-D transpose idiom, so flatten
    # batch/time explicitly before converting rotation-6D to matrices.
    pred_6d = np.asarray(prediction, dtype=np.float64).reshape(-1, 6)
    target_6d = np.asarray(target, dtype=np.float64).reshape(-1, 6)
    pred_matrix = rot6d_to_mat(pred_6d)
    target_matrix = rot6d_to_mat(target_6d)
    pred_rotation = st.Rotation.from_matrix(pred_matrix)
    target_rotation = st.Rotation.from_matrix(target_matrix)
    return (pred_rotation * target_rotation.inv()).magnitude()


def _update_action_metrics(
    accumulators: dict[str, ErrorAccumulator],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape or prediction.shape[-1] != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"action prediction/target shape mismatch: {prediction.shape} vs "
            f"{target.shape}"
        )
    difference = (prediction - target).detach().cpu().numpy()
    accumulators["overall"].update(difference)
    accumulators["position_m"].update(difference[..., :3])
    accumulators["rotation_6d"].update(difference[..., 3:9])
    accumulators["gripper_width_m"].update(difference[..., 9:10])
    accumulators["grasp_force_N"].update(difference[..., 10:11])
    rotation_error_rad = _rotation_error_rad(
        prediction.detach().cpu().numpy()[..., 3:9],
        target.detach().cpu().numpy()[..., 3:9],
    )
    accumulators["rotation_geodesic_rad"].update(rotation_error_rad)
    accumulators["rotation_geodesic_deg"].update(
        np.rad2deg(rotation_error_rad)
    )


def _percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    if not np.all(np.isfinite(array)):
        raise ValueError("timing/age values contain NaN or Inf")
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _numeric_summary(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    if not np.all(np.isfinite(array)):
        raise ValueError("provenance array contains NaN or Inf")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _update_array_digest(digest, name: str, values) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def _effective_non_rgb_sha256(dataset) -> str:
    digest = hashlib.sha256()
    names = (
        "rgb_timestamps",
        "robot_timestamps",
        "gripper_timestamps",
        "eef_pos",
        "eef_rot_axis_angle",
        "gripper_width",
        "grasp_force",
        "ft_left",
        "ft_right",
        "ft_left_timestamps",
        "ft_right_timestamps",
        "rgb_episode_ends",
        "robot_episode_ends",
        "gripper_episode_ends",
        "ft_left_episode_ends",
        "ft_right_episode_ends",
    )
    for name in names:
        _update_array_digest(digest, name, getattr(dataset, name))
    bias = getattr(dataset, "ft_episode_bias_12d", None)
    if bias is not None:
        _update_array_digest(digest, "ft_episode_bias_12d", bias)
    return digest.hexdigest()


def _filesystem_manifest(path: Path) -> dict[str, Any]:
    portable_digest = hashlib.sha256()
    runtime_digest = hashlib.sha256()
    if path.is_file():
        stat = path.stat()
        entries = [(path.name, stat.st_size, stat.st_mtime_ns)]
    else:
        entries = []
        for item in sorted(path.rglob("*")):
            if not item.is_file():
                continue
            stat = item.stat()
            entries.append(
                (item.relative_to(path).as_posix(), stat.st_size, stat.st_mtime_ns)
            )
    total_bytes = 0
    for relative, size, mtime_ns in entries:
        total_bytes += int(size)
        portable_row = f"{relative}\0{int(size)}\n".encode("utf-8")
        portable_digest.update(portable_row)
        runtime_digest.update(portable_row)
        runtime_digest.update(f"{int(mtime_ns)}\n".encode("ascii"))
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "path_and_size_manifest_sha256": portable_digest.hexdigest(),
        "path_size_mtime_runtime_sha256": runtime_digest.hexdigest(),
        "content_hash_coverage": "names_and_sizes_only",
    }


def _sha256_dataset_content(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _split_causal_age_summary(dataset, indices) -> dict[str, Any]:
    left_starts = np.r_[0, dataset.ft_left_episode_ends[:-1]]
    right_starts = np.r_[0, dataset.ft_right_episode_ends[:-1]]
    left_ages = []
    right_ages = []
    for episode, current in indices:
        anchor = float(dataset.rgb_timestamps[current])
        ls, le = int(left_starts[episode]), int(dataset.ft_left_episode_ends[episode])
        rs, re = int(right_starts[episode]), int(dataset.ft_right_episode_ends[episode])
        left_t = dataset.ft_left_timestamps[ls:le]
        right_t = dataset.ft_right_timestamps[rs:re]
        left_idx = int(np.searchsorted(left_t, anchor, side="right") - 1)
        right_idx = int(np.searchsorted(right_t, anchor, side="right") - 1)
        if left_idx < 0 or right_idx < 0:
            raise ValueError("split contains a non-causal F/T anchor")
        left_ages.append((anchor - float(left_t[left_idx])) * 1000.0)
        right_ages.append((anchor - float(right_t[right_idx])) * 1000.0)
    return {
        "left_latest_age_ms": _numeric_summary(left_ages),
        "right_latest_age_ms": _numeric_summary(right_ages),
    }


def _zarr_provenance(dataset_path: Path, bias_key: str | None) -> dict[str, Any]:
    store, root, prefix = open_nested_zip_group(dataset_path)
    try:
        data = root["data"]
        meta = root["meta"]
        array_schema = {}
        for group_name, group in (("data", data), ("meta", meta)):
            for key in sorted(group.array_keys()):
                array = group[key]
                array_schema[f"{group_name}/{key}"] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "chunks": list(array.chunks),
                }
        sync_offsets = (
            np.asarray(meta["sync_offset_s"][:], dtype=np.float64)
            if "sync_offset_s" in meta
            else np.asarray([], dtype=np.float64)
        )
        sync_verdicts = (
            [str(value) for value in np.asarray(meta["sync_verdict"][:]).tolist()]
            if "sync_verdict" in meta
            else []
        )
        source_episodes = (
            [str(value) for value in np.asarray(meta["source_episode"][:]).tolist()]
            if "source_episode" in meta
            else []
        )
        source_digest = hashlib.sha256()
        for value in source_episodes:
            source_digest.update(value.encode("utf-8"))
            source_digest.update(b"\n")
        bias = (
            np.asarray(meta[bias_key][:], dtype=np.float32)
            if bias_key and bias_key in meta
            else None
        )
        bias_digest = None
        if bias is not None:
            digest = hashlib.sha256()
            _update_array_digest(digest, bias_key, bias)
            bias_digest = digest.hexdigest()
        return {
            "nested_prefix": prefix,
            "stored_root_attrs": dict(root.attrs),
            "array_schema": array_schema,
            "sync": {
                "verdict_counts": dict(sorted(Counter(sync_verdicts).items())),
                "offset_seconds": _numeric_summary(sync_offsets),
                "note": (
                    "Offsets and verdicts are recorder provenance; causal age "
                    "after applying an offset does not prove physical sync accuracy."
                ),
            },
            "source_episode_count": len(source_episodes),
            "source_episode_list_sha256": source_digest.hexdigest(),
            "episode_bias": {
                "key": bias_key,
                "shape": None if bias is None else list(bias.shape),
                "sha256": bias_digest,
                "all_channels": _numeric_summary([] if bias is None else bias),
                "per_channel_mean": (
                    None if bias is None else np.mean(bias, axis=0).tolist()
                ),
                "per_channel_std": (
                    None if bias is None else np.std(bias, axis=0).tolist()
                ),
            },
        }
    finally:
        store.close()


def _dataset_provenance(
    dataset,
    split_dataset,
    cfg,
    dataset_path: Path,
    force_sidecar_path: Path,
    *,
    full_content_hash: bool,
) -> dict[str, Any]:
    ft_cfg = OmegaConf.to_container(cfg.task.dataset.ft, resolve=True)
    model_contract = OmegaConf.to_container(cfg.task.model_contract, resolve=True)
    if str(ft_cfg.get("wrench_key")) != "wrench_12d":
        raise ValueError("evaluation requires native sidecar wrench_key='wrench_12d'")
    if str(ft_cfg.get("bias_removal")) != "precomputed_in_sidecar":
        raise ValueError("evaluation requires precomputed_in_sidecar F/T bias removal")
    if str(ft_cfg.get("bias_key")) != "wrench_episode_bias_12d":
        raise ValueError(
            "evaluation requires F/T bias_key='wrench_episode_bias_12d'"
        )
    if not bool(getattr(dataset, "ft_bias_removed", False)):
        raise ValueError("dataset did not apply the required episode F/T bias")
    if getattr(dataset, "ft_episode_bias_12d", None) is None:
        raise ValueError("dataset has no 12-D episode bias metadata")

    dropped = list(getattr(dataset, "causal_drop_report", []))
    split_episode_indices = sorted({int(ep) for ep, _ in split_dataset.indices})
    force = np.asarray(dataset.grasp_force, dtype=np.float64).reshape(-1)
    provenance = {
        "effective_non_rgb_sha256": _effective_non_rgb_sha256(dataset),
        "base_dataset": {
            "path": str(dataset_path),
            "filesystem_manifest": _filesystem_manifest(dataset_path),
            "full_content_sha256": (
                _sha256_dataset_content(dataset_path) if full_content_hash else None
            ),
        },
        "force_sidecar": {
            "path": str(force_sidecar_path),
            "filesystem_manifest": _filesystem_manifest(force_sidecar_path),
            "full_content_sha256": (
                _sha256_dataset_content(force_sidecar_path)
                if full_content_hash else None
            ),
        },
        "full_content_hash_requested": bool(full_content_hash),
        "dataset_class": type(dataset).__name__,
        "episodes": int(dataset.n_episodes),
        "split_episode_count": len(split_episode_indices),
        "split_episode_indices": split_episode_indices,
        "raw_rgb_hz": float(dataset.raw_rgb_hz),
        "ft_left_hz": float(dataset.ft_left_hz),
        "ft_right_hz": float(dataset.ft_right_hz),
        "action_hz": float(dataset.action_frequency),
        "action_horizon": int(dataset.action_horizon),
        "action_future_span_seconds": float(
            (dataset.action_horizon - 1) / dataset.action_frequency
        ),
        "ft_history": {
            "samples": int(dataset.ft_num_steps),
            "stride": int(dataset.ft_stride),
            "configured_span_seconds": float(dataset.ft_history_seconds),
            "dataset_padding": str(ft_cfg["padding"]),
            "encoder_padding_contract": str(
                model_contract["ft_temporal_contract"]
            ),
        },
        "causal_drop_all_episodes": {
            "anchors": int(sum(row["anchors"] for row in dropped)),
            "dropped": int(sum(row["dropped"] for row in dropped)),
            "episodes_with_drops": int(sum(row["dropped"] > 0 for row in dropped)),
        },
        "causal_age_full_split": _split_causal_age_summary(
            dataset, split_dataset.indices
        ),
        "effective_ft": {
            "source_key": ft_cfg["wrench_key"],
            "left_channels": [0, 1, 2, 3, 4, 5],
            "right_channels": [6, 7, 8, 9, 10, 11],
            "frames": [ft_cfg["left_frame"], ft_cfg["right_frame"]],
            "channel_order": list(ft_cfg["channel_order"]),
            "force_unit": ft_cfg["force_unit"],
            "torque_unit": ft_cfg["torque_unit"],
            "axis_permutation": list(ft_cfg["axis_permutation"]),
            "axis_sign": list(ft_cfg["axis_sign"]),
            "coordinate_transform_applied": False,
            "bias_removed": True,
            "bias_key": ft_cfg["bias_key"],
            "bias_removal": ft_cfg["bias_removal"],
            "deployment_bias_requirement": ft_cfg[
                "deployment_bias_requirement"
            ],
        },
        "grasp_force_target": {
            "key": model_contract["grasp_force_source"],
            "unit": "N",
            "formula": model_contract["grasp_force_formula"],
            "alignment": model_contract["grasp_force_alignment"],
            "role": model_contract["grasp_force_role"],
            "validation_max_abs_error": float(
                dataset.grasp_force_contract_max_abs_error
            ),
            "distribution": _numeric_summary(force),
            "negative_fraction": float(np.mean(force < 0)),
        },
        "action_semantics": {
            "pose": model_contract["action_pose_semantics"],
            "gripper_width": model_contract["gripper_width_semantics"],
            "grasp_force": model_contract["grasp_force_role"],
        },
        "model_contract": model_contract,
    }
    provenance["base_zarr"] = _zarr_provenance(dataset_path, None)
    provenance["force_sidecar_zarr"] = _zarr_provenance(
        force_sidecar_path,
        str(ft_cfg["bias_key"]),
    )
    return provenance


@torch.inference_mode()
def evaluate_loader(
    policy,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    prediction_repeats: int,
    seed: int,
    compute_diffusion_loss: bool,
    expected_obs_meta=None,
    ft_max_age_sec: float | None = None,
) -> dict[str, Any]:
    if prediction_repeats <= 0:
        raise ValueError("prediction_repeats must be positive")
    accumulators = {
        name: ErrorAccumulator()
        for name in (
            "overall",
            "position_m",
            "rotation_6d",
            "rotation_geodesic_rad",
            "rotation_geodesic_deg",
            "gripper_width_m",
            "grasp_force_N",
            "normalized_action",
        )
    }
    loss_sum = 0.0
    loss_samples = 0
    evaluated_samples = 0
    evaluated_batches = 0
    inference_seconds: list[float] = []
    left_age_ms: list[float] = []
    right_age_ms: list[float] = []
    action_start_offset_ms: list[float] = []
    action_end_offset_ms: list[float] = []
    episode_indices: set[int] = set()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        obs = dict_apply(batch["obs"], lambda value: value.to(device, non_blocking=True))
        target = batch["action"].to(device, non_blocking=True)
        if expected_obs_meta is not None:
            expected_keys = set(expected_obs_meta.keys())
            if set(obs.keys()) != expected_keys:
                raise ValueError(
                    "dataset observation keys mismatch: "
                    f"expected={sorted(expected_keys)} got={sorted(obs.keys())}"
                )
            for key, meta in expected_obs_meta.items():
                expected_shape = (
                    int(meta["horizon"]),
                    *tuple(int(value) for value in meta["shape"]),
                )
                if tuple(obs[key].shape[1:]) != expected_shape:
                    raise ValueError(
                        f"obs[{key!r}] must end in {expected_shape}, "
                        f"got {tuple(obs[key].shape)}"
                    )
        if target.ndim != 3 or tuple(target.shape[1:]) != (
            EXPECTED_ACTION_HORIZON,
            EXPECTED_ACTION_DIM,
        ):
            raise ValueError(f"dataset action must be [B,16,11], got {target.shape}")
        for key, value in obs.items():
            _assert_finite_tensor(f"obs[{key!r}]", value)
        _assert_finite_tensor("target action", target)

        batch_size = int(target.shape[0])
        if compute_diffusion_loss:
            torch.manual_seed(seed + batch_index * 1009)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + batch_index * 1009)
            loss = policy.compute_loss({"obs": obs, "action": target})
            _assert_finite_tensor("diffusion loss", loss)
            loss_sum += float(loss.detach().cpu()) * batch_size
            loss_samples += batch_size

        for repeat in range(prediction_repeats):
            repeat_seed = seed + batch_index * 1009 + repeat * 1_000_003
            torch.manual_seed(repeat_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(repeat_seed)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = policy.predict_action(obs)["action_pred"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds.append(time.perf_counter() - started)
            _assert_finite_tensor("predicted action", prediction)
            _update_action_metrics(accumulators, prediction, target)
            normalized_prediction = policy.normalizer["action"].normalize(prediction)
            normalized_target = policy.normalizer["action"].normalize(target)
            accumulators["normalized_action"].update(
                (normalized_prediction - normalized_target).detach().cpu().numpy()
            )

        sample_info = batch.get("sample_info", {})
        if expected_obs_meta is not None:
            required_timing_keys = {
                "episode_index",
                "anchor_timestamp",
                "rgb_timestamps",
                "pose_timestamps",
                "left_ft_timestamps",
                "right_ft_timestamps",
                "action_timestamps",
                "left_ft_age",
                "right_ft_age",
            }
            missing_timing = sorted(required_timing_keys - set(sample_info.keys()))
            if missing_timing:
                raise ValueError(
                    f"dataset sample_info is missing timing keys: {missing_timing}"
                )
            for key in required_timing_keys - {"episode_index"}:
                _assert_finite_tensor(f"sample_info[{key!r}]", sample_info[key])

            anchor = sample_info["anchor_timestamp"].reshape(-1, 1)
            for key in (
                "rgb_timestamps",
                "pose_timestamps",
                "left_ft_timestamps",
                "right_ft_timestamps",
            ):
                timestamps = sample_info[key]
                if timestamps.ndim != 2 or timestamps.shape[0] != batch_size:
                    raise ValueError(
                        f"sample_info[{key!r}] must be [B,T], got "
                        f"{tuple(timestamps.shape)}"
                    )
                if bool((timestamps > anchor + 1e-9).any()):
                    raise ValueError(f"future observation timestamp selected in {key}")
                if bool((torch.diff(timestamps, dim=1) < 0).any()):
                    raise ValueError(f"non-monotonic observation timestamps in {key}")

            action_timestamps = sample_info["action_timestamps"]
            if tuple(action_timestamps.shape) != (batch_size, EXPECTED_ACTION_HORIZON):
                raise ValueError(
                    "sample_info['action_timestamps'] must be [B,16], got "
                    f"{tuple(action_timestamps.shape)}"
                )
            if bool((torch.diff(action_timestamps, dim=1) <= 0).any()):
                raise ValueError("action timestamps must be strictly increasing")
            start_offset = action_timestamps[:, :1] - anchor
            if bool((torch.abs(start_offset) > 1e-6).any()):
                raise ValueError("first action timestamp must equal the policy anchor")
            action_start_offset_ms.extend(
                (start_offset[:, 0].detach().cpu().numpy() * 1000.0).tolist()
            )
            action_end_offset_ms.extend(
                (
                    (action_timestamps[:, -1:] - anchor)[:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    * 1000.0
                ).tolist()
            )
        if "left_ft_age" in sample_info:
            left_age_ms.extend(
                (sample_info["left_ft_age"].detach().cpu().numpy() * 1000.0).tolist()
            )
        if "right_ft_age" in sample_info:
            right_age_ms.extend(
                (sample_info["right_ft_age"].detach().cpu().numpy() * 1000.0).tolist()
            )
        if ft_max_age_sec is not None:
            for side in ("left", "right"):
                age_key = f"{side}_ft_age"
                if age_key in sample_info:
                    ages = sample_info[age_key]
                    if bool((ages < -1e-9).any()):
                        raise ValueError(f"{side} F/T sample age is negative")
                    if bool((ages > float(ft_max_age_sec) + 1e-9).any()):
                        maximum = float(ages.max())
                        raise ValueError(
                            f"{side} F/T sample age {maximum:.9f}s exceeds "
                            f"configured {float(ft_max_age_sec):.9f}s"
                        )
                timestamps_key = f"{side}_ft_timestamps"
                if timestamps_key in sample_info and "anchor_timestamp" in sample_info:
                    timestamps = sample_info[timestamps_key]
                    anchor = sample_info["anchor_timestamp"].reshape(-1, 1)
                    if bool((timestamps > anchor + 1e-9).any()):
                        raise ValueError(f"future {side} F/T timestamp selected")
                    expected_age = anchor[:, 0] - timestamps[:, -1]
                    if not bool(
                        torch.allclose(
                            sample_info[age_key], expected_age, atol=1e-9, rtol=0
                        )
                    ):
                        raise ValueError(f"{side} F/T age/timestamp mismatch")
        if "episode_index" in sample_info:
            episode_indices.update(
                int(value)
                for value in sample_info["episode_index"].detach().cpu().reshape(-1)
            )
        evaluated_samples += batch_size
        evaluated_batches += 1

    if evaluated_batches == 0:
        raise ValueError("evaluation dataloader yielded zero batches")
    latency_ms = np.asarray(inference_seconds, dtype=np.float64) * 1000.0
    return {
        "evaluated_batches": evaluated_batches,
        "evaluated_samples": evaluated_samples,
        "prediction_repeats": prediction_repeats,
        "diffusion_loss": (
            loss_sum / loss_samples if compute_diffusion_loss and loss_samples else None
        ),
        "action_error": {
            name: accumulator.result() for name, accumulator in accumulators.items()
        },
        "inference_latency_ms_per_batch": _percentile_summary(latency_ms),
        "left_ft_latest_age_ms": _percentile_summary(left_age_ms),
        "right_ft_latest_age_ms": _percentile_summary(right_age_ms),
        "action_start_offset_ms": _percentile_summary(action_start_offset_ms),
        "action_end_offset_ms": _percentile_summary(action_end_offset_ms),
        "episode_indices": sorted(episode_indices),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    implementation_sha256 = _implementation_sha256()
    checkpoint_sha256, checkpoint_identity = _stable_sha256(checkpoint)
    payload = _load_payload(checkpoint)
    _require_file_identity(checkpoint, checkpoint_identity)
    if "cfg" not in payload:
        raise ValueError("checkpoint payload is missing cfg")
    embedded_dataset = str(
        OmegaConf.select(payload["cfg"], "task.dataset.dataset_path", default="")
    )
    dataset_path = _resolve_dataset(args.dataset, embedded_dataset, checkpoint)
    embedded_force_sidecar = str(
        OmegaConf.select(
            payload["cfg"],
            "task.dataset.force_sidecar_path",
            default="",
        )
    )
    force_sidecar_path = _resolve_dataset(
        args.force_sidecar,
        embedded_force_sidecar,
        checkpoint,
        option_name="--force-sidecar",
    )
    if dataset_path == force_sidecar_path:
        raise ValueError("base dataset and force sidecar paths must differ")
    output = Path(args.output).expanduser().resolve()
    if output == checkpoint:
        raise ValueError("--output must not overwrite the checkpoint")
    if output == dataset_path or (
        dataset_path.is_dir() and _is_within(output, dataset_path)
    ):
        raise ValueError("--output must not overwrite or modify the dataset")
    if output == force_sidecar_path or (
        force_sidecar_path.is_dir() and _is_within(output, force_sidecar_path)
    ):
        raise ValueError("--output must not overwrite or modify the force sidecar")
    if output.exists() and output.is_dir():
        raise ValueError(f"--output must be a file path, got directory: {output}")
    loaded = load_policy(
        payload,
        dataset_path=dataset_path,
        force_sidecar_path=force_sidecar_path,
        device_spec=args.device,
        weights=args.weights,
        num_inference_steps=args.num_inference_steps,
    )
    del payload

    dataset = hydra.utils.instantiate(loaded.cfg.task.dataset)
    if not isinstance(dataset, BaseImageDataset):
        raise TypeError(f"expected BaseImageDataset, got {type(dataset).__name__}")
    split_dataset = (
        dataset.get_validation_dataset() if args.split == "validation" else dataset
    )
    max_samples = int(args.max_samples)
    selected_indices: list[int] | None = None
    if max_samples > 0 and max_samples < len(split_dataset):
        generator = np.random.default_rng(int(args.seed))
        selected_indices = sorted(
            int(index)
            for index in generator.choice(
                len(split_dataset),
                size=max_samples,
                replace=False,
            )
        )
        selected_dataset = Subset(split_dataset, selected_indices)
    else:
        selected_dataset = split_dataset
    worker_count = int(args.num_workers)
    loader = DataLoader(
        selected_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=worker_count,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=worker_count > 0,
        multiprocessing_context="spawn" if worker_count > 0 else None,
    )
    try:
        dataset_provenance = _dataset_provenance(
            dataset,
            split_dataset,
            loaded.cfg,
            dataset_path,
            force_sidecar_path,
            full_content_hash=bool(args.full_dataset_hash),
        )
        max_batches = None if int(args.max_batches) == 0 else int(args.max_batches)
        metrics = evaluate_loader(
            loaded.policy,
            loader,
            device=loaded.device,
            max_batches=max_batches,
            prediction_repeats=int(args.prediction_repeats),
            seed=int(args.seed),
            compute_diffusion_loss=not args.skip_diffusion_loss,
            expected_obs_meta=loaded.cfg.task.shape_meta.obs,
            ft_max_age_sec=float(
                OmegaConf.select(loaded.cfg, "task.ft_max_age_sec", default=0.012)
            ),
        )
        _require_file_identity(checkpoint, checkpoint_identity)
        if _implementation_sha256() != implementation_sha256:
            raise RuntimeError(
                "evaluation implementation changed during the run; rerun with "
                "a stable checkout"
            )
        if _filesystem_manifest(dataset_path) != dataset_provenance[
            "base_dataset"
        ]["filesystem_manifest"]:
            raise RuntimeError(
                "dataset files changed during evaluation; rerun on an immutable copy"
            )
        if _filesystem_manifest(force_sidecar_path) != dataset_provenance[
            "force_sidecar"
        ]["filesystem_manifest"]:
            raise RuntimeError(
                "force sidecar files changed during evaluation; rerun on an "
                "immutable copy"
            )
        report = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_file_identity": checkpoint_identity,
            "implementation_sha256": implementation_sha256,
            "weights": loaded.state_name,
            "dataset": str(dataset_path),
            "force_sidecar": str(force_sidecar_path),
            "split": args.split,
            "split_samples": len(split_dataset),
            "selected_samples": len(selected_dataset),
            "selection_seed": int(args.seed),
            "selected_split_indices": selected_indices,
            "full_split_evaluated": (
                selected_indices is None
                and metrics["evaluated_samples"] == len(split_dataset)
            ),
            "device": str(loaded.device),
            "condition_dim": int(loaded.policy.obs_feature_dim),
            "action_shape": [
                int(loaded.policy.action_horizon),
                int(loaded.policy.action_dim),
            ],
            "num_inference_steps": int(loaded.policy.num_inference_steps),
            "image_transforms_disabled": loaded.disabled_image_transforms,
            "normalizer": _normalizer_summary(loaded.policy),
            "ft_bias_removed": bool(getattr(split_dataset, "ft_bias_removed", False)),
            "dataset_provenance": dataset_provenance,
            "metrics": metrics,
        }
    finally:
        close = getattr(split_dataset, "close", None)
        if callable(close):
            close()
        if split_dataset is not dataset:
            dataset.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output)
    print(serialized)
    print(f"wrote evaluation report: {output}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a dual-F/T checkpoint on its deterministic train or "
            "validation split without accessing robot hardware. Only load a "
            "trusted local checkpoint because torch/dill loading executes code."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override checkpoint base dataset_path (stock UMI zarr.zip).",
    )
    parser.add_argument(
        "--force-sidecar",
        default=None,
        help="Override checkpoint force_sidecar_path (bias-only native F/T zarr).",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "train"),
        default="validation",
    )
    parser.add_argument("--weights", choices=("auto", "ema_model", "model"), default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help=(
            "Seeded random samples from the split; 0 selects the full split. "
            "The chosen split indices are written to the report."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional debugging cap after sample selection; 0 means no cap.",
    )
    parser.add_argument("--prediction-repeats", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-diffusion-loss", action="store_true")
    parser.add_argument(
        "--full-dataset-hash",
        action="store_true",
        help=(
            "Hash every base-dataset and force-sidecar byte. By "
            "default the report hashes all effective non-RGB arrays and records "
            "a path/size manifest, but does not read every RGB chunk."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "data" / "eval_dual_ft" / "offline_metrics.json"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0 or args.max_batches < 0 or args.max_samples < 0:
        raise ValueError(
            "num-workers, max-samples, and max-batches must be non-negative"
        )
    run(args)


if __name__ == "__main__":
    main()
