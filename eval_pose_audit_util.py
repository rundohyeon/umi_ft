"""
Dataset vs live TCP (xyz, especially z) audit helpers for Indy/UMI eval.

Used by inspect_ckpt_pose_eval.py and eval_real_indy.py (--dataset_zarr / auto).
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def resolve_zarr_dataset_path(dataset_path: str, ckpt_path: str) -> Optional[str]:
    """Try common locations for cfg.task.dataset.dataset_path relative to ckpt/cwd."""
    ckpt = pathlib.Path(ckpt_path).resolve()
    ckpt_dir = ckpt.parent
    ds = pathlib.Path(dataset_path).expanduser()
    # Prefer the packaged sibling dataset even when checkpoint metadata still
    # contains an existing path from the training workstation.
    candidates: List[pathlib.Path] = [ckpt_dir.parent / ds.name]
    if ds.is_absolute():
        candidates.append(ds)
    candidates.extend(
        [
            pathlib.Path.cwd() / dataset_path,
            ckpt_dir / dataset_path,
            ckpt_dir.parent / dataset_path,
            ckpt_dir.parent.parent / dataset_path,
        ]
    )
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def _register_imagecodecs() -> None:
    """Register JPEG-XL metadata codecs before ReplayBuffer opens the Zarr.

    ReplayBuffer enumerates every array, including camera0_rgb, even though the
    audit reads only low-dimensional TCP/action arrays.
    """
    try:
        import imagecodecs.numcodecs as imagecodecs_numcodecs
        from numcodecs.registry import codec_registry

        if "imagecodecs_jpegxl" not in codec_registry:
            imagecodecs_numcodecs.register_codecs()
    except ImportError:
        # The caller will receive zarr's useful codec error if the environment
        # genuinely lacks imagecodecs.
        pass


def load_tcp_z_stats_from_replay(
    zarr_path: str,
    *,
    stride: int = 20,
    pos_key: str = "robot0_eef_pos",
    action_key: Optional[str] = "action",
) -> Dict[str, Any]:
    _register_imagecodecs()
    from diffusion_policy.common.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.create_from_path(os.path.expanduser(zarr_path), mode="r")
    if pos_key not in rb:
        raise KeyError(f"{pos_key!r} not in zarr. keys={list(rb.keys())}")
    pos = rb[pos_key]
    n = int(pos.shape[0])
    st = max(1, int(stride))
    xyz = np.asarray(pos[::st])
    z = xyz[:, 2].astype(np.float64)
    out: Dict[str, Any] = {
        "path": os.path.abspath(os.path.expanduser(zarr_path)),
        "n_timesteps": n,
        "stride": st,
        "pos_z_min": float(z.min()),
        "pos_z_max": float(z.max()),
        "pos_z_p5": float(np.percentile(z, 5)),
        "pos_z_p50": float(np.percentile(z, 50)),
        "pos_z_p95": float(np.percentile(z, 95)),
        "pos_abs_xyz_max": float(np.max(np.abs(xyz))),
    }
    if action_key and action_key in rb:
        act = np.asarray(rb[action_key][::st])
        if act.shape[-1] >= 3:
            az = act[:, 2].astype(np.float64)
            out["action_z_min"] = float(az.min())
            out["action_z_max"] = float(az.max())
            out["action_z_p5"] = float(np.percentile(az, 5))
            out["action_z_p50"] = float(np.percentile(az, 50))
            out["action_z_p95"] = float(np.percentile(az, 95))
            out["action_abs_xyz_max"] = float(np.max(np.abs(act[:, :3])))
    return out


def per_episode_tcp_z_rows(
    zarr_path: str,
    *,
    max_episodes: int = 8,
    pos_key: str = "robot0_eef_pos",
) -> List[Tuple[int, float, float, float]]:
    """First N episodes: (episode_idx, z_min, z_max, z_mean)."""
    _register_imagecodecs()
    from diffusion_policy.common.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.create_from_path(os.path.expanduser(zarr_path), mode="r")
    ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
    pos = rb[pos_key]
    rows: List[Tuple[int, float, float, float]] = []
    prev = 0
    for i, end in enumerate(ends):
        if i >= max_episodes:
            break
        e = int(end)
        z = np.asarray(pos[prev:e, 2], dtype=np.float64)
        rows.append((i, float(z.min()), float(z.max()), float(z.mean())))
        prev = e
    return rows


def format_dataset_z_block(stats: Dict[str, Any]) -> str:
    lines = [
        f"  dataset: {stats.get('path', '')}",
        f"  timesteps={stats.get('n_timesteps')} stride={stats.get('stride')}",
        f"  robot0_eef_pos z (m): min={stats['pos_z_min']:.5f} max={stats['pos_z_max']:.5f} "
        f"p5/p50/p95={stats['pos_z_p5']:.5f} / {stats['pos_z_p50']:.5f} / {stats['pos_z_p95']:.5f}",
        f"  |eef_xyz|_max (subsampled): {stats['pos_abs_xyz_max']:.5f} "
        "(>>2–3 may suggest mm or wrong frame)",
    ]
    if "action_z_p50" in stats:
        lines.append(
            f"  stored action[:,2] z (m): p5/p50/p95={stats['action_z_p5']:.5f} / "
            f"{stats['action_z_p50']:.5f} / {stats['action_z_p95']:.5f} "
            f"(pose10d slice; compare to policy raw after training convention)"
        )
        lines.append(
            f"  |action_xyz|_max (first 3): {stats.get('action_abs_xyz_max', float('nan')):.5f}"
        )
    return "\n".join(lines)
