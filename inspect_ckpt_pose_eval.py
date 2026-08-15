#!/usr/bin/env python3
"""
Offline audit: ckpt pose_repr + optional training zarr TCP z / action magnitude.

  python3 inspect_ckpt_pose_eval.py -i path/to/latest.ckpt
  python3 inspect_ckpt_pose_eval.py -i path/to/latest.ckpt --zarr path/to/replay_buffer.zarr
  python3 inspect_ckpt_pose_eval.py -i path/to/latest.ckpt --zarr auto --stride 30 --episode_z 10

If --zarr auto, resolves cfg.task.dataset.dataset_path next to the ckpt / cwd.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import dill
import torch
from omegaconf import OmegaConf

from eval_pose_audit_util import (
    format_dataset_z_block,
    load_tcp_z_stats_from_replay,
    per_episode_tcp_z_rows,
    resolve_zarr_dataset_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Checkpoint .ckpt or run dir (uses checkpoints/latest.ckpt)",
    )
    p.add_argument(
        "--zarr",
        default=None,
        metavar="PATH|auto",
        help="Training replay zarr (directory or .zarr). Use 'auto' to resolve from ckpt cfg.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Subsample timesteps when scanning zarr (larger = faster, coarser stats).",
    )
    p.add_argument(
        "--episode_z",
        type=int,
        default=0,
        metavar="N",
        help="If >0, print per-episode z min/max/mean for first N episodes.",
    )
    args = p.parse_args()

    ckpt_path = args.input
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    ckpt_path = os.path.expanduser(ckpt_path)
    if not pathlib.Path(ckpt_path).is_file():
        print(f"ERROR: not a file: {ckpt_path}", file=sys.stderr)
        return 1

    ckpt_abs = str(pathlib.Path(ckpt_path).resolve())

    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    print("=== checkpoint ===")
    print("path:", ckpt_abs)
    print("dataset_path (cfg):", cfg.task.dataset.dataset_path)

    print("\n=== task.pose_repr (eval_real_indy uses these) ===")
    if hasattr(cfg.task, "pose_repr"):
        print(OmegaConf.to_yaml(cfg.task.pose_repr))
    else:
        print("(missing cfg.task.pose_repr)")

    print("=== shape_meta.obs (robot TCP / camera) ===")
    sm = cfg.task.shape_meta.obs
    for key in sorted(sm.keys()):
        if key.startswith("robot") or key.startswith("camera"):
            node = sm[key]
            sh = getattr(node, "shape", None)
            print(f"  {key}: shape={sh}")

    if args.zarr:
        zpath = args.zarr.strip()
        if zpath.lower() == "auto":
            zpath = resolve_zarr_dataset_path(
                str(cfg.task.dataset.dataset_path), ckpt_abs
            )
            if not zpath:
                print(
                    "\n=== dataset tcp z (skipped) ===\n"
                    f"  could not find zarr for {cfg.task.dataset.dataset_path!r}\n"
                    "  pass explicit --zarr /path/to/replay_buffer.zarr"
                )
            else:
                print("\n=== resolved zarr ===\n ", zpath)
        else:
            zpath = os.path.expanduser(zpath)
            if not pathlib.Path(zpath).exists():
                print(f"\nERROR: --zarr path not found: {zpath}", file=sys.stderr)
                return 1

        if zpath:
            try:
                stats = load_tcp_z_stats_from_replay(
                    zpath, stride=args.stride, action_key="action"
                )
                print("\n=== dataset TCP / action z (subsampled) ===")
                print(format_dataset_z_block(stats))
            except Exception as exc:
                print(f"\nERROR reading zarr: {exc}", file=sys.stderr)
                return 1

            if args.episode_z > 0:
                print(f"\n=== first {args.episode_z} episodes robot0_eef_pos z (m) ===")
                for row in per_episode_tcp_z_rows(
                    zpath, max_episodes=args.episode_z
                ):
                    ei, zmin, zmax, zm = row
                    print(
                        f"  ep{ei}: z_min={zmin:.5f} z_max={zmax:.5f} z_mean={zm:.5f}"
                    )

    print("\n=== notes ===")
    print(
        "- Live vs train z: run eval_real_indy with "
        "--pose_eval_audit [--dataset_zarr auto] to compare obs/action to these stats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
