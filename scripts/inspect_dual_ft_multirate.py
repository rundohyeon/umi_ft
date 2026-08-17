#!/usr/bin/env python3
"""Read-only audit for dataset_multirate_clean.zarr.zip.

This script intentionally opens the nested Zarr through ZipStore and never
rewrites metadata or pixels. Outputs are ordinary reports and figures outside
the source ZIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.nested_zarr import (
    detect_zarr_prefix,
    open_nested_zip_group,
    sha256_file,
)


def cumulative_slices(ends):
    ends = np.asarray(ends, dtype=np.int64).reshape(-1)
    return list(zip(np.r_[0, ends[:-1]], ends))


def episode_deltas(timestamps, ends):
    return np.concatenate(
        [np.diff(timestamps[start:end]) for start, end in cumulative_slices(ends)]
    )


def numeric_stats(values):
    # Match UmiDualFTDataset._stats exactly: training values and accumulation
    # are float32, which is also what is stored in LinearNormalizer.
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def dt_report(timestamps, ends):
    dt = episode_deltas(timestamps, ends)
    if np.any(dt <= 0):
        raise ValueError("timestamps are not strictly increasing within episodes")
    return {
        "frequency_hz_from_median_dt": float(1.0 / np.median(dt)),
        "dt_min_s": float(np.min(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "dt_median_s": float(np.median(dt)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_max_s": float(np.max(dt)),
        "duplicate_count": int(np.sum(dt == 0)),
        "nonmonotonic_count": int(np.sum(dt < 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "zip_path",
        nargs="?",
        default="data/dataset_multirate_clean.zarr.zip",
    )
    parser.add_argument(
        "--output-dir", default="docs/dataset_multirate_clean_audit"
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    register_codecs()
    zip_path = Path(args.zip_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before_hash = sha256_file(zip_path)
    nested = detect_zarr_prefix(zip_path)
    store, root, prefix = open_nested_zip_group(zip_path, prefix=nested.prefix)
    try:
        data = root["data"]
        meta = root["meta"]
        rgb = data["rgb_0"]
        rgb_t = np.asarray(data["rgb_time_stamps_0"][:], dtype=np.float64).reshape(-1)
        pose_t = np.asarray(data["robot_time_stamps_0"][:], dtype=np.float64).reshape(-1)
        gripper_t = np.asarray(
            data["gripper_time_stamps_0"][:], dtype=np.float64
        ).reshape(-1)
        left = np.asarray(data["wrench_left_0"][:], dtype=np.float32)
        right = np.asarray(data["wrench_right_0"][:], dtype=np.float32)
        ft_t = np.asarray(data["wrench_time_stamps_0"][:], dtype=np.float64).reshape(-1)
        rgb_ends = np.asarray(meta["episode_rgb0_len"][:], dtype=np.int64).reshape(-1)
        pose_ends = np.asarray(meta["episode_robot0_len"][:], dtype=np.int64).reshape(-1)
        gripper_ends = np.asarray(
            meta["episode_gripper0_len"][:], dtype=np.int64
        ).reshape(-1)
        ft_ends = np.asarray(meta["episode_wrench0_len"][:], dtype=np.int64).reshape(-1)

        if tuple(rgb.shape) != (len(rgb_t), 224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"RGB metadata contract mismatch: {rgb.shape}, {rgb.dtype}")
        if left.shape != (len(ft_t), 6) or right.shape != (len(ft_t), 6):
            raise ValueError("left/right wrench arrays must be [N,6]")
        if not np.array_equal(rgb_t, pose_t) or not np.array_equal(rgb_t, gripper_t):
            raise ValueError("RGB, pose, and gripper policy anchor grids differ")
        if not np.array_equal(rgb_ends, pose_ends) or not np.array_equal(
            rgb_ends, gripper_ends
        ):
            raise ValueError("RGB, pose, and gripper episode ends differ")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("wrench contains NaN or Inf")

        rgb_slices = cumulative_slices(rgb_ends)
        ft_slices = cumulative_slices(ft_ends)
        dropped = []
        causal_ages = []
        for episode, ((rgb_start, rgb_end), (ft_start, ft_end)) in enumerate(
            zip(rgb_slices, ft_slices)
        ):
            anchors = rgb_t[rgb_start:rgb_end]
            sensor_t = ft_t[ft_start:ft_end]
            left_indices = np.searchsorted(sensor_t, anchors, side="right") - 1
            right_indices = np.searchsorted(sensor_t, anchors, side="right") - 1
            has_left = left_indices >= 0
            has_right = right_indices >= 0
            valid = has_left & has_right
            selected = sensor_t[left_indices[valid]]
            if np.any(selected > anchors[valid]):
                raise AssertionError("future wrench selected")
            causal_ages.extend((anchors[valid] - selected).tolist())
            n_dropped = int(np.sum(~valid))
            dropped.append(
                {
                    "episode": episode,
                    "anchors": int(len(anchors)),
                    "dropped": n_dropped,
                    "ratio": float(n_dropped / len(anchors)),
                    "left_missing": int(np.sum(~has_left)),
                    "right_missing": int(np.sum(~has_right)),
                    "first_anchor_timestamp": float(anchors[0]),
                    "first_left_ft_timestamp": float(sensor_t[0]),
                    "first_right_ft_timestamp": float(sensor_t[0]),
                }
            )

        with open(output_dir / "dropped_anchors.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=dropped[0].keys())
            writer.writeheader()
            writer.writerows(dropped)

        n_episodes = len(rgb_ends)
        n_val = min(max(1, round(n_episodes * args.val_ratio)), n_episodes - 1)
        rng = np.random.default_rng(seed=args.seed)
        val_episodes = np.sort(rng.choice(n_episodes, size=n_val, replace=False))
        val_mask = np.zeros(n_episodes, dtype=bool)
        val_mask[val_episodes] = True
        train_wrench_indices = np.concatenate(
            [
                np.arange(start, end, dtype=np.int64)
                for episode, (start, end) in enumerate(ft_slices)
                if not val_mask[episode]
            ]
        )
        left_stats = numeric_stats(left[train_wrench_indices])
        right_stats = numeric_stats(right[train_wrench_indices])

        decode_episodes = [0, n_episodes // 2, n_episodes - 1]
        decode_rows = []
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for row, episode in enumerate(decode_episodes):
            start, end = rgb_slices[episode]
            frame_indices = [start, start + (end - start - 1) // 2, end - 1]
            for column, (label, frame_index) in enumerate(
                zip(("start", "middle", "end"), frame_indices)
            ):
                frame = rgb[int(frame_index)]
                if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                    raise ValueError(
                        "decoded RGB differs from Zarr metadata: "
                        f"episode={episode}, index={frame_index}, "
                        f"shape={frame.shape}, dtype={frame.dtype}"
                    )
                record = {
                    "episode": episode,
                    "position": label,
                    "global_index": int(frame_index),
                    "shape": list(frame.shape),
                    "dtype": str(frame.dtype),
                    "channel_order": "HWC RGB",
                    "min": int(np.min(frame)),
                    "max": int(np.max(frame)),
                    "mean": float(np.mean(frame)),
                }
                decode_rows.append(record)
                axes[row, column].imshow(frame)
                axes[row, column].set_title(
                    f"ep {episode} {label}\nidx {frame_index} [{record['min']},{record['max']}]"
                )
                axes[row, column].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "rgb_decode_montage.png", dpi=140)
        plt.close(fig)

        plot_episode = n_episodes // 2
        ft_start, ft_end = ft_slices[plot_episode]
        local_t = ft_t[ft_start:ft_end] - ft_t[ft_start]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for channel in range(6):
            axes[0].plot(local_t, left[ft_start:ft_end, channel], label=f"L{channel}")
            axes[1].plot(local_t, right[ft_start:ft_end, channel], label=f"R{channel}")
        axes[0].set_ylabel("left native wrench")
        axes[1].set_ylabel("right native wrench")
        axes[1].set_xlabel("episode time [s]")
        axes[0].legend(ncol=6)
        axes[1].legend(ncol=6)
        fig.tight_layout()
        fig.savefig(output_dir / "ft_timeseries.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for channel in range(6):
            axes[0, channel].hist(left[train_wrench_indices, channel], bins=80)
            axes[0, channel].set_title(f"left {channel}")
            axes[1, channel].hist(right[train_wrench_indices, channel], bins=80)
            axes[1, channel].set_title(f"right {channel}")
        fig.tight_layout()
        fig.savefig(output_dir / "ft_histograms.png", dpi=140)
        plt.close(fig)

        rgb_rate = dt_report(rgb_t, rgb_ends)
        ft_rate = dt_report(ft_t, ft_ends)
        action_frequency = (
            rgb_rate["frequency_hz_from_median_dt"] / 3.0
        )
        age = np.asarray(causal_ages, dtype=np.float64)
        report = {
            "zip_path": str(zip_path),
            "zip_size_bytes": os.path.getsize(zip_path),
            "sha256_before": before_hash,
            "detected_nested_prefix": prefix,
            "open_mode": "read-only ZipStore + zarr.open_group(path=prefix)",
            "jpegxl_compatibility_fields": {
                "bitspersample": None,
                "squeeze": None,
                "policy": "accept and log only when None; reject non-None",
            },
            "zarr_attrs": dict(root.attrs),
            "rgb_metadata": {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "chunks": list(rgb.chunks),
                "compressor": repr(rgb.compressor),
            },
            "rgb_decode": decode_rows,
            "rgb_timeline": rgb_rate,
            "ft_timeline": ft_rate,
            "policy_anchor": "latest selected rgb_0 timestamp; pose/gripper share exact grid",
            "causal_lookup": "searchsorted(timestamp, anchor, side='right') - 1",
            "causal_alignment_age_s": {
                "min": float(np.min(age)),
                "mean": float(np.mean(age)),
                "median": float(np.median(age)),
                "p95": float(np.percentile(age, 95)),
                "max": float(np.max(age)),
            },
            "dropped_anchor_total": int(sum(x["dropped"] for x in dropped)),
            "anchor_total": int(sum(x["anchors"] for x in dropped)),
            "dropped_anchor_ratio": float(
                sum(x["dropped"] for x in dropped)
                / sum(x["anchors"] for x in dropped)
            ),
            "dropped_anchor_by_episode": dropped,
            "train_split": {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "val_episodes": val_episodes.tolist(),
                "train_episode_count": int(np.sum(~val_mask)),
                "wrench_sample_count": int(len(train_wrench_indices)),
            },
            "train_split_wrench_stats": {
                "channel_order_per_side": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "left": left_stats,
                "right": right_stats,
            },
            "actual_action_frequency_hz": float(action_frequency),
            "prediction_horizon": 16,
            "default_n_action_steps": 2,
            "default_replanning_interval_ms": float(2000.0 / action_frequency),
            "tensor_shapes": {
                "camera0_rgb": [2, 3, 224, 224],
                "robot0_eef_pos": [2, 3],
                "robot0_eef_rot_axis_angle": [2, 6],
                "robot0_gripper_width": [2, 1],
                "robot0_eef_rot_axis_angle_wrt_start": [2, 6],
                "robot0_ft_left": [32, 6],
                "robot0_ft_right": [32, 6],
                "action": [16, 10],
                "model_condition": [800],
            },
        }
    finally:
        store.close()

    after_hash = sha256_file(zip_path)
    report["sha256_after"] = after_hash
    report["sha256_unchanged"] = before_hash == after_hash
    if before_hash != after_hash:
        raise RuntimeError("source ZIP SHA-256 changed during read-only inspection")
    with open(output_dir / "inspection.json", "w") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
