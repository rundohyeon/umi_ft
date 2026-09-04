#!/usr/bin/env python3
"""Offline inference for a portable NPZ sensor recording."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from valve_state_classifier import ERROR_REASON_NAMES, PHASE_NAMES, ValveStateRuntime


REQUIRED_KEYS = (
    "rgb",
    "rgb_timestamp_s",
    "position_m",
    "rotation_axis_angle_rad",
    "gripper_width_m",
    "wrench_12d",
    "wrench_timestamp_s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ containing synchronized RGB/robot/F-T arrays")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "model/final.pt")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"input is missing keys: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}

    count = len(arrays["rgb"])
    for key in ("rgb_timestamp_s", "position_m", "rotation_axis_angle_rad", "gripper_width_m"):
        if len(arrays[key]) != count:
            raise ValueError(f"{key} length differs from rgb")
    if arrays["wrench_12d"].shape != (len(arrays["wrench_timestamp_s"]), 12):
        raise ValueError("wrench_12d must have shape (M,12)")

    runtime = ValveStateRuntime(args.checkpoint, device=args.device)
    wrench_index = 0
    rows = []
    for index, rgb_time in enumerate(arrays["rgb_timestamp_s"]):
        while (
            wrench_index < len(arrays["wrench_timestamp_s"])
            and arrays["wrench_timestamp_s"][wrench_index] <= rgb_time
        ):
            runtime.append_wrench(
                float(arrays["wrench_timestamp_s"][wrench_index]),
                arrays["wrench_12d"][wrench_index],
            )
            wrench_index += 1
        prediction = runtime.predict(
            timestamp_s=float(rgb_time),
            rgb=arrays["rgb"][index],
            position_m=arrays["position_m"][index],
            rotation_axis_angle_rad=arrays["rotation_axis_angle_rad"][index],
            gripper_width_m=float(np.asarray(arrays["gripper_width_m"][index]).reshape(-1)[0]),
        )
        row = {
            "frame": index,
            "timestamp_s": prediction.timestamp_s,
            "phase_id": prediction.phase_id,
            "phase": prediction.phase,
            "confidence": prediction.confidence,
            "error_reason_id": prediction.error_reason_id,
            "error_reason": prediction.error_reason,
            "warmed_up": int(prediction.warmed_up),
            "rgb_context_s": prediction.rgb_context_s,
        }
        row.update({f"p_{name}": prediction.phase_probabilities[name] for name in PHASE_NAMES})
        row.update({f"p_reason_{name}": prediction.error_reason_probabilities[name] for name in ERROR_REASON_NAMES})
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"inference {index + 1}/{count}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
