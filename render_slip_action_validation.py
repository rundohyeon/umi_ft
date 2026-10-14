#!/usr/bin/env python3
"""Render a saved slip-action live run as a self-contained validation video."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import re

import cv2
import imageio.v2 as imageio
import numpy as np
import scipy.spatial.transform as st


ROOT = Path(__file__).resolve().parent
PHASES = ("approach", "error1", "error2", "turning")
PHASE_COLORS = (
    (76, 190, 90),
    (65, 150, 255),
    (80, 80, 245),
    (230, 180, 65),
)
BG = (24, 27, 33)
PANEL = (36, 41, 49)
TEXT = (238, 240, 244)
MUTED = (165, 172, 184)
GRID = (72, 79, 91)
GREEN = (76, 190, 90)
AMBER = (65, 190, 245)
RED = (70, 70, 245)


def put_text(image, text, xy, *, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(
        image,
        str(text),
        tuple(int(x) for x in xy),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def panel(image, xyxy, title=None):
    x0, y0, x1, y1 = (int(x) for x in xyxy)
    cv2.rectangle(image, (x0, y0), (x1, y1), PANEL, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), GRID, 1)
    if title:
        put_text(image, title, (x0 + 12, y0 + 24), scale=0.55, thickness=1)


def rgb_from_npz(data, index=-1):
    value = np.asarray(data["camera0_rgb"])[index]
    if value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating):
        value = value * 255.0
    rgb = np.clip(value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_image(value, width, height):
    h, w = value.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(
        value,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
    )
    canvas = np.full((height, width, 3), (15, 17, 21), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def motion_metrics(row):
    targets = np.asarray(row["targets_tcp7"], dtype=np.float64)
    previous = np.concatenate(
        [np.asarray(row["current_tcp6"], dtype=np.float64), [row["current_width_m"]]]
    )
    pos, rot, width = [], [], []
    for target in targets:
        pos.append(float(np.linalg.norm(target[:3] - previous[:3])))
        rot.append(
            float(
                (
                    st.Rotation.from_rotvec(target[3:6])
                    * st.Rotation.from_rotvec(previous[3:6]).inv()
                ).magnitude()
            )
        )
        width.append(abs(float(target[6] - previous[6])))
        previous = target
    return {
        "first_position_m": pos[0],
        "max_position_m": max(pos),
        "max_rotation_rad": max(rot),
        "max_width_m": max(width),
    }


def draw_bar(image, *, label, value, maximum, xyxy, color, suffix=""):
    x0, y0, x1, y1 = (int(x) for x in xyxy)
    put_text(image, label, (x0, y0 - 5), scale=0.43, color=MUTED)
    cv2.rectangle(image, (x0, y0), (x1, y1), (22, 25, 30), -1)
    ratio = float(np.clip(value / max(maximum, 1e-12), 0.0, 1.0))
    cv2.rectangle(image, (x0, y0), (x0 + int((x1 - x0) * ratio), y1), color, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), GRID, 1)
    put_text(
        image,
        f"{value:.2f}{suffix} / {maximum:.2f}{suffix}",
        (x1 - 142, y0 - 5),
        scale=0.42,
        color=TEXT,
    )


def draw_signal_plot(image, left, right, xyxy):
    x0, y0, x1, y1 = (int(x) for x in xyxy)
    for frac in (0.0, 0.5, 1.0):
        y = int(y0 + frac * (y1 - y0))
        cv2.line(image, (x0, y), (x1, y), GRID, 1)
    values = np.concatenate([left, right])
    bound = max(1.0, float(np.max(np.abs(values))) * 1.15)
    for signal, color in ((left, (80, 190, 255)), (right, (255, 170, 80))):
        xs = np.linspace(x0, x1, len(signal))
        ys = y0 + (bound - np.asarray(signal)) / (2 * bound) * (y1 - y0)
        points = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    put_text(image, f"+{bound:.1f}N", (x0 + 3, y0 + 14), scale=0.36, color=MUTED)
    put_text(image, f"-{bound:.1f}N", (x0 + 3, y1 - 4), scale=0.36, color=MUTED)
    put_text(image, "left Fz", (x1 - 135, y0 + 15), scale=0.4, color=(80, 190, 255))
    put_text(image, "right Fz", (x1 - 70, y0 + 15), scale=0.4, color=(255, 170, 80))


def draw_timeline(image, metrics, contexts, current_index, xyxy, position_limit_m):
    x0, y0, x1, y1 = (int(x) for x in xyxy)
    values = np.asarray([m["max_position_m"] * 1000.0 for m in metrics])
    limit_mm = position_limit_m * 1000.0
    ymax = max(limit_mm * 1.15, float(values.max(initial=0.0)) * 1.15)
    for value in (0.0, limit_mm, ymax):
        y = int(y1 - value / ymax * (y1 - y0))
        cv2.line(image, (x0, y), (x1, y), RED if value == limit_mm else GRID, 1)
    if len(values) > 1:
        xs = np.linspace(x0, x1, len(values))
        ys = y1 - values / ymax * (y1 - y0)
        points = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        cv2.polylines(image, [points], False, GREEN, 2, cv2.LINE_AA)
        marker_x = int(xs[current_index])
        cv2.line(image, (marker_x, y0), (marker_x, y1), (230, 230, 230), 1)
        stripe_y0 = y1 + 5
        stripe_y1 = y1 + 15
        for idx, phase in enumerate(contexts):
            left_x = int(xs[idx - 1] if idx else x0)
            right_x = int(xs[idx + 1] if idx + 1 < len(xs) else x1)
            phase_id = PHASES.index(phase) if phase in PHASES else 0
            cv2.rectangle(
                image,
                (left_x, stripe_y0),
                (right_x, stripe_y1),
                PHASE_COLORS[phase_id],
                -1,
            )
    put_text(image, f"15 mm guard", (x0 + 4, int(y1 - limit_mm / ymax * (y1-y0)) - 4), scale=0.36, color=RED)


def load_run(run_dir):
    run = json.loads((run_dir / "run.json").read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "cycles.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in records
        if row.get("input_file") and row.get("targets_tcp7") is not None
    ]
    if not rows:
        raise ValueError(f"no completed cycle records in {run_dir}")
    inputs = []
    for row in rows:
        name = row.get("input_file")
        if not name or not (run_dir / name).is_file():
            raise FileNotFoundError(f"missing saved input for cycle {row.get('cycle')}: {name}")
        inputs.append(run_dir / name)
    rejected_name = run.get("last_rejected_candidate")
    if rejected_name and (run_dir / rejected_name).is_file():
        rejected = json.loads((run_dir / rejected_name).read_text())
        rejected_input = run_dir / rejected["input_file"]
        if rejected_input.is_file():
            rows.append(rejected)
            inputs.append(rejected_input)
    return run, rows, inputs, records


def render_frame(run, rows, input_paths, metrics, index, elapsed_s, *, end_card=False):
    row = rows[index]
    data = np.load(input_paths[index])
    frame = np.full((720, 1280, 3), BG, dtype=np.uint8)
    put_text(frame, "RGB / F-T / CONTEXT POLICY VALIDATION", (20, 35), scale=0.78, thickness=2)
    put_text(
        frame,
        f"cycle {row['cycle']:03d}/{rows[-1]['cycle']:03d}   t={elapsed_s:5.2f}s   "
        f"mode={run['mode']}",
        (20, 62),
        scale=0.48,
        color=MUTED,
    )

    panel(frame, (15, 75, 520, 585), "LATEST POLICY RGB")
    current = fit_image(rgb_from_npz(data, -1), 475, 475)
    frame[100:575, 30:505] = current
    thumb_y0, thumb_h = 600, 105
    for i in range(4):
        x0 = 15 + i * 127
        thumb = fit_image(rgb_from_npz(data, i), 118, thumb_h)
        frame[thumb_y0 : thumb_y0 + thumb_h, x0 : x0 + 118] = thumb
        offset = float(row["rgb_timestamps_s"][i] - row["rgb_timestamps_s"][-1])
        put_text(frame, f"{offset:+.3f}s", (x0 + 5, thumb_y0 + 17), scale=0.38, color=TEXT)

    panel(frame, (535, 75, 1265, 235), "CONTEXT: USED BARS / MODEL MARKERS")
    probabilities = np.asarray(row["context"]["probabilities"], dtype=np.float64)
    model_probabilities = np.asarray(
        row["context"].get("model_probabilities", probabilities), dtype=np.float64
    )
    for i, (name, probability, model_probability, color) in enumerate(
        zip(PHASES, probabilities, model_probabilities, PHASE_COLORS)
    ):
        y = 115 + i * 27
        put_text(frame, name, (550, y + 14), scale=0.45, color=TEXT)
        cv2.rectangle(frame, (635, y), (1190, y + 17), (22, 25, 30), -1)
        cv2.rectangle(frame, (635, y), (635 + int(555 * probability), y + 17), color, -1)
        marker_x = 635 + int(555 * model_probability)
        cv2.line(frame, (marker_x, y - 2), (marker_x, y + 19), TEXT, 2)
        put_text(frame, f"{probability:7.4f}", (1197, y + 14), scale=0.42, color=TEXT)

    panel(frame, (535, 245, 900, 420), "WAYPOINT SAFETY")
    cfg = run["motion_safety"]
    m = metrics[index]
    draw_bar(
        frame,
        label="max position step",
        value=m["max_position_m"] * 1000.0,
        maximum=float(cfg["max_position_delta_m"]) * 1000.0,
        xyxy=(550, 285, 882, 302),
        color=GREEN,
        suffix="mm",
    )
    draw_bar(
        frame,
        label="max rotation step",
        value=m["max_rotation_rad"],
        maximum=float(cfg["max_rotation_delta_rad"]),
        xyxy=(550, 330, 882, 347),
        color=GREEN,
        suffix="rad",
    )
    draw_bar(
        frame,
        label="max gripper step",
        value=m["max_width_m"] * 1000.0,
        maximum=float(cfg["max_gripper_delta_m"]) * 1000.0,
        xyxy=(550, 375, 882, 392),
        color=GREEN,
        suffix="mm",
    )

    panel(frame, (910, 245, 1265, 420), "TIMING / FORCE")
    feedback = row["force_feedback"]
    values = (
        f"result age       {row['command_result_age_s']*1000:7.1f} ms",
        f"processing age   {row['command_processing_age_s']*1000:7.1f} ms",
        f"context history  {row['context']['history_samples']:7d} / 20",
        f"measured grasp   {feedback['measured_force_n']:7.2f} N",
        f"predicted grasp  {feedback['predicted_force_n'][0]:7.2f} N",
        f"gripper width    {row['current_width_m']*1000:7.2f} mm",
    )
    for i, value in enumerate(values):
        value_color = AMBER if value.startswith("context history") and row["context"]["history_samples"] == 0 else TEXT
        put_text(frame, value, (928, 278 + 24 * i), scale=0.46, color=value_color)

    panel(frame, (535, 430, 900, 565), "LATEST PAIRED F/T HISTORY")
    left_fz = np.asarray(data["robot0_ft_left"])[-1, :, 2]
    right_fz = np.asarray(data["robot0_ft_right"])[-1, :, 2]
    draw_signal_plot(frame, left_fz, right_fz, (550, 465, 882, 548))

    panel(frame, (910, 430, 1265, 565), "FIRST COMMAND DELTA")
    delta = np.asarray(row["delta_xyz_mm"])[0]
    put_text(frame, f"dx {delta[0]:+8.3f} mm", (930, 474), scale=0.55)
    put_text(frame, f"dy {delta[1]:+8.3f} mm", (930, 504), scale=0.55)
    put_text(frame, f"dz {delta[2]:+8.3f} mm", (930, 534), scale=0.55)
    put_text(frame, f"norm {m['first_position_m']*1000:7.3f} mm", (1080, 504), scale=0.50, color=GREEN)

    panel(frame, (535, 575, 1265, 705), "POSITION STEP TIMELINE (GREEN) + CONTEXT STRIPE")
    draw_timeline(
        frame,
        metrics,
        [item["context"]["class_name"] for item in rows],
        index,
        (550, 610, 1248, 680),
        float(cfg["max_position_delta_m"]),
    )

    if end_card:
        overlay = frame.copy()
        cv2.rectangle(overlay, (510, 260), (1245, 535), (20, 20, 75), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0.0, frame)
        cv2.rectangle(frame, (510, 260), (1245, 535), RED, 3)
        put_text(frame, "RUN STOPPED BY WAYPOINT SAFETY", (545, 315), scale=0.86, color=(220, 225, 255), thickness=2)
        match = re.search(
            r"position delta ([0-9.]+) m > ([0-9.]+) m", run.get("exit_status", "")
        )
        if match:
            actual_mm = float(match.group(1)) * 1000.0
            limit_mm = float(match.group(2)) * 1000.0
            put_text(frame, f"next waypoint: {actual_mm:.2f} mm", (565, 375), scale=0.75, color=RED, thickness=2)
            put_text(frame, f"configured limit: {limit_mm:.2f} mm", (565, 415), scale=0.68, color=TEXT)
            put_text(frame, f"excess: {actual_mm-limit_mm:.2f} mm", (565, 455), scale=0.68, color=AMBER)
        put_text(
            frame,
            "Rejected candidate input/output was not saved by this runner version.",
            (565, 500),
            scale=0.46,
            color=MUTED,
        )
    return frame


def render(run_dir: Path, output: Path, fps: float):
    run, rows, inputs, records = load_run(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = [motion_metrics(row) for row in rows]
    recorded = np.asarray([row["recorded_at"] for row in rows], dtype=np.float64)
    elapsed = recorded - recorded[0]
    tail_s = 2.0 if run.get("exit_status", "").startswith("failed:") else 0.5
    duration = float(elapsed[-1] + tail_s)
    if output.exists() and output.stat().st_size == 0:
        output.unlink()
    writer = imageio.get_writer(
        output,
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        macro_block_size=None,
    )
    try:
        for frame_number in range(int(np.ceil(duration * fps)) + 1):
            timestamp = frame_number / fps
            index = min(len(rows) - 1, max(0, bisect.bisect_right(elapsed, timestamp) - 1))
            end_card = timestamp > elapsed[-1] + 0.35 and run.get(
                "exit_status", ""
            ).startswith("failed:")
            frame = render_frame(
                run,
                rows,
                inputs,
                metrics,
                index,
                timestamp,
                end_card=end_card,
            )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

    valid_metrics = [
        value
        for value, row in zip(metrics, rows)
        if row.get("status") != "waypoint_safety_rejected"
    ]
    summary = {
        "run_dir": str(run_dir.resolve()),
        "video": str(output.resolve()),
        "completed_cycles": int(run.get("completed_cycles", len(valid_metrics))),
        "stale_results_discarded": sum(
            record.get("status") in {
                "stale_result_discarded",
                "stale_before_command_discarded",
            }
            for record in records
        ),
        "rejected_candidate_saved": any(
            row.get("status") == "waypoint_safety_rejected" for row in rows
        ),
        "recorded_duration_s": float(elapsed[-1]),
        "exit_status": run.get("exit_status"),
        "valid_max_position_step_m": float(
            max(value["max_position_m"] for value in valid_metrics)
        ),
        "position_step_limit_m": float(run["motion_safety"]["max_position_delta_m"]),
        "note": (
            "The rejected candidate was not saved; the final card uses run.json exit_status."
            if run.get("exit_status", "").startswith("failed:")
            and not any(row.get("status") == "waypoint_safety_rejected" for row in rows)
            else "Video uses completed policy cycles; stale-result records have no saved RGB input."
        ),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary, summary_path


def newest_run(base: Path):
    candidates = [path for path in base.iterdir() if path.is_dir() and (path / "run.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no run directories under {base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    run_dir = args.run_dir or newest_run(ROOT / "data/eval_contextFT4")
    run_dir = run_dir.expanduser().resolve()
    output = args.output or (
        ROOT / "exports/slip_action_validation" / f"{run_dir.name}_validation.mp4"
    )
    output = output.expanduser().resolve()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive")
    summary, summary_path = render(run_dir, output, args.fps)
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
