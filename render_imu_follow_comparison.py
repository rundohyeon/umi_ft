#!/usr/bin/env python3
"""Render a four-panel demo-vs-execution IMU follow comparison video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ORIGINAL_VIDEO = (
    ROOT
    / "data"
    / "desktop_gx010453_imu_axis"
    / "demos"
    / "demo_gx010453_2026.07.02_18.23.07"
    / "raw_video.mp4"
)
DEFAULT_ORIGINAL_VIDEO = ROOT / "data" / "desktop_gx010453_imu_axis" / "gx010453_imu_axis_overlay.mp4"
DEFAULT_ORIGINAL_IMU = DEFAULT_RAW_ORIGINAL_VIDEO.parent / "imu_data_axis_exchange.json"
DEFAULT_EXEC_VIDEO = ROOT / "data" / "eval_run" / "videos" / "20" / "0.mp4"
DEFAULT_EXEC_LOG = (
    ROOT / "data" / "eval_run" / "eval_logs" / "ep20_20260702_082248" / "log.csv"
)
DEFAULT_OUTPUT = DEFAULT_EXEC_LOG.parent / "imu_follow_comparison.mp4"

AXIS_COLORS = (
    (70, 80, 245),   # x: red-ish in BGR
    (80, 210, 80),   # y: green
    (245, 155, 70),  # z: blue-ish
)
BG = (18, 20, 24)
PANEL_BG = (28, 31, 36)
GRID = (62, 66, 74)
TEXT = (232, 234, 238)
MUTED = (150, 154, 162)
YELLOW = (0, 220, 255)


def put_text(img, text, org, scale=0.58, color=TEXT, thickness=1):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    return cap


def video_info(cap: cv2.VideoCapture) -> tuple[float, int, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames > 0 and fps > 0 else 0.0
    return fps, frames, duration


class SequentialVideoReader:
    """Read monotonically increasing frame indices without slow MP4 seeking."""

    def __init__(self, cap: cv2.VideoCapture, fallback_shape=(450, 800, 3)):
        self.cap = cap
        self.fallback_shape = fallback_shape
        self.index = -1
        self.frame = None

    def read_at(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, frame_idx)
        if self.index > frame_idx:
            # The renderer only asks for increasing indices. If a caller changes
            # that, fall back to seeking rather than returning the wrong frame.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.index = frame_idx - 1
            self.frame = None
        while self.index < frame_idx:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.frame is not None:
                    return self.frame
                return np.full(self.fallback_shape, (30, 30, 30), dtype=np.uint8)
            self.index += 1
            self.frame = frame
        return self.frame


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((target_h, target_w, 3), PANEL_BG, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def load_stream(imu_path: Path, stream_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(imu_path.read_text())
    for dev in data.values():
        if not isinstance(dev, dict):
            continue
        stream = dev.get("streams", {}).get(stream_name)
        if not stream:
            continue
        samples = [
            sample
            for sample in stream.get("samples", [])
            if isinstance(sample, dict) and "cts" in sample and "value" in sample
        ]
        if not samples:
            continue
        t = np.asarray([float(sample["cts"]) * 1e-3 for sample in samples], dtype=np.float64)
        values = np.asarray([sample["value"] for sample in samples], dtype=np.float64)
        return t - t[0], values
    raise RuntimeError(f"IMU stream not found: {stream_name}")


def interp_vec(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_dst), values.shape[1]), dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, values[:, i])
    return out


def quat_wxyz_to_rotvec_delta(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.column_stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]])
    norms = np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    q_xyzw = q_xyzw / np.maximum(norms, 1e-8)
    rots = Rotation.from_quat(q_xyzw)
    rel = rots[0].inv() * rots
    return rel.as_rotvec()


def load_exec_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"log has no rows: {path}")
    t = np.asarray([float(row["wall_time"]) for row in rows], dtype=np.float64)
    t = t - t[0]
    obs_pos = np.asarray(
        [[float(row[f"obs_pos_{i}"]) for i in range(3)] for row in rows],
        dtype=np.float64,
    )
    obs_rot = np.asarray(
        [[float(row[f"obs_rot_{i}"]) for i in range(3)] for row in rows],
        dtype=np.float64,
    )
    rots = Rotation.from_rotvec(obs_rot)
    rel_rot = (rots[0].inv() * rots).as_rotvec()
    return t, obs_pos - obs_pos[0], rel_rot


def robust_ylim(*series: np.ndarray, min_span: float = 0.1) -> tuple[float, float]:
    values = np.concatenate([s.reshape(-1) for s in series])
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    span = max(float(hi - lo), min_span)
    mid = float((hi + lo) * 0.5)
    return mid - span * 0.62, mid + span * 0.62


def corrcoef(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = []
    for i in range(a.shape[1]):
        aa = a[:, i] - np.mean(a[:, i])
        bb = b[:, i] - np.mean(b[:, i])
        denom = np.linalg.norm(aa) * np.linalg.norm(bb)
        out.append(float(np.dot(aa, bb) / denom) if denom > 1e-12 else float("nan"))
    return np.asarray(out)


def draw_panel_header(img, rect, title, subtitle=None):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 59, 67), 1)
    put_text(img, title, (x + 16, y + 28), scale=0.7, thickness=2)
    if subtitle:
        put_text(img, subtitle, (x + 16, y + 54), scale=0.45, color=MUTED)


def draw_graph(
    img,
    rect,
    title,
    t_norm: np.ndarray,
    rotvec: np.ndarray,
    rate: np.ndarray,
    idx: int,
    ylim_rot: tuple[float, float],
    ylim_rate: tuple[float, float],
    extra_lines: list[str],
):
    x, y, w, h = rect
    draw_panel_header(img, rect, title, "solid: relative rotation rad, dotted: angular trend")
    graph_x = x + 58
    graph_y = y + 76
    graph_w = w - 86
    graph_h = h - 132
    cv2.rectangle(img, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (23, 25, 29), -1)
    cv2.rectangle(img, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), GRID, 1)

    for frac in (0.25, 0.5, 0.75):
        gx = int(graph_x + graph_w * frac)
        gy = int(graph_y + graph_h * frac)
        cv2.line(img, (gx, graph_y), (gx, graph_y + graph_h), GRID, 1)
        cv2.line(img, (graph_x, gy), (graph_x + graph_w, gy), GRID, 1)

    def to_points(values, ylim):
        lo, hi = ylim
        xs = graph_x + np.clip(t_norm, 0.0, 1.0) * graph_w
        ys = graph_y + (1.0 - np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)) * graph_h
        return np.column_stack([xs, ys]).astype(np.int32)

    for axis in range(3):
        pts = to_points(rotvec[:, axis], ylim_rot)
        cv2.polylines(img, [pts], False, AXIS_COLORS[axis], 2, cv2.LINE_AA)

        rate_pts = to_points(rate[:, axis], ylim_rate)
        for p0, p1 in zip(rate_pts[:-1:8], rate_pts[1::8]):
            cv2.line(img, tuple(p0), tuple(p1), tuple(int(c * 0.72) for c in AXIS_COLORS[axis]), 1, cv2.LINE_AA)

    cursor_x = int(graph_x + graph_w * float(t_norm[idx]))
    cv2.line(img, (cursor_x, graph_y), (cursor_x, graph_y + graph_h), YELLOW, 2, cv2.LINE_AA)

    labels = ("x", "y", "z")
    base_y = y + h - 42
    for axis, label in enumerate(labels):
        px = x + 18 + axis * 112
        cv2.rectangle(img, (px, base_y - 18), (px + 16, base_y - 2), AXIS_COLORS[axis], -1)
        put_text(
            img,
            f"{label}: {rotvec[idx, axis]:+.3f}",
            (px + 22, base_y - 4),
            scale=0.42,
            color=TEXT,
        )

    for i, line in enumerate(extra_lines[:3]):
        put_text(img, line, (x + w - 275, y + h - 58 + i * 18), scale=0.42, color=MUTED)

    put_text(img, f"{ylim_rot[0]:+.2f}", (graph_x - 52, graph_y + graph_h), scale=0.36, color=MUTED)
    put_text(img, f"{ylim_rot[1]:+.2f}", (graph_x - 52, graph_y + 10), scale=0.36, color=MUTED)


def draw_timeline(img, progress, frame_idx, total_frames, notes):
    h, w = img.shape[:2]
    y = h - 42
    x0, x1 = 24, w - 24
    cv2.line(img, (x0, y), (x1, y), (80, 84, 94), 7, cv2.LINE_AA)
    cv2.line(img, (x0, y), (int(x0 + (x1 - x0) * progress), y), YELLOW, 7, cv2.LINE_AA)
    cv2.circle(img, (int(x0 + (x1 - x0) * progress), y), 9, YELLOW, -1, cv2.LINE_AA)
    put_text(img, f"progress {progress * 100:5.1f}%   frame {frame_idx + 1}/{total_frames}", (24, h - 14), scale=0.52)
    put_text(img, notes, (w - 760, h - 14), scale=0.44, color=MUTED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-video", type=Path, default=DEFAULT_ORIGINAL_VIDEO)
    parser.add_argument("--original-imu", type=Path, default=DEFAULT_ORIGINAL_IMU)
    parser.add_argument("--exec-video", type=Path, default=DEFAULT_EXEC_VIDEO)
    parser.add_argument("--exec-log", type=Path, default=DEFAULT_EXEC_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    original_cap = open_video(args.original_video)
    exec_cap = open_video(args.exec_video)
    original_reader = SequentialVideoReader(original_cap)
    exec_reader = SequentialVideoReader(exec_cap)
    original_fps, original_frames, original_duration = video_info(original_cap)
    exec_fps, exec_frames, exec_duration = video_info(exec_cap)

    t_cori, cori_q = load_stream(args.original_imu, "CORI")
    t_gyro, gyro = load_stream(args.original_imu, "GYRO")
    original_rot = quat_wxyz_to_rotvec_delta(cori_q)
    original_t_norm = t_cori / max(float(t_cori[-1]), 1e-8)
    original_gyro_on_cori = interp_vec(t_gyro, gyro, t_cori)

    exec_t, _exec_pos, exec_rot = load_exec_log(args.exec_log)
    exec_t_norm = np.linspace(0.0, 1.0, len(exec_rot), dtype=np.float64)
    exec_rate = np.gradient(exec_rot, exec_t_norm, axis=0)

    # Compare shapes after progress-based resampling. This is a visual aid, not a
    # calibrated frame transform between GoPro and robot coordinates.
    sample_norm = np.linspace(0.0, 1.0, 600, dtype=np.float64)
    original_resampled = interp_vec(original_t_norm, original_rot, sample_norm)
    exec_resampled = interp_vec(exec_t_norm, exec_rot, sample_norm)
    shape_corr = corrcoef(original_resampled, exec_resampled)

    output_duration = min(original_duration, exec_duration)
    if args.max_seconds > 0:
        output_duration = min(output_duration, args.max_seconds)
    total_frames = max(1, int(round(output_duration * args.fps)))
    canvas_size = (args.width, args.height)
    panel_w = args.width // 2
    top_h = int(args.height * 0.50)
    graph_h = args.height - top_h - 62

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        canvas_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output writer: {args.output}")

    ylim_rot = robust_ylim(original_rot, exec_rot, min_span=0.12)
    ylim_original_rate = robust_ylim(original_gyro_on_cori, min_span=0.3)
    ylim_exec_rate = robust_ylim(exec_rate, min_span=0.3)

    for out_idx in range(total_frames):
        progress = out_idx / max(total_frames - 1, 1)
        original_idx = min(original_frames - 1, int(round(progress * max(original_frames - 1, 0))))
        exec_idx = min(exec_frames - 1, int(round(progress * max(exec_frames - 1, 0))))
        imu_idx = min(len(original_rot) - 1, int(round(progress * (len(original_rot) - 1))))
        log_idx = min(len(exec_rot) - 1, int(round(progress * (len(exec_rot) - 1))))

        canvas = np.full((args.height, args.width, 3), BG, dtype=np.uint8)

        original_frame = fit_frame(original_reader.read_at(original_idx), (panel_w, top_h))
        exec_frame = fit_frame(exec_reader.read_at(exec_idx), (panel_w, top_h))
        canvas[0:top_h, 0:panel_w] = original_frame
        canvas[0:top_h, panel_w : panel_w * 2] = exec_frame
        cv2.rectangle(canvas, (0, 0), (panel_w - 1, top_h - 1), (55, 59, 67), 1)
        cv2.rectangle(canvas, (panel_w, 0), (args.width - 1, top_h - 1), (55, 59, 67), 1)

        put_text(canvas, "Column 1: original demo video (hand)", (18, 34), scale=0.75, thickness=2)
        put_text(
            canvas,
            f"{progress * original_duration:05.1f}s / {original_duration:05.1f}s",
            (18, 66),
            scale=0.52,
            color=MUTED,
        )
        put_text(canvas, "Column 2: robot execution video", (panel_w + 18, 34), scale=0.75, thickness=2)
        put_text(
            canvas,
            f"{progress * exec_duration:05.1f}s / {exec_duration:05.1f}s",
            (panel_w + 18, 66),
            scale=0.52,
            color=MUTED,
        )

        draw_graph(
            canvas,
            (0, top_h, panel_w, graph_h),
            "Column 3: original IMU orientation",
            original_t_norm,
            original_rot,
            original_gyro_on_cori,
            imu_idx,
            ylim_rot,
            ylim_original_rate,
            [
                f"gyro x/y/z rad/s",
                f"source CORI + GYRO",
                f"samples {len(original_rot)}",
            ],
        )
        draw_graph(
            canvas,
            (panel_w, top_h, panel_w, graph_h),
            "Column 4: execution observed rotation",
            exec_t_norm,
            exec_rot,
            exec_rate,
            log_idx,
            ylim_rot,
            ylim_exec_rate,
            [
                "obs_rot relative rotvec",
                f"shape corr x/y/z {shape_corr[0]:+.2f} {shape_corr[1]:+.2f} {shape_corr[2]:+.2f}",
                f"log rows {len(exec_rot)}",
            ],
        )

        draw_timeline(
            canvas,
            progress,
            out_idx,
            total_frames,
            "Progress-synced: use this to judge whether execution follows the demo motion shape.",
        )
        writer.write(canvas)

        if out_idx % max(1, int(args.fps * 5)) == 0:
            print(f"[render] {out_idx + 1}/{total_frames} ({progress * 100:.1f}%)")

    writer.release()
    original_cap.release()
    exec_cap.release()
    print(f"[render] wrote {args.output}")


if __name__ == "__main__":
    main()
