#!/usr/bin/env python3
"""Render a GoPro video with IMU orientation/6-axis panels.

Left column: raw video frame.
Right column: XY/YZ/XZ projections of the current local xyz frame plus
accelerometer and gyroscope values from a GoPro imu_data*.json file.
Optional extra column: XY/YZ/XZ position trajectory from camera_trajectory.csv.
"""

from __future__ import annotations

import csv
import json
import pathlib

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_VIDEO = (
    ROOT_DIR
    / "data"
    / "desktop_gx010453_imu_axis"
    / "demos"
    / "demo_gx010453_2026.07.02_18.23.07"
    / "raw_video.mp4"
)
DEFAULT_IMU = DEFAULT_VIDEO.parent / "imu_data_axis_exchange.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "desktop_gx010453_imu_axis" / "gx010453_imu_axis_overlay.mp4"

AXIS_NAMES = ("X", "Y", "Z")
LOCAL_AXIS_COLORS_BGR = (
    (60, 60, 230),   # local x: red
    (60, 200, 60),   # local y: green
    (230, 140, 60),  # local z: blue
)
ACCEL_COLORS_BGR = LOCAL_AXIS_COLORS_BGR
GYRO_COLORS_BGR = (
    (90, 120, 255),
    (90, 240, 120),
    (255, 180, 90),
)
PATH_COLOR = (110, 110, 110)
START_COLOR = (210, 210, 210)
CURRENT_COLOR = (0, 230, 255)
TARGET_COLOR = (60, 100, 255)


def _put_text(img, text, org, *, scale=0.5, color=(230, 230, 230), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _stream_samples(imu: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    for dev in imu.values():
        if not isinstance(dev, dict):
            continue
        stream = dev.get("streams", {}).get(name)
        if stream is None:
            continue
        samples = [s for s in stream.get("samples", []) if isinstance(s, dict) and "value" in s and "cts" in s]
        if not samples:
            break
        t = np.asarray([float(s["cts"]) * 1e-3 for s in samples], dtype=np.float64)
        values = np.asarray([s["value"] for s in samples], dtype=np.float64)
        t = t - t[0]
        return t, values
    raise click.ClickException(f"IMU stream {name!r} not found")


def _interp_vec(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_dst), v_src.shape[1]), dtype=np.float64)
    for i in range(v_src.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, v_src[:, i])
    return out


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    fields_lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in fields_lower:
            return fields_lower[candidate]
    return None


def _load_position_csv(path: pathlib.Path, *, src_fps: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise click.ClickException(f"position csv has no header: {path}")

        x_col = _find_column(fieldnames, ("x", "pos_x", "position_x", "camera_x", "camera_position_x", "tx"))
        y_col = _find_column(fieldnames, ("y", "pos_y", "position_y", "camera_y", "camera_position_y", "ty"))
        z_col = _find_column(fieldnames, ("z", "pos_z", "position_z", "camera_z", "camera_position_z", "tz"))
        t_col = _find_column(fieldnames, ("t", "time", "time_sec", "timestamp", "timestamp_sec", "seconds"))
        frame_col = _find_column(fieldnames, ("frame", "frame_idx", "frame_index", "idx", "index"))

        rows = list(reader)

    if not rows:
        raise click.ClickException(f"position csv has no rows: {path}")

    if x_col is None or y_col is None or z_col is None:
        excluded = {c for c in (t_col, frame_col) if c is not None}
        numeric_cols = []
        for name in fieldnames:
            if name in excluded:
                continue
            try:
                float(rows[0][name])
            except (KeyError, TypeError, ValueError):
                continue
            numeric_cols.append(name)
        if len(numeric_cols) < 3:
            raise click.ClickException(
                "could not find position columns in csv. Expected x/y/z, "
                "pos_x/pos_y/pos_z, tx/ty/tz, or at least three numeric columns."
            )
        x_col, y_col, z_col = numeric_cols[:3]

    pos = np.asarray(
        [[float(row[x_col]), float(row[y_col]), float(row[z_col])] for row in rows],
        dtype=np.float64,
    )
    if t_col is not None:
        t = np.asarray([float(row[t_col]) for row in rows], dtype=np.float64)
        if np.nanmax(t) > 1e6:
            t = (t - t[0]) * 1e-9
        else:
            t = t - t[0]
    elif frame_col is not None:
        t = np.asarray([float(row[frame_col]) / src_fps for row in rows], dtype=np.float64)
        t = t - t[0]
    else:
        t = np.linspace(0.0, duration, len(pos), dtype=np.float64)
    return t, pos


def _maybe_load_position_csv(
    spec: str | None,
    *,
    video_path: pathlib.Path,
    src_fps: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if spec is None:
        return None
    if spec.lower() == "auto":
        path = video_path.parent / "camera_trajectory.csv"
        if not path.exists():
            print(f"[render] position csv not found: {path}")
            return None
    else:
        path = pathlib.Path(spec).expanduser()
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        if not path.exists():
            raise click.ClickException(f"position csv not found: {path}")
    t_pos, pos = _load_position_csv(path, src_fps=src_fps, duration=duration)
    print(f"[render] position={path} rows={len(pos)}")
    return t_pos, pos


def _interp_quat_wxyz(t_src: np.ndarray, q_wxyz: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    # Nearest-neighbor is sufficient for this debug overlay and avoids scipy
    # version differences around Slerp.
    idx = np.searchsorted(t_src, t_dst, side="left")
    idx = np.clip(idx, 0, len(t_src) - 1)
    prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(t_dst - t_src[prev]) < np.abs(t_dst - t_src[idx])
    idx[use_prev] = prev[use_prev]
    q = q_wxyz[idx]
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)


def _draw_axis_panel(canvas, rect, *, title: str, axes: tuple[int, int], rot_mat: np.ndarray, gravity: np.ndarray):
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (45, 45, 45), 1)
    _put_text(canvas, title, (x0 + 8, y0 + 22), scale=0.5, color=(255, 255, 255))

    center = (x0 + w // 2, y0 + h // 2 + 10)
    radius = min(w, h) // 3
    cv2.circle(canvas, center, radius, (60, 60, 60), 1, cv2.LINE_AA)

    a, b = axes
    origin = (x0 + 34, y0 + h - 28)
    cv2.arrowedLine(canvas, origin, (origin[0] + 45, origin[1]), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(canvas, origin, (origin[0], origin[1] - 45), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.3)
    _put_text(canvas, f"+{AXIS_NAMES[a]}", (origin[0] + 49, origin[1] + 4), scale=0.35, color=(150, 150, 150))
    _put_text(canvas, f"+{AXIS_NAMES[b]}", (origin[0] - 8, origin[1] - 49), scale=0.35, color=(150, 150, 150))

    for j in range(3):
        da = rot_mat[a, j]
        db = rot_mat[b, j]
        end = (int(center[0] + da * radius), int(center[1] - db * radius))
        cv2.arrowedLine(canvas, center, end, LOCAL_AXIS_COLORS_BGR[j], 2, cv2.LINE_AA, tipLength=0.25)
        _put_text(canvas, AXIS_NAMES[j].lower(), (end[0] + 4, end[1] + 4), scale=0.38, color=LOCAL_AXIS_COLORS_BGR[j])

    g = gravity / max(float(np.linalg.norm(gravity)), 1e-8)
    g_end = (int(center[0] + g[a] * radius * 0.85), int(center[1] - g[b] * radius * 0.85))
    cv2.arrowedLine(canvas, center, g_end, (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.25)
    _put_text(canvas, "g", (g_end[0] + 4, g_end[1] + 4), scale=0.38, color=(0, 230, 255))


def _draw_position_panel(canvas, rect, *, title: str, axes: tuple[int, int], pos_all: np.ndarray, idx: int):
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (45, 45, 45), 1)
    _put_text(canvas, title, (x0 + 8, y0 + 22), scale=0.5, color=(255, 255, 255))

    pts = pos_all[:, list(axes)]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum((hi - lo) * 1.18, 0.02)
    center = (lo + hi) / 2.0
    lo = center - span / 2.0
    hi = center + span / 2.0

    def to_px(v):
        u = (v[0] - lo[0]) / (hi[0] - lo[0])
        vv = (v[1] - lo[1]) / (hi[1] - lo[1])
        px = int(np.clip(x0 + 38 + u * (w - 62), x0 + 8, x0 + w - 8))
        py = int(np.clip(y0 + h - 30 - vv * (h - 62), y0 + 28, y0 + h - 8))
        return px, py

    a, b = axes
    origin = (x0 + 36, y0 + h - 30)
    cv2.arrowedLine(canvas, origin, (origin[0] + 45, origin[1]), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(canvas, origin, (origin[0], origin[1] - 45), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.3)
    _put_text(canvas, f"+{AXIS_NAMES[a]}", (origin[0] + 49, origin[1] + 4), scale=0.35, color=(150, 150, 150))
    _put_text(canvas, f"+{AXIS_NAMES[b]}", (origin[0] - 8, origin[1] - 49), scale=0.35, color=(150, 150, 150))

    poly = np.asarray([to_px(v) for v in pts], dtype=np.int32)
    if len(poly) >= 2:
        cv2.polylines(canvas, [poly], False, PATH_COLOR, 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[0]), 4, START_COLOR, -1)
    cv2.circle(canvas, tuple(poly[-1]), 4, TARGET_COLOR, -1)
    cv2.circle(canvas, tuple(poly[idx]), 5, CURRENT_COLOR, -1)


def _draw_plot(canvas, rect, *, title: str, t: np.ndarray, values: np.ndarray, idx: int, colors, unit: str):
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (45, 45, 45), 1)
    _put_text(canvas, title, (x0 + 8, y0 + 20), scale=0.43, color=(255, 255, 255))
    plot_x0, plot_y0 = x0 + 36, y0 + 30
    plot_w, plot_h = w - 48, h - 48
    cv2.line(canvas, (plot_x0, plot_y0 + plot_h // 2), (plot_x0 + plot_w, plot_y0 + plot_h // 2), (70, 70, 70), 1)

    lo = max(0, idx - 90)
    hi = min(len(t), idx + 1)
    if hi - lo < 2:
        return
    segment = values[lo:hi]
    max_abs = float(np.nanmax(np.abs(segment)))
    max_abs = max(max_abs, 1e-6)
    xs = np.linspace(plot_x0, plot_x0 + plot_w, hi - lo)
    for j in range(3):
        ys = plot_y0 + plot_h / 2 - (segment[:, j] / max_abs) * (plot_h * 0.43)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, colors[j], 1, cv2.LINE_AA)
    v = values[idx]
    _put_text(
        canvas,
        f"x={v[0]:+.2f} y={v[1]:+.2f} z={v[2]:+.2f} {unit}",
        (x0 + 8, y0 + h - 9),
        scale=0.36,
        color=(220, 220, 220),
    )


@click.command()
@click.option("--video", default=str(DEFAULT_VIDEO), show_default=True)
@click.option("--imu", "imu_path", default=str(DEFAULT_IMU), show_default=True)
@click.option("--output", "-o", default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--fps", default=30.0, type=float, show_default=True)
@click.option("--max_width", default=640, type=int, show_default=True)
@click.option(
    "--position_csv",
    default="auto",
    show_default=True,
    help=(
        "Optional position trajectory CSV. 'auto' looks for camera_trajectory.csv "
        "next to the input video. Use 'none' to disable."
    ),
)
def main(video, imu_path, output, fps, max_width, position_csv):
    video_path = pathlib.Path(video).expanduser().resolve()
    imu_path = pathlib.Path(imu_path).expanduser().resolve()
    out_path = pathlib.Path(output).expanduser()
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imu_path.open() as f:
        imu = json.load(f)
    t_acc, acc = _stream_samples(imu, "ACCL")
    t_gyro, gyro = _stream_samples(imu, "GYRO")
    t_grav, grav = _stream_samples(imu, "GRAV")
    t_cori, q_wxyz = _stream_samples(imu, "CORI")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise click.ClickException(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_src / src_fps if n_src > 0 else float(t_acc[-1])
    position_data = _maybe_load_position_csv(
        None if position_csv.lower() == "none" else position_csv,
        video_path=video_path,
        src_fps=src_fps,
        duration=duration,
    )
    scale = min(1.0, max_width / max(width, 1))
    left_w = int(width * scale)
    left_h = int(height * scale)
    panel_w = 430
    total_w = left_w + panel_w + (panel_w if position_data is not None else 0)
    total_h = max(left_h, 720)

    t_out = np.arange(0.0, duration, 1.0 / fps)
    acc_out = _interp_vec(t_acc, acc, t_out)
    gyro_out = _interp_vec(t_gyro, gyro, t_out)
    grav_out = _interp_vec(t_grav, grav, t_out)
    q_out = _interp_quat_wxyz(t_cori, q_wxyz, t_out)
    # scipy wants xyzw.
    rot_mats = Rotation.from_quat(q_out[:, [1, 2, 3, 0]]).as_matrix()
    pos_out = None
    if position_data is not None:
        t_pos, pos_src = position_data
        pos_out = _interp_vec(t_pos, pos_src, np.clip(t_out, t_pos[0], t_pos[-1]))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        raise click.ClickException(f"could not create writer: {out_path}")

    print(f"[render] video={video_path}")
    print(f"[render] imu={imu_path}")
    print(f"[render] source {width}x{height} {src_fps:.3f}fps frames={n_src}")
    print(f"[render] writing {len(t_out)} frames -> {out_path}")

    next_out_idx = 0
    src_idx = 0
    while next_out_idx < len(t_out):
        ok, frame = cap.read()
        if not ok:
            break
        t_src = src_idx / src_fps
        src_idx += 1
        if t_src + (0.5 / src_fps) < t_out[next_out_idx]:
            continue

        i = next_out_idx
        t = t_out[i]
        frame = cv2.resize(frame, (left_w, left_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (14, 14, 14)
        yoff = (total_h - left_h) // 2
        canvas[yoff:yoff + left_h, :left_w] = frame
        _put_text(canvas, f"GX010453  t={t:05.2f}s", (12, yoff + 28), scale=0.58)

        rx = left_w
        panel_h = 150
        _draw_axis_panel(canvas, (rx, 0, panel_w, panel_h), title="XY  local xyz + gravity", axes=(0, 1), rot_mat=rot_mats[i], gravity=grav_out[i])
        _draw_axis_panel(canvas, (rx, panel_h, panel_w, panel_h), title="YZ  local xyz + gravity", axes=(1, 2), rot_mat=rot_mats[i], gravity=grav_out[i])
        _draw_axis_panel(canvas, (rx, panel_h * 2, panel_w, panel_h), title="XZ  local xyz + gravity", axes=(0, 2), rot_mat=rot_mats[i], gravity=grav_out[i])
        _draw_plot(canvas, (rx, panel_h * 3, panel_w, 135), title="ACCL x/y/z", t=t_out, values=acc_out, idx=i, colors=ACCEL_COLORS_BGR, unit="m/s2")
        _draw_plot(canvas, (rx, panel_h * 3 + 135, panel_w, 135), title="GYRO x/y/z", t=t_out, values=gyro_out, idx=i, colors=GYRO_COLORS_BGR, unit="rad/s")
        _put_text(canvas, "axis: x=red y=green z=blue  gravity=yellow", (rx + 8, total_h - 12), scale=0.38, color=(210, 210, 210))

        if pos_out is not None:
            px = left_w + panel_w
            pos_panel_h = 215
            _draw_position_panel(canvas, (px, 0, panel_w, pos_panel_h), title="Position XY trajectory", axes=(0, 1), pos_all=pos_out, idx=i)
            _draw_position_panel(canvas, (px, pos_panel_h, panel_w, pos_panel_h), title="Position YZ trajectory", axes=(1, 2), pos_all=pos_out, idx=i)
            _draw_position_panel(canvas, (px, pos_panel_h * 2, panel_w, pos_panel_h), title="Position XZ trajectory", axes=(0, 2), pos_all=pos_out, idx=i)
            p = pos_out[i] * 100.0
            _put_text(canvas, f"pos cm: x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f}", (px + 8, total_h - 30), scale=0.4, color=(255, 255, 255))
            _put_text(canvas, "path=gray start=white current=cyan end=red", (px + 8, total_h - 12), scale=0.38, color=(210, 210, 210))

        writer.write(canvas)
        if i % 100 == 0:
            print(f"[render] frame {i + 1}/{len(t_out)}")
        next_out_idx += 1

    cap.release()
    writer.release()
    print(f"[render] wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
