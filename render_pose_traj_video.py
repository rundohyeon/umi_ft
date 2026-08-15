#!/usr/bin/env python3
"""Render a debug video for one episode of a UMI training zarr:
column 1 = camera0_rgb, column 2 = three panels (XY / YZ / ZX) each showing
the position trajectory plus the live end-effector orientation drawn as a
small local xyz axis frame (red=local x, green=local y, blue=local z).

Example:
    python3 render_pose_traj_video.py --episode 0
    python3 render_pose_traj_video.py -d axix_data_zarrfile/dataset_axis_newP.zarr.zip -e 5 -o data/pose_render/ep5.mp4
"""

from __future__ import annotations

import pathlib
import sys

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import imagecodecs.numcodecs as _icn
    _icn.register_codecs()
except Exception:
    pass
import zarr

DEFAULT_DATASET = "data/axix_data_zarrfile/dataset_axis_newP.zarr.zip"

AXIS_NAMES = ("X", "Y", "Z")
WORLD_AXIS_COLOR = (140, 140, 140)      # gray, for the base-frame legend
LOCAL_AXIS_COLORS_BGR = (
    (60, 60, 230),   # local x: red
    (60, 200, 60),   # local y: green
    (230, 140, 60),  # local z: blue
)
PATH_COLOR = (110, 110, 110)
START_COLOR = (210, 210, 210)
CURRENT_COLOR = (0, 230, 255)


def _put_text(img, text, org, *, scale=0.5, color=(230, 230, 230), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _panel_bounds(pos: np.ndarray, axes: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pts = pos[:, list(axes)]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum((hi - lo) * 1.25, 0.02)
    center = (lo + hi) / 2.0
    return center - span / 2.0, center + span / 2.0


def _make_to_px(rect: tuple[int, int, int, int], lo: np.ndarray, hi: np.ndarray):
    x0, y0, w, h = rect

    def to_px(v):
        u = (v[0] - lo[0]) / (hi[0] - lo[0])
        vv = (v[1] - lo[1]) / (hi[1] - lo[1])
        px = int(np.clip(x0 + 34 + u * (w - 56), x0 + 8, x0 + w - 8))
        py = int(np.clip(y0 + h - 26 - vv * (h - 56), y0 + 24, y0 + h - 8))
        return px, py

    return to_px


def _draw_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    pos_all: np.ndarray,
    step_idx: int,
    rot_mat_curr: np.ndarray,
    axes: tuple[int, int],
    axis_len_px: int = 34,
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (45, 45, 45), 1)
    _put_text(canvas, title, (x0 + 8, y0 + 20), scale=0.5, color=(255, 255, 255))

    lo, hi = _panel_bounds(pos_all, axes)
    to_px = _make_to_px(rect, lo, hi)

    a, b = axes
    origin_px = (x0 + 40, y0 + h - 34)
    cv2.arrowedLine(canvas, origin_px, (origin_px[0] + 46, origin_px[1]), WORLD_AXIS_COLOR, 1, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(canvas, origin_px, (origin_px[0], origin_px[1] - 46), WORLD_AXIS_COLOR, 1, cv2.LINE_AA, tipLength=0.3)
    _put_text(canvas, f"+{AXIS_NAMES[a]}", (origin_px[0] + 50, origin_px[1] + 4), scale=0.36, color=WORLD_AXIS_COLOR)
    _put_text(canvas, f"+{AXIS_NAMES[b]}", (origin_px[0] - 10, origin_px[1] - 50), scale=0.36, color=WORLD_AXIS_COLOR)

    pts_2d = pos_all[:, [a, b]]
    poly = np.asarray([to_px(v) for v in pts_2d], dtype=np.int32)
    if len(poly) >= 2:
        cv2.polylines(canvas, [poly], False, PATH_COLOR, 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[0]), 4, START_COLOR, -1)

    cur_px = tuple(poly[step_idx])
    cv2.circle(canvas, cur_px, 5, CURRENT_COLOR, -1)

    # local xyz axis frame at the current pose, projected onto this plane.
    for j in range(3):
        color = LOCAL_AXIS_COLORS_BGR[j]
        da = rot_mat_curr[a, j]
        db = rot_mat_curr[b, j]
        end = (int(cur_px[0] + da * axis_len_px), int(cur_px[1] - db * axis_len_px))
        cv2.arrowedLine(canvas, cur_px, end, color, 2, cv2.LINE_AA, tipLength=0.3)


def _video_column(rgb: np.ndarray, title: str, *, size: int = 480) -> np.ndarray:
    img = np.asarray(rgb)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    col = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size + 40, size, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    canvas[40:, :, :] = col
    _put_text(canvas, title, (10, 26), scale=0.55)
    return canvas


@click.command()
@click.option("--dataset", "-d", default=DEFAULT_DATASET, show_default=True)
@click.option("--episode", "-e", default=0, type=int, show_default=True)
@click.option("--output", "-o", default=None, help="Output mp4 path. Default: data/pose_render/ep{N}_pose_traj.mp4")
@click.option("--fps", default=30.0, type=float, show_default=True)
@click.option("--stride", default=1, type=int, show_default=True, help="Use every Nth frame (speeds up render).")
@click.option("--cam_size", default=480, type=int, show_default=True)
@click.option("--panel_size", default=340, type=int, show_default=True)
def main(dataset, episode, output, fps, stride, cam_size, panel_size):
    dataset_path = pathlib.Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT_DIR / dataset_path).resolve()
    if output is None:
        out_path = ROOT_DIR / "data" / "pose_render" / f"ep{episode}_pose_traj.mp4"
    else:
        out_path = pathlib.Path(output)
        if not out_path.is_absolute():
            out_path = (ROOT_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render] dataset={dataset_path}")
    store = zarr.ZipStore(str(dataset_path), mode="r")
    root = zarr.open_group(store=store, mode="r")

    ee = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode < 0 or episode >= len(ee):
        raise click.ClickException(f"episode {episode} out of range [0, {len(ee)})")
    start = 0 if episode == 0 else int(ee[episode - 1])
    end = int(ee[episode])
    print(f"[render] episode {episode}: frames [{start}, {end}) -> {end - start} steps")

    idxs = np.arange(start, end, stride)
    pos = np.asarray(root["data"]["robot0_eef_pos"][start:end])[::stride]
    rotvec = np.asarray(root["data"]["robot0_eef_rot_axis_angle"][start:end])[::stride]
    print(f"[render] loading {len(idxs)} camera frames (JPEG-XL decode, may take a bit)...")
    cam = np.asarray(root["data"]["camera0_rgb"][start:end])[::stride]
    store.close()

    rot_mats = Rotation.from_rotvec(rotvec).as_matrix()

    panel_h = panel_size
    panel_w = panel_size + 40
    right_w = panel_w
    right_h = panel_h * 3
    total_h = max(cam_size + 40, right_h)
    total_w = cam_size + right_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, total_h))

    plane_specs = [((0, 1), "XY (top-down)"), ((1, 2), "YZ"), ((0, 2), "XZ")]

    n = len(idxs)
    for i in range(n):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (14, 14, 14)

        cam_col = _video_column(cam[i], f"camera0_rgb  ep{episode}  frame {idxs[i]}", size=cam_size)
        canvas[: cam_col.shape[0], :cam_size, :] = cam_col

        for k, (axes, title) in enumerate(plane_specs):
            rect = (cam_size, k * panel_h, right_w, panel_h)
            _draw_panel(
                canvas, rect,
                title=title,
                pos_all=pos, step_idx=i, rot_mat_curr=rot_mats[i], axes=axes,
            )
        _put_text(
            canvas,
            f"pos(cm)=[{pos[i,0]*100:6.1f},{pos[i,1]*100:6.1f},{pos[i,2]*100:6.1f}]",
            (cam_size + 8, 42),
            scale=0.42,
            color=(255, 255, 255),
        )

        _put_text(
            canvas, "local: x=red y=grn z=blu", (cam_size + 8, total_h - 22),
            scale=0.38, color=(200, 200, 200),
        )
        _put_text(
            canvas, "world axes=gray  path=gray  current=cyan", (cam_size + 8, total_h - 6),
            scale=0.38, color=(200, 200, 200),
        )
        writer.write(canvas)
        if i % 100 == 0:
            print(f"[render] frame {i + 1}/{n}")

    writer.release()
    print(f"[render] wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
