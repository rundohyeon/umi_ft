#!/usr/bin/env python3
"""Inspect or replay one episode from the axix training zarr.

Default mode is dry-run: it reads one episode, prints trajectory stats, and
optionally writes a CSV / contact sheet. Add --execute to send the trajectory
to the Indy through UmiEnv.

Examples:
    python3 replay_axix_zarr_episode.py --episode 0
    python3 replay_axix_zarr_episode.py --episode 0 --save_csv data/debug_ep0.csv --save_images data/debug_ep0.png
    python3 replay_axix_zarr_episode.py --episode 0 --execute --start_at_current --duration_scale 2.0
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import threading
import time
import zipfile
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import yaml
import zarr

ROOT_DIR = pathlib.Path(__file__).resolve().parent
CALLER_CWD = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.common.pose_trajectory_interpolator import pose_distance  # noqa: E402
from umi.common.interpolation_util import PoseInterpolator, get_interp1d  # noqa: E402
from umi.common.precise_sleep import precise_wait  # noqa: E402

DEFAULT_DATASET = "axix_data_zarrfile/dataset_axis_newP.zarr.zip"
DEFAULT_ROBOT_CONFIG = "example/eval_robots_config_indy.yaml"


XYZ_MAP_PRESETS = {
    "identity": "x,y,z",
    # Mapping previously inferred between axix_data_zarrfile and an older raw zarr.
    # It is useful for comparison, but may not match the live Indy command frame.
    "axix_to_old_zarr": "-x,-z,-y",
}


class RosTcpPoseSubscriber:
    """Optional ROS2 TCP pose subscriber for debug visualization."""

    TYPE_MAP = {
        "geometry_msgs/msg/PoseStamped": ("geometry_msgs.msg", "PoseStamped"),
        "geometry_msgs/msg/Pose": ("geometry_msgs.msg", "Pose"),
        "geometry_msgs/msg/TransformStamped": ("geometry_msgs.msg", "TransformStamped"),
        "nav_msgs/msg/Odometry": ("nav_msgs.msg", "Odometry"),
        "std_msgs/msg/Float64MultiArray": ("std_msgs.msg", "Float64MultiArray"),
        "std_msgs/msg/Float32MultiArray": ("std_msgs.msg", "Float32MultiArray"),
    }

    def __init__(self, topic: str):
        self.topic = str(topic)
        self.latest_pose = None
        self.latest_time = None
        self.msg_type_name = None
        self._rclpy = None
        self._node = None
        self._thread = None
        self._ok = False

    @staticmethod
    def _quat_to_rotvec(q) -> np.ndarray:
        return st.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()

    @staticmethod
    def _pose_msg_to_tcp6(msg) -> np.ndarray:
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            msg = msg.pose.pose
        elif hasattr(msg, "pose"):
            msg = msg.pose
        p = msg.position
        rv = RosTcpPoseSubscriber._quat_to_rotvec(msg.orientation)
        return np.asarray([p.x, p.y, p.z, rv[0], rv[1], rv[2]], dtype=np.float64)

    @staticmethod
    def _transform_msg_to_tcp6(msg) -> np.ndarray:
        tr = msg.transform
        p = tr.translation
        rv = RosTcpPoseSubscriber._quat_to_rotvec(tr.rotation)
        return np.asarray([p.x, p.y, p.z, rv[0], rv[1], rv[2]], dtype=np.float64)

    @staticmethod
    def _array_msg_to_tcp6(msg) -> np.ndarray:
        data = np.asarray(list(msg.data), dtype=np.float64).reshape(-1)
        if data.size >= 7:
            # Heuristic: xyz + quaternion if the last 4 values look unit-length.
            q = data[3:7]
            q_norm = np.linalg.norm(q)
            if 0.5 < q_norm < 1.5:
                rv = st.Rotation.from_quat(q / q_norm).as_rotvec()
                return np.concatenate([data[:3], rv])
        if data.size >= 6:
            return data[:6].astype(np.float64)
        raise ValueError("array TCP topic must contain xyz+rotvec(6) or xyz+quat(7)")

    def _msg_to_tcp6(self, msg) -> np.ndarray:
        if self.msg_type_name in (
            "geometry_msgs/msg/PoseStamped",
            "geometry_msgs/msg/Pose",
            "nav_msgs/msg/Odometry",
        ):
            return self._pose_msg_to_tcp6(msg)
        if self.msg_type_name == "geometry_msgs/msg/TransformStamped":
            return self._transform_msg_to_tcp6(msg)
        if self.msg_type_name in (
            "std_msgs/msg/Float64MultiArray",
            "std_msgs/msg/Float32MultiArray",
        ):
            return self._array_msg_to_tcp6(msg)
        raise ValueError(f"unsupported ROS TCP topic type: {self.msg_type_name}")

    def __enter__(self):
        try:
            import importlib
            import rclpy
        except Exception as exc:
            print(f"[WARN] ROS TCP topic disabled; could not import rclpy: {exc}")
            return self

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("umi_zarr_replay_tcp_debug")
        deadline = time.time() + 2.0
        topic_types = []
        while time.time() < deadline:
            topic_types = self._node.get_topic_names_and_types()
            matches = [types for name, types in topic_types if name == self.topic]
            if matches:
                break
            rclpy.spin_once(self._node, timeout_sec=0.1)
        else:
            print(
                f"[WARN] ROS TCP topic {self.topic!r} not found. "
                "Run `ros2 topic list` and pass --ros_tcp_topic."
            )
            return self

        msg_types = matches[0]
        supported = [t for t in msg_types if t in self.TYPE_MAP]
        if not supported:
            print(
                f"[WARN] ROS TCP topic {self.topic!r} type(s) {msg_types} not supported. "
                f"Supported: {sorted(self.TYPE_MAP)}"
            )
            return self
        self.msg_type_name = supported[0]
        module_name, class_name = self.TYPE_MAP[self.msg_type_name]
        msg_cls = getattr(importlib.import_module(module_name), class_name)

        def cb(msg):
            try:
                pose = self._msg_to_tcp6(msg)
                if np.all(np.isfinite(pose)):
                    self.latest_pose = pose
                    self.latest_time = time.time()
            except Exception as exc:
                print(f"[WARN] failed to parse ROS TCP message: {exc}")

        self._node.create_subscription(msg_cls, self.topic, cb, 10)
        self._ok = True
        self._thread = threading.Thread(
            target=lambda: rclpy.spin(self._node),
            name="ros_tcp_pose_subscriber",
            daemon=True,
        )
        self._thread.start()
        print(f"ROS TCP visualization enabled: {self.topic} ({self.msg_type_name})")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None

    def get_latest(self, max_age_s: float | None = None) -> np.ndarray | None:
        if self.latest_pose is None:
            return None
        if max_age_s is not None and self.latest_time is not None:
            if time.time() - self.latest_time > float(max_age_s):
                return None
        return np.asarray(self.latest_pose, dtype=np.float64).copy()


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        ROOT_DIR / p,
        ROOT_DIR / "data" / p,
        CALLER_CWD / p,
        CALLER_CWD / "data" / p,
        ROOT_DIR.parent / p,
        ROOT_DIR.parents[2] / p,  # /home/.../dkim + "umi/..."
        ROOT_DIR.parents[2] / "umi" / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _transform_pose_frame(pose6: np.ndarray, p_mat: np.ndarray) -> np.ndarray:
    """Apply signed permutation frame transform to xyz + rotvec pose(s)."""
    pose6 = np.asarray(pose6, dtype=np.float64)
    orig_shape = pose6.shape
    flat = pose6.reshape(-1, 6)
    out = np.empty_like(flat)
    out[:, :3] = flat[:, :3] @ p_mat.T
    rot_mat = st.Rotation.from_rotvec(flat[:, 3:6]).as_matrix()
    rot_mat = np.einsum("ij,tjk,kl->til", p_mat, rot_mat, p_mat.T)
    out[:, 3:6] = st.Rotation.from_matrix(rot_mat).as_rotvec()
    return out.reshape(orig_shape)


def _transform_pose_with_T(pose6: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a full 4x4 frame transform: T_out_tcp = T @ T_in_tcp."""
    pose6 = np.asarray(pose6, dtype=np.float64)
    orig_shape = pose6.shape
    flat = pose6.reshape(-1, 6)
    R_base = np.asarray(T[:3, :3], dtype=np.float64)
    t_base = np.asarray(T[:3, 3], dtype=np.float64)
    out = np.empty_like(flat)
    out[:, :3] = flat[:, :3] @ R_base.T + t_base
    rot_mat = st.Rotation.from_rotvec(flat[:, 3:6]).as_matrix()
    rot_mat = np.einsum("ij,tjk->tik", R_base, rot_mat)
    out[:, 3:6] = st.Rotation.from_matrix(rot_mat).as_rotvec()
    return out.reshape(orig_shape)


def _parse_xyz_map(spec: str) -> tuple[np.ndarray, str]:
    spec = XYZ_MAP_PRESETS.get(spec, spec)
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise click.BadParameter(
            "--xyz_map expects x,y,z style mapping, e.g. identity, x,y,z, or -x,-z,-y"
        )
    axes = {"x": 0, "y": 1, "z": 2}
    used = set()
    p_mat = np.zeros((3, 3), dtype=np.float64)
    normalized = []
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        axis_name = part[1:] if part.startswith("-") else part
        if axis_name not in axes:
            raise click.BadParameter(f"invalid axis in --xyz_map: {part!r}")
        axis = axes[axis_name]
        if axis in used:
            raise click.BadParameter(f"--xyz_map must use x/y/z once each, got {spec!r}")
        used.add(axis)
        p_mat[row, axis] = sign
        normalized.append(("-" if sign < 0 else "") + axis_name)
    return p_mat, ",".join(normalized)


def _read_camera0_resolution(dataset_path: pathlib.Path) -> tuple[int, int]:
    """Read camera0_rgb resolution from zarr metadata without image codecs."""
    with zipfile.ZipFile(dataset_path) as zf:
        meta = json.loads(zf.read("data/camera0_rgb/.zarray"))
    _, height, width, _ = meta["shape"]
    return int(width), int(height)


def _register_image_codecs_for_debug_images(*, required: bool) -> bool:
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    except ModuleNotFoundError as exc:
        msg = (
            "Reading original zarr camera frames needs the imagecodecs package. "
            "Original-video column will be blank; current camera, xyz visualization, "
            "and trajectory CSV will still be saved."
        )
        if required:
            raise click.ClickException(msg) from exc
        print(f"[WARN] {msg}")
        return False
    register_codecs()
    return True


def _episode_slice(root: zarr.Group, episode: int) -> slice:
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if not (0 <= episode < len(episode_ends)):
        raise click.BadParameter(
            f"episode must be in [0, {len(episode_ends) - 1}], got {episode}"
        )
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return slice(start, end)


def _episode_arrays(
    root: zarr.Group,
    episode: int,
    *,
    xyz_map_mat: np.ndarray,
    robot_from_dataset_T: np.ndarray | None = None,
    load_rgb: bool = False,
    require_rgb: bool = False,
) -> dict[str, np.ndarray]:
    ep_slice = _episode_slice(root, episode)
    rgb = None
    if load_rgb:
        if _register_image_codecs_for_debug_images(required=require_rgb):
            try:
                if "camera0_rgb" in root["data"]:
                    rgb = np.asarray(root["data/camera0_rgb"][ep_slice])
            except Exception as exc:
                if require_rgb:
                    raise
                print(f"[WARN] Could not read original zarr camera frames: {exc}")
    pose6_train = np.concatenate([
        np.asarray(root["data/robot0_eef_pos"][ep_slice], dtype=np.float64),
        np.asarray(root["data/robot0_eef_rot_axis_angle"][ep_slice], dtype=np.float64),
    ], axis=-1)
    grip = np.asarray(root["data/robot0_gripper_width"][ep_slice], dtype=np.float64).reshape(-1, 1)
    pose6_mapped = _transform_pose_frame(pose6_train, xyz_map_mat)
    if robot_from_dataset_T is not None:
        pose6_mapped = _transform_pose_with_T(pose6_mapped, robot_from_dataset_T)
    return {
        "pose6_train": pose6_train,
        "pose6_robot": pose6_mapped,
        "grip": grip,
        "rgb": rgb,
    }


def _slice_arrays(arrays: dict[str, np.ndarray], max_samples: int | None) -> dict[str, np.ndarray]:
    if max_samples is None:
        return arrays
    n = max(1, min(int(max_samples), len(arrays["pose6_robot"])))
    out = dict(arrays)
    for key in ("pose6_train", "pose6_robot", "grip", "rgb"):
        value = out.get(key)
        if value is not None:
            out[key] = value[:n]
    return out


def _net_move_for_map(pose6_train: np.ndarray, xyz_map: str) -> np.ndarray:
    p_mat, _ = _parse_xyz_map(xyz_map)
    pose = _transform_pose_frame(pose6_train, p_mat)
    return (pose[-1, :3] - pose[0, :3]) * 100.0


def _print_map_preview(pose6_train: np.ndarray) -> None:
    print("coordinate map preview, full episode net move cm:")
    for name, spec in XYZ_MAP_PRESETS.items():
        net = _net_move_for_map(pose6_train, spec)
        print(f"  {name:16s} ({spec:8s}) -> {np.array2string(net, precision=3)}")


def _print_episode_stats(
    arrays: dict[str, np.ndarray],
    episode: int,
    data_frequency: float,
    *,
    label: str,
    xyz_map_label: str,
) -> None:
    pose = arrays["pose6_robot"]
    pose_train = arrays["pose6_train"]
    grip = arrays["grip"][:, 0]
    dpos = np.diff(pose[:, :3], axis=0)
    drot = (
        st.Rotation.from_rotvec(pose[1:, 3:6])
        * st.Rotation.from_rotvec(pose[:-1, 3:6]).inv()
    ).magnitude()
    path_len = float(np.linalg.norm(dpos, axis=1).sum()) if len(dpos) else 0.0
    print(f"{label}: episode={episode} samples={len(pose)} duration={len(pose) / data_frequency:.2f}s xyz_map={xyz_map_label}")
    print("training-frame start/end xyz(m):")
    print("  start", np.array2string(pose_train[0, :3], precision=5))
    print("  end  ", np.array2string(pose_train[-1, :3], precision=5))
    print("robot-frame replay start/end xyz(m):")
    print("  start", np.array2string(pose[0, :3], precision=5), "rotvec", np.array2string(pose[0, 3:6], precision=5))
    print("  end  ", np.array2string(pose[-1, :3], precision=5), "rotvec", np.array2string(pose[-1, 3:6], precision=5))
    print("net move cm:", np.array2string((pose[-1, :3] - pose[0, :3]) * 100.0, precision=3))
    print(f"path length cm: {path_len * 100.0:.2f}")
    if len(dpos):
        print(
            "step xyz delta mm mean/max:",
            np.array2string(dpos.mean(axis=0) * 1000.0, precision=3),
            f"/ {float(np.linalg.norm(dpos, axis=1).max() * 1000.0):.3f}",
        )
        print(f"step rot delta deg mean/max: {np.degrees(drot).mean():.3f} / {np.degrees(drot).max():.3f}")
    print(f"gripper width m min/max/mean: {grip.min():.5f} / {grip.max():.5f} / {grip.mean():.5f}")


def _save_csv(path: str, arrays: dict[str, np.ndarray], data_frequency: float) -> None:
    out_path = _resolve(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pose_train = arrays["pose6_train"]
    pose_robot = arrays["pose6_robot"]
    grip = arrays["grip"][:, 0]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["idx", "t"]
            + [f"train_pose_{i}" for i in range(6)]
            + [f"robot_pose_{i}" for i in range(6)]
            + ["gripper_width"]
        )
        for i in range(len(pose_robot)):
            writer.writerow([i, i / data_frequency] + pose_train[i].tolist() + pose_robot[i].tolist() + [grip[i]])
    print(f"wrote csv: {out_path}")


def _save_images(path: str, arrays: dict[str, np.ndarray], max_frames: int) -> None:
    rgb = arrays.get("rgb")
    if rgb is None:
        print("no camera0_rgb in episode; skipped image contact sheet")
        return
    out_path = _resolve(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_frames, len(rgb))
    idxs = np.linspace(0, len(rgb) - 1, n, dtype=np.int64)
    frames = []
    for idx in idxs:
        img = np.asarray(rgb[idx])
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = img.copy()
        img = cv2.resize(img, (160, 160))
        cv2.putText(img, str(idx), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        frames.append(img)
    cols = min(6, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    blank = np.zeros_like(frames[0])
    tiles = []
    for r in range(rows):
        row = frames[r * cols:(r + 1) * cols]
        row += [blank] * (cols - len(row))
        tiles.append(np.concatenate(row, axis=1))
    sheet_rgb = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out_path), sheet_rgb[:, :, ::-1])
    print(f"wrote image contact sheet: {out_path}")


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


AXIS_NAMES = ("X", "Y", "Z")
AXIS_COLORS_BGR = (
    (70, 70, 255),   # X: red
    (70, 220, 70),   # Y: green
    (255, 170, 60),  # Z: blue/orange for contrast on dark UI
)


def _draw_axis_legend(panel: np.ndarray, origin: tuple[int, int]) -> None:
    x0, y0 = origin
    _put_text(panel, "base-frame axes", (x0, y0 - 8), scale=0.42)
    vectors = [
        ((52, 0), 0),
        ((0, -48), 1),
        ((34, -34), 2),
    ]
    for (dx, dy), axis in vectors:
        color = AXIS_COLORS_BGR[axis]
        end = (x0 + dx, y0 + dy)
        cv2.arrowedLine(panel, (x0, y0), end, color, 2, cv2.LINE_AA, tipLength=0.25)
        _put_text(panel, f"+{AXIS_NAMES[axis]}", (end[0] + 4, end[1] + 5), scale=0.42, color=color)


def _draw_projection(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    plan_xyz: np.ndarray,
    target_xyz: np.ndarray,
    current_xyz: np.ndarray | None,
    ros_xyz: np.ndarray | None,
    axes: tuple[int, int],
    title: str,
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (42, 42, 42), 1)
    _put_text(panel, title, (x0 + 8, y0 + 18), scale=0.43, color=(255, 255, 255))

    pts = plan_xyz[:, list(axes)]
    extra = [target_xyz[list(axes)]]
    if current_xyz is not None:
        extra.append(current_xyz[list(axes)])
    if ros_xyz is not None:
        extra.append(ros_xyz[list(axes)])
    pts_all = np.vstack([pts] + extra)
    lo = pts_all.min(axis=0)
    hi = pts_all.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    center = (lo + hi) / 2.0
    span = np.maximum(span * 1.15, 0.02)
    lo = center - span / 2.0
    hi = center + span / 2.0

    def to_px(v):
        u = (v[0] - lo[0]) / (hi[0] - lo[0])
        vv = (v[1] - lo[1]) / (hi[1] - lo[1])
        px = int(np.clip(x0 + 30 + u * (w - 48), x0 + 8, x0 + w - 8))
        py = int(np.clip(y0 + h - 22 - vv * (h - 48), y0 + 24, y0 + h - 8))
        return px, py

    axis_a, axis_b = axes
    origin_px = (x0 + 34, y0 + h - 28)
    cv2.arrowedLine(panel, origin_px, (origin_px[0] + 58, origin_px[1]), AXIS_COLORS_BGR[axis_a], 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(panel, origin_px, (origin_px[0], origin_px[1] - 58), AXIS_COLORS_BGR[axis_b], 2, cv2.LINE_AA, tipLength=0.25)
    _put_text(panel, f"+{AXIS_NAMES[axis_a]}", (origin_px[0] + 62, origin_px[1] + 5), scale=0.38, color=AXIS_COLORS_BGR[axis_a])
    _put_text(panel, f"+{AXIS_NAMES[axis_b]}", (origin_px[0] - 12, origin_px[1] - 62), scale=0.38, color=AXIS_COLORS_BGR[axis_b])

    if len(pts) >= 2:
        poly = np.asarray([to_px(v) for v in pts], dtype=np.int32)
        cv2.polylines(panel, [poly], False, (130, 130, 130), 1, cv2.LINE_AA)
    start_px = to_px(pts[0])
    end_px = to_px(pts[-1])
    target_px = to_px(target_xyz[list(axes)])
    cv2.circle(panel, start_px, 4, (200, 200, 200), -1)
    cv2.circle(panel, end_px, 4, (120, 120, 255), -1)
    cv2.circle(panel, target_px, 5, (40, 80, 255), -1)
    if current_xyz is not None:
        cv2.circle(panel, to_px(current_xyz[list(axes)]), 5, (40, 220, 80), -1)
    if ros_xyz is not None:
        cv2.circle(panel, to_px(ros_xyz[list(axes)]), 5, (40, 190, 255), -1)


def _video_column(
    rgb: np.ndarray | None,
    title: str,
    *,
    width: int = 480,
    height: int = 720,
) -> np.ndarray:
    if rgb is None:
        col = np.zeros((height, width, 3), dtype=np.uint8)
        col[:] = (18, 18, 18)
        _put_text(col, title, (18, 34), scale=0.65)
        _put_text(col, "frame unavailable", (18, 68), scale=0.52, color=(180, 180, 180))
        return col
    img = np.asarray(rgb)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    col = cv2.resize(bgr, (width, height))
    cv2.rectangle(col, (0, 0), (width - 1, height - 1), (35, 35, 35), 1)
    _put_text(col, title, (18, 34), scale=0.65)
    return col


def _render_xyz_panel(
    *,
    actions: np.ndarray,
    step_idx: int,
    exec_t: np.ndarray,
    current_pose: np.ndarray | None,
    ros_tcp_pose: np.ndarray | None,
    xyz_map_label: str,
    episode: int,
    width: int = 640,
    height: int = 720,
) -> np.ndarray:
    right = np.zeros((height, width, 3), dtype=np.uint8)
    right[:] = (24, 24, 24)
    plan_xyz = np.asarray(actions[:, :3], dtype=np.float64)
    target = np.asarray(actions[step_idx, :6], dtype=np.float64)
    current = None if current_pose is None else np.asarray(current_pose[:6], dtype=np.float64)
    curr_xyz = None if current is None else current[:3]
    ros_tcp = None if ros_tcp_pose is None else np.asarray(ros_tcp_pose[:6], dtype=np.float64)
    ros_xyz = None if ros_tcp is None else ros_tcp[:3]

    _put_text(right, f"zarr replay debug  episode={episode}  map={xyz_map_label}", (18, 28), scale=0.55)
    _put_text(right, f"step {step_idx + 1}/{len(actions)}   t={exec_t[step_idx]:.2f}s", (18, 54), scale=0.5)
    _put_text(right, "gray:path  red:target  green:UmiEnv TCP  orange:ROS TCP", (18, 78), scale=0.43, color=(210, 210, 210))
    _draw_axis_legend(right, (520, 104))

    target_cm = target[:3] * 100.0
    _put_text(right, f"target xyz cm : {target_cm[0]: .2f}, {target_cm[1]: .2f}, {target_cm[2]: .2f}", (18, 112))
    if current is not None:
        current_cm = current[:3] * 100.0
        err_cm = (target[:3] - current[:3]) * 100.0
        _put_text(right, f"current xyz cm: {current_cm[0]: .2f}, {current_cm[1]: .2f}, {current_cm[2]: .2f}", (18, 136), color=(170, 255, 190))
        _put_text(right, f"target-current cm: {err_cm[0]: .2f}, {err_cm[1]: .2f}, {err_cm[2]: .2f}", (18, 160), color=(255, 220, 160))
    else:
        _put_text(right, "current xyz cm: unavailable", (18, 136), color=(170, 255, 190))
    if ros_tcp is not None:
        ros_cm = ros_tcp[:3] * 100.0
        ros_err_cm = (target[:3] - ros_tcp[:3]) * 100.0
        _put_text(right, f"ROS tcp xyz cm: {ros_cm[0]: .2f}, {ros_cm[1]: .2f}, {ros_cm[2]: .2f}", (18, 184), color=(120, 220, 255))
        _put_text(right, f"target-ROS cm : {ros_err_cm[0]: .2f}, {ros_err_cm[1]: .2f}, {ros_err_cm[2]: .2f}", (18, 208), color=(120, 220, 255))
    else:
        _put_text(right, "ROS tcp xyz cm: unavailable", (18, 184), color=(120, 220, 255))

    _put_text(right, f"target rotvec rad: {target[3]: .3f}, {target[4]: .3f}, {target[5]: .3f}", (18, 232))
    if current is not None:
        _put_text(right, f"current rotvec  : {current[3]: .3f}, {current[4]: .3f}, {current[5]: .3f}", (18, 256), color=(170, 255, 190))
    if ros_tcp is not None:
        _put_text(right, f"ROS rotvec      : {ros_tcp[3]: .3f}, {ros_tcp[4]: .3f}, {ros_tcp[5]: .3f}", (18, 280), color=(120, 220, 255))

    _draw_projection(right, (18, 305, 290, 165), plan_xyz, target[:3], curr_xyz, ros_xyz, (0, 1), "XY base frame")
    _draw_projection(right, (330, 305, 290, 165), plan_xyz, target[:3], curr_xyz, ros_xyz, (0, 2), "XZ base frame")
    _draw_projection(right, (18, 495, 290, 165), plan_xyz, target[:3], curr_xyz, ros_xyz, (1, 2), "YZ base frame")

    # Simple per-axis progress strip.
    strip = right[495:680, 330:620]
    cv2.rectangle(right, (330, 495), (620, 680), (42, 42, 42), 1)
    _put_text(right, "XYZ target over time", (338, 514), scale=0.43)
    vals = plan_xyz
    lo = vals.min(axis=0)
    hi = vals.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    colors = [(80, 120, 255), (80, 220, 120), (255, 180, 80)]
    labels = ["x", "y", "z"]
    for axis, color in enumerate(colors):
        y_base = 550 + axis * 38
        _put_text(right, labels[axis], (338, y_base + 4), scale=0.42, color=color)
        x_start, x_end = 365, 602
        cv2.line(right, (x_start, y_base), (x_end, y_base), (80, 80, 80), 1)
        if len(vals) >= 2:
            xs = np.linspace(x_start, x_end, len(vals)).astype(np.int32)
            ys = (y_base - 24 + (1.0 - (vals[:, axis] - lo[axis]) / span[axis]) * 48).astype(np.int32)
            poly = np.stack([xs, ys], axis=1)
            cv2.polylines(right, [poly], False, color, 1, cv2.LINE_AA)
        cx = int(x_start + (x_end - x_start) * step_idx / max(1, len(vals) - 1))
        cv2.line(right, (cx, y_base - 32), (cx, y_base + 32), (240, 240, 240), 1)

    return right


def _render_replay_debug_frame(
    original_rgb: np.ndarray | None,
    current_rgb: np.ndarray | None,
    *,
    actions: np.ndarray,
    step_idx: int,
    exec_t: np.ndarray,
    current_pose: np.ndarray | None,
    ros_tcp_pose: np.ndarray | None,
    xyz_map_label: str,
    episode: int,
    source_idx: int | None,
) -> np.ndarray:
    original_title = "1. original zarr video"
    if source_idx is not None:
        original_title += f"  frame={source_idx}"
    original_col = _video_column(original_rgb, original_title)
    current_col = _video_column(current_rgb, "2. current camera")
    xyz_col = _render_xyz_panel(
        actions=actions,
        step_idx=step_idx,
        exec_t=exec_t,
        current_pose=current_pose,
        ros_tcp_pose=ros_tcp_pose,
        xyz_map_label=xyz_map_label,
        episode=episode,
    )
    _put_text(xyz_col, "3. x/y/z visualization", (18, 704), scale=0.5, color=(230, 230, 230))
    return np.concatenate([original_col, current_col, xyz_col], axis=1)


def _load_robot_config(path: str) -> dict:
    cfg_path = _resolve(path)
    data = yaml.safe_load(cfg_path.read_text())
    robots = data["robots"]
    if len(robots) != 1:
        raise click.ClickException("This debug replay script expects exactly one robot in robot_config.")
    return robots[0]


def _robot_from_dataset_transform_from_config(rc: dict, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    transform = rc.get("indy_robot_from_dataset_transform", None)
    if transform is None:
        return None
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        raise click.ClickException(
            "indy_robot_from_dataset_transform in robot_config must be a 4x4 matrix"
        )
    if not np.all(np.isfinite(T)):
        raise click.ClickException(
            "indy_robot_from_dataset_transform in robot_config contains non-finite values"
        )
    return T


def _make_replay_actions(
    arrays: dict[str, np.ndarray],
    *,
    frequency: float,
    data_frequency: float,
    duration_scale: float,
    max_samples: int | None,
    start_at_current_pose: np.ndarray | None,
    freeze_rotation: bool,
    gripper_width: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pose = arrays["pose6_robot"].copy()
    grip = arrays["grip"].copy()
    if max_samples is not None:
        pose = pose[:max_samples]
        grip = grip[:max_samples]

    src_t = np.arange(len(pose), dtype=np.float64) / float(data_frequency)
    duration = src_t[-1] * float(duration_scale) if len(src_t) > 1 else 0.0
    exec_t = np.arange(int(np.floor(duration * frequency)) + 1, dtype=np.float64) / frequency
    src_query_t = np.clip(exec_t / float(duration_scale), src_t[0], src_t[-1])

    pose_interp = PoseInterpolator(src_t, pose)
    grip_interp = get_interp1d(src_t, grip)
    replay_pose = pose_interp(src_query_t)
    replay_grip = grip_interp(src_query_t)

    if start_at_current_pose is not None:
        data_start = replay_pose[0].copy()
        cur = np.asarray(start_at_current_pose, dtype=np.float64)
        offset_pos = cur[:3] - data_start[:3]
        replay_pose[:, :3] += offset_pos
        if freeze_rotation:
            replay_pose[:, 3:6] = cur[3:6]
        else:
            r_offset = st.Rotation.from_rotvec(cur[3:6]) * st.Rotation.from_rotvec(data_start[3:6]).inv()
            replay_pose[:, 3:6] = (r_offset * st.Rotation.from_rotvec(replay_pose[:, 3:6])).as_rotvec()

    if gripper_width is not None:
        replay_grip[:] = float(gripper_width)
    actions = np.concatenate([replay_pose, replay_grip.reshape(-1, 1)], axis=-1)
    return actions, exec_t


def _write_coordinate_debug_video(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    *,
    episode: int,
    data_frequency: float,
    xyz_map_label: str,
    ros_tcp_sub: RosTcpPoseSubscriber | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.concatenate(
        [arrays["pose6_robot"], arrays["grip"].reshape(-1, 1)],
        axis=-1,
    )
    exec_t = np.arange(len(actions), dtype=np.float64) / float(data_frequency)
    writer = None
    try:
        for i in range(len(actions)):
            original_rgb = None
            if arrays.get("rgb") is not None:
                original_rgb = arrays["rgb"][i]
            frame = _render_replay_debug_frame(
                original_rgb,
                None,
                actions=actions,
                step_idx=i,
                exec_t=exec_t,
                current_pose=None,
                ros_tcp_pose=None if ros_tcp_sub is None else ros_tcp_sub.get_latest(max_age_s=2.0),
                xyz_map_label=xyz_map_label,
                episode=episode,
                source_idx=i,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    min(float(data_frequency), 30.0),
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise click.ClickException(
                        f"failed to open coordinate debug video writer: {path}"
                    )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    if path.exists():
        print(f"wrote coordinate debug video: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"coordinate debug video was not created: {path}")


@click.command()
@click.option("--dataset", "-d", default=DEFAULT_DATASET, show_default=True)
@click.option("--robot_config", "-rc", default=DEFAULT_ROBOT_CONFIG, show_default=True)
@click.option("--output", "-o", default="data/zarr_replay_debug", show_default=True)
@click.option("--episode", "-e", default=0, type=int, show_default=True)
@click.option("--data_frequency", default=59.94, type=float, show_default=True)
@click.option("--frequency", "-f", default=20.0, type=float, show_default=True)
@click.option("--duration_scale", default=2.0, type=float, show_default=True, help=">1 replays slower.")
@click.option("--max_samples", default=None, type=int, help="Limit source samples for a short test.")
@click.option("--save_csv", default=None, help="Optional CSV path for the selected episode.")
@click.option("--save_images", default=None, help="Optional PNG contact sheet path for camera0_rgb.")
@click.option("--max_image_frames", default=18, type=int, show_default=True)
@click.option(
    "--xyz_map",
    default="identity",
    show_default=True,
    help=(
        "Dataset xyz -> robot command xyz mapping. Use identity, axix_to_old_zarr, "
        "or explicit x,y,z / -x,-z,-y."
    ),
)
@click.option(
    "--robot_from_dataset_transform/--raw_dataset_frame",
    default=True,
    show_default=True,
    help="Apply indy_robot_from_dataset_transform from robot_config to zarr TCP poses.",
)
@click.option("--execute", is_flag=True, default=False, help="Actually send trajectory to robot.")
@click.option("--start_at_current/--absolute_dataset_pose", default=True, show_default=True,
              help="Offset dataset trajectory to current TCP before replay.")
@click.option("--freeze_rotation/--replay_rotation", default=True, show_default=True)
@click.option("--gripper_width", default=0.048, type=float, show_default=True,
              help="Width sent during replay; use null only by editing code if real gripper replay is needed.")
@click.option("--debug_video/--no_debug_video", default=True, show_default=True,
              help="Save 3-column video: original/current/xyz visualization. Dry-run writes original+xyz only.")
@click.option("--debug_video_path", default=None,
              help="Optional mp4 path. Default: <output>/replay_ep##_debug.mp4")
@click.option("--trajectory_csv_path", default=None,
              help="Optional CSV path for executed target/current trajectory. Default: <output>/replay_ep##_trajectory.csv")
@click.option(
    "--ros_tcp_topic",
    default=None,
    help=(
        "Optional ROS2 topic for TCP pose visualization. Supports PoseStamped, "
        "Pose, TransformStamped, Odometry, or Float*MultiArray xyz+rotvec/xyz+quat."
    ),
)
def main(
    dataset,
    robot_config,
    output,
    episode,
    data_frequency,
    frequency,
    duration_scale,
    max_samples,
    save_csv,
    save_images,
    max_image_frames,
    xyz_map,
    robot_from_dataset_transform,
    execute,
    start_at_current,
    freeze_rotation,
    gripper_width,
    debug_video,
    debug_video_path,
    trajectory_csv_path,
    ros_tcp_topic,
):
    dataset_path = _resolve(dataset)
    if not dataset_path.exists():
        raise click.ClickException(f"dataset not found: {dataset_path}")

    rc = _load_robot_config(robot_config)
    robot_from_dataset_T = _robot_from_dataset_transform_from_config(
        rc, robot_from_dataset_transform
    )
    xyz_map_mat, xyz_map_label = _parse_xyz_map(xyz_map)
    if robot_from_dataset_T is not None:
        xyz_map_label = xyz_map_label + "+robot_from_dataset_T"
        print("using indy_robot_from_dataset_transform from robot_config")
        print(np.array2string(robot_from_dataset_T, precision=6))

    with zarr.ZipStore(str(dataset_path), mode="r") as zip_store:
        root = zarr.group(zip_store)
        obs_res = _read_camera0_resolution(dataset_path)
        arrays = _episode_arrays(
            root,
            episode,
            xyz_map_mat=xyz_map_mat,
            robot_from_dataset_T=robot_from_dataset_T,
            load_rgb=(save_images is not None) or debug_video,
            require_rgb=(save_images is not None),
        )

    _print_map_preview(arrays["pose6_train"])
    _print_episode_stats(
        arrays,
        episode,
        data_frequency,
        label="full episode",
        xyz_map_label=xyz_map_label,
    )
    selected_arrays = _slice_arrays(arrays, max_samples)
    if selected_arrays is not arrays:
        _print_episode_stats(
            selected_arrays,
            episode,
            data_frequency,
            label="selected replay segment",
            xyz_map_label=xyz_map_label,
        )
    if save_csv:
        _save_csv(save_csv, selected_arrays, data_frequency)
    if save_images:
        _save_images(save_images, selected_arrays, max_image_frames)

    if not execute:
        if debug_video:
            output_path = _resolve(output)
            output_path.mkdir(parents=True, exist_ok=True)
            video_path = (
                _resolve(debug_video_path)
                if debug_video_path
                else output_path / f"replay_ep{episode:03d}_debug.mp4"
            )
            if ros_tcp_topic:
                with RosTcpPoseSubscriber(ros_tcp_topic) as ros_tcp_sub:
                    time.sleep(0.5)
                    _write_coordinate_debug_video(
                        video_path,
                        selected_arrays,
                        episode=episode,
                        data_frequency=data_frequency,
                        xyz_map_label=xyz_map_label,
                        ros_tcp_sub=ros_tcp_sub,
                    )
            else:
                _write_coordinate_debug_video(
                    video_path,
                    selected_arrays,
                    episode=episode,
                    data_frequency=data_frequency,
                    xyz_map_label=xyz_map_label,
                )
        print("dry-run only. Add --execute to replay on the robot.")
        return

    from umi.real_world.umi_env import UmiEnv  # noqa: E402

    output_path = _resolve(output)
    output_path.mkdir(parents=True, exist_ok=True)
    video_path = None
    if debug_video:
        video_path = _resolve(debug_video_path) if debug_video_path else output_path / f"replay_ep{episode:03d}_debug.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"debug video will be saved to: {video_path}")
    trajectory_csv = _resolve(trajectory_csv_path) if trajectory_csv_path else output_path / f"replay_ep{episode:03d}_trajectory.csv"
    trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"trajectory csv will be saved to: {trajectory_csv}")

    ros_tcp_cm = RosTcpPoseSubscriber(ros_tcp_topic) if ros_tcp_topic else nullcontext(None)
    with ros_tcp_cm as ros_tcp_sub, SharedMemoryManager() as shm_manager:
        with UmiEnv(
            output_dir=str(output_path),
            robot_ip=rc["robot_ip"],
            robot_type=rc.get("robot_type", "indyrp2"),
            use_gripper=False,
            frequency=frequency,
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_reorder=None,
            init_joints=False,
            enable_multi_cam_vis=True,
            camera_obs_latency=0.125,
            robot_obs_latency=float(rc.get("robot_obs_latency", 0.0001)),
            gripper_obs_latency=0.01,
            robot_action_latency=float(rc.get("robot_action_latency", 0.0)),
            gripper_action_latency=0.0,
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            tcp_offset=rc.get("tcp_offset", 0.235),
            indy_task_rot_is_euler=rc.get("indy_task_rot_is_euler", True),
            indy_task_rot_euler_seq=rc.get("indy_task_rot_euler_seq", "zxz"),
            indy_task_rot_euler_in_degrees=rc.get("indy_task_rot_euler_in_degrees", True),
            indy_task_rot_euler_extrinsic=rc.get("indy_task_rot_euler_extrinsic", False),
            indy_task_frame_xyz_signs=rc.get("indy_task_frame_xyz_signs", (1, 1, 1)),
            shm_manager=shm_manager,
        ) as env:
            print("Waiting for camera/robot...")
            time.sleep(1.0)
            obs = env.get_obs()
            current_pose = np.concatenate([
                obs["robot0_eef_pos"][-1],
                obs["robot0_eef_rot_axis_angle"][-1],
            ])
            data_start = selected_arrays["pose6_robot"][0]
            pos_dist, rot_dist = pose_distance(data_start, current_pose)
            print("current TCP:", np.array2string(current_pose, precision=5))
            print("dataset start robot-frame:", np.array2string(data_start, precision=5))
            print(f"gap to absolute dataset start: pos={pos_dist:.4f}m rot={rot_dist:.4f}rad")

            start_pose = current_pose if start_at_current else None
            actions, exec_t = _make_replay_actions(
                selected_arrays,
                frequency=frequency,
                data_frequency=data_frequency,
                duration_scale=duration_scale,
                max_samples=None,
                start_at_current_pose=start_pose,
                freeze_rotation=freeze_rotation,
                gripper_width=gripper_width,
            )
            print(f"replay actions={len(actions)} duration={exec_t[-1]:.2f}s")
            print("first action:", np.array2string(actions[0], precision=5))
            print("last action :", np.array2string(actions[-1], precision=5))
            print("Starting in 2 seconds. Press Ctrl+C to abort.")
            time.sleep(2.0)

            eval_t_start = time.time() + 1.0
            t_start = time.monotonic() + 1.0
            env.start_episode(eval_t_start)
            precise_wait(eval_t_start, time_func=time.time)
            video_writer = None
            trajectory_file = open(trajectory_csv, "w", newline="")
            trajectory_writer = csv.writer(trajectory_file)
            trajectory_writer.writerow(
                ["step_idx", "t", "wall_time"]
                + [f"target_pos_{j}" for j in range(3)]
                + [f"target_rot_{j}" for j in range(3)]
                + ["target_gripper_width"]
                + [f"current_pos_{j}" for j in range(3)]
                + [f"current_rot_{j}" for j in range(3)]
                + [f"error_pos_{j}" for j in range(3)]
                + [f"ros_pos_{j}" for j in range(3)]
                + [f"ros_rot_{j}" for j in range(3)]
                + [f"target_ros_error_pos_{j}" for j in range(3)]
            )
            try:
                for i, t in enumerate(exec_t):
                    t_cycle_end = t_start + t
                    loop_obs = env.get_obs()
                    loop_current_pose = np.concatenate([
                        loop_obs["robot0_eef_pos"][-1],
                        loop_obs["robot0_eef_rot_axis_angle"][-1],
                    ])
                    target_row = np.asarray(actions[i], dtype=np.float64)
                    pos_error = target_row[:3] - loop_current_pose[:3]
                    ros_tcp_pose = (
                        None if ros_tcp_sub is None
                        else ros_tcp_sub.get_latest(max_age_s=2.0)
                    )
                    if ros_tcp_pose is None:
                        ros_pose_for_csv = [np.nan] * 6
                        ros_error_for_csv = [np.nan] * 3
                    else:
                        ros_pose_for_csv = ros_tcp_pose[:6].tolist()
                        ros_error_for_csv = (target_row[:3] - ros_tcp_pose[:3]).tolist()
                    trajectory_writer.writerow(
                        [i, float(t), time.time()]
                        + target_row[:3].tolist()
                        + target_row[3:6].tolist()
                        + [float(target_row[6])]
                        + loop_current_pose[:3].tolist()
                        + loop_current_pose[3:6].tolist()
                        + pos_error.tolist()
                        + ros_pose_for_csv[:3]
                        + ros_pose_for_csv[3:6]
                        + ros_error_for_csv
                    )
                    trajectory_file.flush()
                    if debug_video:
                        camera_keys = sorted(k for k in loop_obs.keys() if k.endswith("_rgb"))
                        if camera_keys:
                            source_idx = None
                            original_rgb = None
                            if selected_arrays.get("rgb") is not None:
                                source_idx = int(round((float(t) / float(duration_scale)) * float(data_frequency)))
                                source_idx = int(np.clip(source_idx, 0, len(selected_arrays["rgb"]) - 1))
                                original_rgb = selected_arrays["rgb"][source_idx]
                            frame = _render_replay_debug_frame(
                                original_rgb,
                                loop_obs[camera_keys[0]][-1],
                                actions=actions,
                                step_idx=i,
                                exec_t=exec_t,
                                current_pose=loop_current_pose,
                                ros_tcp_pose=ros_tcp_pose,
                                xyz_map_label=xyz_map_label,
                                episode=episode,
                                source_idx=source_idx,
                            )
                            if video_writer is None:
                                video_writer = cv2.VideoWriter(
                                    str(video_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    float(frequency),
                                    (frame.shape[1], frame.shape[0]),
                                )
                                if not video_writer.isOpened():
                                    raise click.ClickException(
                                        "failed to open debug video writer: "
                                        f"{video_path}. Check that the parent directory "
                                        "exists and OpenCV has mp4v support."
                                    )
                                print(f"debug video: {video_path}")
                            video_writer.write(frame)
                    env.exec_actions(
                        actions=actions[[i]],
                        timestamps=np.array([t_cycle_end - time.monotonic() + time.time()]),
                        compensate_latency=False,
                    )
                    if i % max(1, int(frequency)) == 0:
                        print(f"replay {i + 1}/{len(actions)} t={t:.2f}s")
                    precise_wait(t_cycle_end)
            finally:
                if video_writer is not None:
                    video_writer.release()
                    if video_path.exists():
                        print(
                            f"wrote debug video: {video_path} "
                            f"({video_path.stat().st_size / 1024 / 1024:.2f} MB)"
                        )
                    else:
                        print(
                            "debug video writer was released, but the file is missing: "
                            f"{video_path}"
                        )
                elif debug_video:
                    print(
                        "debug video was not written. No camera *_rgb frame was available "
                        "before replay stopped."
                    )
                trajectory_file.close()
                print(f"wrote trajectory csv: {trajectory_csv}")
                env.end_episode()
                print("Replay stopped.")


if __name__ == "__main__":
    main()
