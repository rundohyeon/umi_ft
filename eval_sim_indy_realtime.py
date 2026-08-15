#!/usr/bin/env python3
"""Closed-loop eval_real_indy policy execution for Indy simulation.

This script intentionally does not pre-generate a trajectory.  Each cycle:

1. read the next zarr image horizon,
2. read the current TCP pose from either Neuromeka SDK emulator or MoveIt FK,
3. build the same policy input path used by eval_real_indy.py,
4. decode policy output with eval_real_indy.py's Indy TCP transforms,
5. execute the near-term TCP target(s) through the selected backend.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
import yaml
import zarr
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.common.cv2_util import get_image_transform  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from umi.common.cv_util import draw_predefined_mask  # noqa: E402
from umi.real_world.real_inference_util import (  # noqa: E402
    get_real_obs_resolution,
    get_real_umi_obs_dict,
)

import eval_real_indy as eval_indy  # noqa: E402
from eval_real_indy import (  # noqa: E402
    _apply_policy_tcp7_rot_roundtrip,
    _apply_slam_frame_fix,
    _apply_slam_frame_fix_to_obs,
    _apply_slam_frame_fix_to_start_pose,
    _check_finite_array,
    _check_policy_inputs_finite,
    _decode_real_umi_action_checked,
    _disable_policy_image_transforms,
    _limit_policy_waypoints,
    _parse_tcp_delta_scales,
    _print_motion_debug,
    _print_policy_action_debug,
    _set_robot_dataset_transform,
    _transform_tcp7_action,
)


DEFAULT_CKPT = ROOT_DIR / "artifacts" / "0709_robot_tcp" / "checkpoints" / "latest.ckpt"
DEFAULT_ROBOT_CONFIG = ROOT_DIR / "example" / "eval_robots_config_indy.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "eval_sim_indy_realtime"
DEFAULT_DATASET = ROOT_DIR / "artifacts" / "0709_robot_tcp" / "dataset_robot_tcp.zarr.zip"
SYNTHETIC_GRIPPER_WIDTH = np.float32(0.05651384)


@dataclass
class PolicyBundle:
    cfg: Any
    policy: Any
    device: torch.device
    obs_pose_repr: str
    action_pose_repr: str
    policy_rot_rt: bool
    policy_rot_seq: str
    policy_rot_ext: bool


def _resolve(path: str | os.PathLike, *, must_exist: bool = True) -> pathlib.Path:
    raw = pathlib.Path(os.path.expanduser(str(path)))
    candidates = [raw] if raw.is_absolute() else [pathlib.Path.cwd() / raw, ROOT_DIR / raw]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    if must_exist:
        raise FileNotFoundError(str(candidates[0]))
    return candidates[0].resolve()


def _load_robot_config(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    robots = data["robots"]
    if len(robots) != 1:
        raise ValueError("eval_sim_indy_realtime expects exactly one robot")
    return robots[0]


def _load_policy(args: argparse.Namespace, rc: dict[str, Any]) -> PolicyBundle:
    ckpt_path = _resolve(args.input)
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is not None:
        cfg.policy.obs_encoder.pretrained = False
    if hasattr(cfg, "task") and OmegaConf.select(cfg, "task.dataset.dataset_path") is not None:
        cfg.task.dataset.dataset_path = str(_resolve(args.dataset))

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if args.disable_eval_image_aug:
        disabled = _disable_policy_image_transforms(policy)
        if disabled:
            print("eval image augmentation disabled:", disabled)
    policy.num_inference_steps = int(args.policy_num_inference_steps)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        policy.eval().to(device)
    except Exception as exc:
        if device.type == "cuda":
            print(f"CUDA init failed ({exc}); falling back to CPU")
            device = torch.device("cpu")
            policy.eval().to(device)
        else:
            raise

    policy_rot_seq = rc.get("indy_policy_tcp7_rot_euler_seq")
    policy_rot_ext = rc.get("indy_policy_tcp7_rot_euler_extrinsic")
    if policy_rot_seq is None:
        policy_rot_seq = rc.get("indy_task_rot_euler_seq", "xyz")
    if policy_rot_ext is None:
        policy_rot_ext = rc.get("indy_task_rot_euler_extrinsic", True)

    return PolicyBundle(
        cfg=cfg,
        policy=policy,
        device=device,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        action_pose_repr=cfg.task.pose_repr.action_pose_repr,
        policy_rot_rt=bool(rc.get("indy_policy_tcp7_rot_euler_roundtrip", False)),
        policy_rot_seq=str(policy_rot_seq),
        policy_rot_ext=bool(policy_rot_ext),
    )


def _stamp_to_float(stamp) -> float:
    t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return t if t > 0.0 else time.time()


def _duration_msg(Duration, seconds: float):
    seconds = max(0.0, float(seconds))
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    return st.Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()


def _nearest_history(history: deque[tuple[float, np.ndarray]], target_t: float) -> np.ndarray:
    if not history:
        raise RuntimeError("observation history is empty")
    idx = min(range(len(history)), key=lambda i: abs(history[i][0] - target_t))
    return np.asarray(history[idx][1])


class ZarrImageEpisodeProvider:
    """Sequential zarr image source for Gazebo closed-loop tests.

    The robot state still comes from Gazebo.  Only camera0_rgb is replayed from
    the selected zarr episode, one control step at a time.
    """

    def __init__(
        self,
        *,
        zarr_path: pathlib.Path,
        episode: int,
        obs_res: tuple[int, int],
        horizon: int,
        down_sample_steps: int,
    ):
        try:
            import imagecodecs.numcodecs as imagecodecs_numcodecs

            imagecodecs_numcodecs.register_codecs()
        except Exception:
            pass

        self.zarr_path = pathlib.Path(zarr_path)
        self.episode = int(episode)
        self.obs_res = tuple(obs_res)
        self.horizon = int(horizon)
        self.down_sample_steps = int(down_sample_steps)
        self.control_step = 0
        self.store = None
        if self.zarr_path.name.endswith(".zarr.zip") or self.zarr_path.suffix == ".zip":
            self.store = zarr.ZipStore(str(self.zarr_path), mode="r")
            self.root = zarr.open_group(store=self.store, mode="r")
        else:
            self.root = zarr.open_group(str(self.zarr_path), mode="r")
        ends = np.asarray(self.root["meta"]["episode_ends"][:], dtype=np.int64)
        if not (0 <= self.episode < len(ends)):
            raise ValueError(f"episode {self.episode} out of range [0, {len(ends)})")
        self.start_idx = 0 if self.episode == 0 else int(ends[self.episode - 1])
        self.end_idx = int(ends[self.episode])
        if "camera0_rgb" not in self.root["data"]:
            raise KeyError(f"camera0_rgb not found in {self.zarr_path}")
        self.images = self.root["data"]["camera0_rgb"]

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    @property
    def current_idx(self) -> int:
        idx = self.start_idx + self.control_step * self.down_sample_steps
        return int(np.clip(idx, self.start_idx, self.end_idx - 1))

    @property
    def is_done(self) -> bool:
        return self.current_idx >= self.end_idx - 1

    def advance(self, n_control_steps: int) -> None:
        self.control_step += max(1, int(n_control_steps))

    def get_horizon(self) -> np.ndarray:
        current_idx = self.current_idx
        num_valid = min(
            self.horizon,
            (current_idx - self.start_idx) // self.down_sample_steps + 1,
        )
        slice_start = current_idx - (num_valid - 1) * self.down_sample_steps
        imgs = np.asarray(
            self.images[slice_start : current_idx + 1 : self.down_sample_steps]
        )
        if imgs.shape[0] < self.horizon:
            padding = np.repeat(imgs[:1], self.horizon - imgs.shape[0], axis=0)
            imgs = np.concatenate([padding, imgs], axis=0)

        out = []
        for img in imgs:
            if img.shape[:2] != (self.obs_res[1], self.obs_res[0]):
                img = cv2.resize(img, self.obs_res, interpolation=cv2.INTER_AREA)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32, copy=False)
                if float(np.nanmax(img)) > 2.0:
                    img = img / 255.0
            out.append(np.clip(img, 0.0, 1.0))
        return np.ascontiguousarray(np.stack(out, axis=0))


class NeuromekaIndyRealtimeBridge:
    """TCP backend matching the real eval_real_indy Indy path.

    This talks to a Neuromeka IndyDCP3-compatible robot/emulator through the
    existing IndyInterpolationController.  No MoveIt IK is used here.
    """

    def __init__(
        self,
        *,
        zarr_image_provider: ZarrImageEpisodeProvider,
        rc: dict[str, Any],
        robot_ip: str,
        frequency: float,
        camera_horizon: int,
        robot_horizon: int,
        gripper_horizon: int,
        camera_down_sample_steps: int,
        robot_down_sample_steps: int,
        gripper_down_sample_steps: int,
        synthetic_gripper_width: float,
        timeout_s: float,
        startup_timeout_s: float,
        vel_ratio: float,
        acc_ratio: float,
    ):
        self.zarr_image_provider = zarr_image_provider
        self.rc = rc
        self.robot_ip = str(robot_ip)
        self.frequency = float(frequency)
        self.camera_horizon = int(camera_horizon)
        self.robot_horizon = int(robot_horizon)
        self.gripper_horizon = int(gripper_horizon)
        self.camera_down_sample_steps = int(camera_down_sample_steps)
        self.robot_down_sample_steps = int(robot_down_sample_steps)
        self.gripper_down_sample_steps = int(gripper_down_sample_steps)
        self.synthetic_gripper_width = float(synthetic_gripper_width)
        self.timeout_s = float(timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.vel_ratio = float(vel_ratio)
        self.acc_ratio = float(acc_ratio)
        self.shm_manager = None
        self.robot = None
        self.PoseInterpolator = None

    def __enter__(self):
        from multiprocessing.managers import SharedMemoryManager
        from umi.common.interpolation_util import PoseInterpolator
        from umi.real_world.indy_interpolation_controller import IndyInterpolationController

        self.PoseInterpolator = PoseInterpolator
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        tcp_offset = float(self.rc.get("tcp_offset", 0.235))
        self.robot = IndyInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=self.robot_ip,
            robot_type=self.rc.get("robot_type", "indyrp2"),
            frequency=30,
            launch_timeout=max(3.0, min(self.timeout_s, self.startup_timeout_s)),
            receive_latency=float(self.rc.get("robot_obs_latency", 0.0001)),
            verbose=True,
            vel_ratio=self.vel_ratio,
            acc_ratio=self.acc_ratio,
            startup_timeout=self.startup_timeout_s,
            task_rot_is_euler=self.rc.get("indy_task_rot_is_euler", True),
            task_rot_euler_seq=self.rc.get("indy_task_rot_euler_seq", "xyz"),
            task_rot_euler_in_degrees=self.rc.get(
                "indy_task_rot_euler_in_degrees", True
            ),
            task_rot_euler_extrinsic=self.rc.get(
                "indy_task_rot_euler_extrinsic", True
            ),
            task_frame_xyz_signs=tuple(
                self.rc.get("indy_task_frame_xyz_signs", [1, 1, 1])
            ),
            tool_rot_offset_deg=tuple(
                self.rc.get("indy_tool_rot_offset_deg", [0, 0, 0])
            ),
            flange_to_tcp_pose=(0.0, 0.0, tcp_offset, 0.0, 0.0, 0.0),
        )
        self.robot.start(wait=True)
        self.wait_until_ready(timeout_s=self.timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.robot is not None:
            try:
                self.robot.stop(wait=True)
            finally:
                self.robot = None
        if self.shm_manager is not None:
            self.shm_manager.shutdown()
            self.shm_manager = None

    def wait_until_ready(self, *, timeout_s: float) -> None:
        deadline = time.time() + float(timeout_s)
        last_exc = None
        while time.time() < deadline:
            try:
                data = self.robot.get_all_state()
                if len(np.asarray(data["robot_timestamp"]).reshape(-1)) > 0:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.05)
        raise RuntimeError(
            "timed out waiting for Neuromeka emulator TCP feedback"
            + (f": {last_exc}" if last_exc is not None else "")
        )

    def _get_robot_data(self) -> dict[str, np.ndarray]:
        data = self.robot.get_all_state()
        out = {k: np.asarray(v) for k, v in data.items()}
        if len(out.get("robot_timestamp", [])) == 0:
            raise RuntimeError("Neuromeka backend has no robot state samples")
        return out

    def get_obs(self) -> dict[str, np.ndarray]:
        self.wait_until_ready(timeout_s=self.timeout_s)
        latest_t = time.time()
        dt = 1.0 / self.frequency
        camera_times = latest_t - (
            np.arange(self.camera_horizon)[::-1] * self.camera_down_sample_steps * dt
        )
        robot_times = latest_t - (
            np.arange(self.robot_horizon)[::-1] * self.robot_down_sample_steps * dt
        )
        gripper_times = latest_t - (
            np.arange(self.gripper_horizon)[::-1] * self.gripper_down_sample_steps * dt
        )

        data = self._get_robot_data()
        t = np.asarray(data["robot_timestamp"], dtype=np.float64).reshape(-1)
        tcp = np.asarray(data["ActualTCPPose"], dtype=np.float64).reshape(-1, 6)
        if len(t) >= 2 and len(tcp) >= 2:
            tcp_obs = self.PoseInterpolator(t=t, x=tcp)(robot_times)
        else:
            tcp_obs = np.repeat(tcp[-1:].copy(), len(robot_times), axis=0)

        camera_obs = self.zarr_image_provider.get_horizon()
        gripper_obs = np.full(
            (len(gripper_times), 1),
            self.synthetic_gripper_width,
            dtype=np.float32,
        )
        return {
            "camera0_rgb": camera_obs,
            "robot0_eef_pos": tcp_obs[:, :3],
            "robot0_eef_rot_axis_angle": tcp_obs[:, 3:6],
            "robot0_gripper_width": gripper_obs,
            "timestamp": camera_times,
        }

    def send_tcp7_chunk(
        self,
        tcp7_chunk: np.ndarray,
        *,
        dt: float,
        speed_scale: float,
        dry_run: bool,
    ) -> np.ndarray:
        tcp7_chunk = np.asarray(tcp7_chunk, dtype=np.float64)
        if tcp7_chunk.ndim == 1:
            tcp7_chunk = tcp7_chunk.reshape(1, -1)
        data = self._get_robot_data()
        actual_q = np.asarray(data.get("ActualQ", np.zeros((1, 7))), dtype=np.float64)
        if actual_q.ndim >= 2:
            actual_q = actual_q.reshape(-1, actual_q.shape[-1])[-1]
        else:
            actual_q = actual_q.reshape(-1)
        joint_points = np.repeat(actual_q[None, :], len(tcp7_chunk), axis=0)
        if dry_run:
            return joint_points

        start_t = time.time()
        for i, row in enumerate(tcp7_chunk):
            target_time = start_t + (i + 1) * float(dt) * float(speed_scale)
            self.robot.schedule_waypoint(row[:6], target_time=target_time)
        final_t = start_t + len(tcp7_chunk) * float(dt) * float(speed_scale)
        sleep_s = max(0.0, final_t - time.time())
        if sleep_s > 0:
            time.sleep(sleep_s)
        return joint_points


class RosIndyRealtimeBridge:
    def __init__(
        self,
        *,
        image_topic: str,
        zarr_image_provider: ZarrImageEpisodeProvider | None,
        group_name: str,
        ik_link_name: str,
        fk_link_name: str,
        frame_id: str,
        joint_names: list[str],
        ik_service: str,
        fk_service: str,
        action_name: str,
        obs_res: tuple[int, int],
        frequency: float,
        camera_horizon: int,
        robot_horizon: int,
        gripper_horizon: int,
        camera_down_sample_steps: int,
        robot_down_sample_steps: int,
        gripper_down_sample_steps: int,
        synthetic_gripper_width: float,
        preprocessed_image: bool,
        eval_image_mask: bool,
        no_mirror: bool,
        timeout_s: float,
        ik_timeout_s: float,
        history_size: int,
    ):
        import rclpy
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionFK, GetPositionIK
        from rclpy.action import ActionClient
        from sensor_msgs.msg import Image, JointState
        from trajectory_msgs.msg import JointTrajectoryPoint

        self.rclpy = rclpy
        self.Duration = Duration
        self.FollowJointTrajectory = FollowJointTrajectory
        self.GetPositionFK = GetPositionFK
        self.GetPositionIK = GetPositionIK
        self.PoseStamped = PoseStamped
        self.JointState = JointState
        self.JointTrajectoryPoint = JointTrajectoryPoint
        self.group_name = group_name
        self.ik_link_name = ik_link_name
        self.fk_link_name = fk_link_name
        self.frame_id = frame_id
        self.joint_names = joint_names
        self.obs_res = tuple(obs_res)
        self.frequency = float(frequency)
        self.camera_horizon = int(camera_horizon)
        self.robot_horizon = int(robot_horizon)
        self.gripper_horizon = int(gripper_horizon)
        self.camera_down_sample_steps = int(camera_down_sample_steps)
        self.robot_down_sample_steps = int(robot_down_sample_steps)
        self.gripper_down_sample_steps = int(gripper_down_sample_steps)
        self.synthetic_gripper_width = float(synthetic_gripper_width)
        self.preprocessed_image = bool(preprocessed_image)
        self.eval_image_mask = bool(eval_image_mask)
        self.no_mirror = bool(no_mirror)
        self.timeout_s = float(timeout_s)
        self.ik_timeout_s = float(ik_timeout_s)
        self.zarr_image_provider = zarr_image_provider
        self.image_history: deque[tuple[float, np.ndarray]] = deque(maxlen=int(history_size))
        self.tcp_history: deque[tuple[float, np.ndarray]] = deque(maxlen=int(history_size))
        self.current_joint_state = None
        self.last_ik_seed: np.ndarray | None = None

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self.node = rclpy.create_node("umi_indy_realtime_policy_sim")
        self.ik_client = self.node.create_client(GetPositionIK, ik_service)
        self.fk_client = self.node.create_client(GetPositionFK, fk_service)
        self.action_client = ActionClient(self.node, FollowJointTrajectory, action_name)
        if self.zarr_image_provider is None:
            self.node.create_subscription(Image, image_topic, self._image_cb, 10)
        self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 50)

    def __enter__(self):
        if not self.ik_client.wait_for_service(timeout_sec=self.timeout_s):
            raise RuntimeError("MoveIt IK service is not available")
        if not self.fk_client.wait_for_service(timeout_sec=self.timeout_s):
            raise RuntimeError("MoveIt FK service is not available")
        if not self.action_client.wait_for_server(timeout_sec=self.timeout_s):
            raise RuntimeError("FollowJointTrajectory action server is not available")
        self.wait_until_ready(timeout_s=self.timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()

    def _joint_state_cb(self, msg) -> None:
        self.current_joint_state = msg

    def _image_cb(self, msg) -> None:
        try:
            rgb = self._preprocess_image_msg(msg)
        except Exception as exc:
            self.node.get_logger().error(f"image preprocess failed: {exc}")
            return
        self.image_history.append((_stamp_to_float(msg.header.stamp), rgb))

    def wait_until_ready(self, *, timeout_s: float) -> None:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            image_ready = self.zarr_image_provider is not None or bool(self.image_history)
            if image_ready and self.current_joint_state is not None:
                return
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        missing = []
        if self.zarr_image_provider is None and not self.image_history:
            missing.append("image")
        if self.current_joint_state is None:
            missing.append("/joint_states")
        raise RuntimeError("timed out waiting for " + ", ".join(missing))

    def _preprocess_image_msg(self, msg) -> np.ndarray:
        rgb = self._decode_image_msg_to_rgb(msg)
        if self.eval_image_mask:
            rgb = draw_predefined_mask(
                rgb,
                color=(0, 0, 0),
                mirror=self.no_mirror,
                gripper=True,
                finger=False,
                use_aa=False,
            )

        out_w, out_h = self.obs_res
        if self.preprocessed_image:
            if (rgb.shape[1], rgb.shape[0]) != self.obs_res:
                rgb = cv2.resize(rgb, self.obs_res, interpolation=cv2.INTER_AREA)
        else:
            transform = get_image_transform(
                input_res=(rgb.shape[1], rgb.shape[0]),
                output_res=self.obs_res,
                bgr_to_rgb=False,
            )
            rgb = np.ascontiguousarray(transform(rgb))

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = rgb.astype(np.float32, copy=False)
            if float(np.nanmax(rgb)) > 2.0:
                rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.shape[:2] != (out_h, out_w):
            raise RuntimeError(f"policy image has wrong shape {rgb.shape}, expected {(out_h, out_w, 3)}")
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _decode_image_msg_to_rgb(msg) -> np.ndarray:
        encoding = str(msg.encoding).lower()
        enc_channels = {
            "rgb8": (np.uint8, 3, "rgb"),
            "bgr8": (np.uint8, 3, "bgr"),
            "rgba8": (np.uint8, 4, "rgba"),
            "bgra8": (np.uint8, 4, "bgra"),
            "mono8": (np.uint8, 1, "mono"),
            "8uc1": (np.uint8, 1, "mono"),
            "8uc3": (np.uint8, 3, "rgb"),
            "32fc1": (np.float32, 1, "mono"),
            "32fc3": (np.float32, 3, "rgb"),
        }
        if encoding not in enc_channels:
            raise ValueError(f"unsupported image encoding {msg.encoding!r}")
        dtype, channels, layout = enc_channels[encoding]
        itemsize = np.dtype(dtype).itemsize
        row_elems = int(msg.step) // itemsize
        raw = np.frombuffer(msg.data, dtype=dtype)
        rows = raw.reshape(int(msg.height), row_elems)
        if channels == 1:
            arr = rows[:, : int(msg.width)]
            rgb = np.repeat(arr[..., None], 3, axis=-1)
        else:
            arr = rows[:, : int(msg.width) * channels].reshape(
                int(msg.height), int(msg.width), channels
            )
            if layout == "bgr":
                rgb = arr[..., ::-1]
            elif layout == "rgba":
                rgb = arr[..., :3]
            elif layout == "bgra":
                rgb = arr[..., :3][..., ::-1]
            else:
                rgb = arr[..., :3]
        return np.ascontiguousarray(rgb)

    def compute_fk_tcp6(self) -> np.ndarray:
        if self.current_joint_state is None:
            raise RuntimeError("no /joint_states received")
        req = self.GetPositionFK.Request()
        req.header.frame_id = self.frame_id
        req.fk_link_names = [self.fk_link_name]
        req.robot_state.joint_state = self.current_joint_state
        future = self.fk_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout_s)
        if not future.done():
            raise TimeoutError("FK request timed out")
        resp = future.result()
        if int(resp.error_code.val) != 1:
            raise RuntimeError(f"FK failed with MoveIt error code {resp.error_code.val}")
        if not resp.pose_stamped:
            raise RuntimeError("FK response did not contain a pose")
        pose = resp.pose_stamped[0].pose
        quat = np.asarray(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=np.float64,
        )
        rotvec = st.Rotation.from_quat(quat).as_rotvec()
        return np.asarray(
            [pose.position.x, pose.position.y, pose.position.z, *rotvec],
            dtype=np.float64,
        )

    def get_obs(self) -> dict[str, np.ndarray]:
        self.wait_until_ready(timeout_s=self.timeout_s)
        for _ in range(3):
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

        tcp6 = self.compute_fk_tcp6()
        latest_t = (
            time.time()
            if self.zarr_image_provider is not None
            else self.image_history[-1][0]
        )
        self.tcp_history.append((latest_t, tcp6))

        dt = 1.0 / self.frequency
        camera_times = latest_t - (
            np.arange(self.camera_horizon)[::-1] * self.camera_down_sample_steps * dt
        )
        robot_times = latest_t - (
            np.arange(self.robot_horizon)[::-1] * self.robot_down_sample_steps * dt
        )
        gripper_times = latest_t - (
            np.arange(self.gripper_horizon)[::-1] * self.gripper_down_sample_steps * dt
        )

        if self.zarr_image_provider is not None:
            camera_obs = self.zarr_image_provider.get_horizon()
        else:
            camera_obs = np.stack(
                [_nearest_history(self.image_history, float(t)) for t in camera_times],
                axis=0,
            )
        tcp_obs = np.stack(
            [_nearest_history(self.tcp_history, float(t)) for t in robot_times],
            axis=0,
        )
        gripper_obs = np.full(
            (len(gripper_times), 1),
            self.synthetic_gripper_width,
            dtype=np.float32,
        )

        return {
            "camera0_rgb": camera_obs,
            "robot0_eef_pos": tcp_obs[:, :3],
            "robot0_eef_rot_axis_angle": tcp_obs[:, 3:6],
            "robot0_gripper_width": gripper_obs,
            "timestamp": camera_times,
        }

    def _current_seed_positions(self) -> np.ndarray | None:
        if self.current_joint_state is None:
            return self.last_ik_seed
        by_name = {
            name: pos
            for name, pos in zip(self.current_joint_state.name, self.current_joint_state.position)
        }
        if all(name in by_name for name in self.joint_names):
            return np.asarray([by_name[name] for name in self.joint_names], dtype=np.float64)
        return self.last_ik_seed

    def compute_ik(self, tcp6: np.ndarray, seed_positions: np.ndarray | None) -> np.ndarray:
        req = self.GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = _duration_msg(self.Duration, self.ik_timeout_s)

        pose = self.PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(tcp6[0])
        pose.pose.position.y = float(tcp6[1])
        pose.pose.position.z = float(tcp6[2])
        quat = _rotvec_to_quat_xyzw(tcp6[3:6])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        req.ik_request.pose_stamped = pose

        if seed_positions is not None:
            seed = self.JointState()
            seed.name = list(self.joint_names)
            seed.position = [float(x) for x in seed_positions]
            req.ik_request.robot_state.joint_state = seed
        elif self.current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.current_joint_state

        future = self.ik_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout_s)
        if not future.done():
            raise TimeoutError("IK request timed out")
        resp = future.result()
        if int(resp.error_code.val) != 1:
            raise RuntimeError(f"IK failed with MoveIt error code {resp.error_code.val}")

        js = resp.solution.joint_state
        by_name = {name: pos for name, pos in zip(js.name, js.position)}
        missing = [name for name in self.joint_names if name not in by_name]
        if missing:
            raise RuntimeError(f"IK solution missing joints: {missing}")
        q = np.asarray([by_name[name] for name in self.joint_names], dtype=np.float64)
        self.last_ik_seed = q.copy()
        return q

    def send_tcp7_chunk(
        self,
        tcp7_chunk: np.ndarray,
        *,
        dt: float,
        speed_scale: float,
        dry_run: bool,
    ) -> np.ndarray:
        tcp7_chunk = np.asarray(tcp7_chunk, dtype=np.float64)
        if tcp7_chunk.ndim == 1:
            tcp7_chunk = tcp7_chunk.reshape(1, -1)
        seed = self._current_seed_positions()
        joint_points = []
        for row in tcp7_chunk:
            seed = self.compute_ik(row[:6], seed)
            joint_points.append(seed.copy())
        joint_points_np = np.asarray(joint_points, dtype=np.float64)
        if dry_run:
            return joint_points_np

        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for i, q in enumerate(joint_points_np):
            point = self.JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [0.0] * len(q)
            point.time_from_start = _duration_msg(
                self.Duration,
                max(0.02, (i + 1) * float(dt) * float(speed_scale)),
            )
            goal.trajectory.points.append(point)

        future = self.action_client.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout_s)
        if not future.done():
            raise TimeoutError("sending trajectory goal timed out")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("trajectory goal rejected")

        result_future = handle.get_result_async()
        wait_s = max(1.0, len(joint_points_np) * float(dt) * float(speed_scale) + self.timeout_s)
        self.rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=wait_s)
        if not result_future.done():
            raise TimeoutError("trajectory execution timed out")
        result = result_future.result().result
        if int(result.error_code) != 0:
            msg = result.error_string or f"error_code={result.error_code}"
            raise RuntimeError(f"trajectory execution failed: {msg}")
        return joint_points_np


def _shape_horizon(cfg: Any, key: str, fallback: int = 2) -> int:
    meta = OmegaConf.to_container(cfg.task.shape_meta.obs[key], resolve=True)
    return int(meta.get("horizon", fallback))


def _shape_downsample(cfg: Any, key: str, fallback: int = 1) -> int:
    meta = OmegaConf.to_container(cfg.task.shape_meta.obs[key], resolve=True)
    return int(meta.get("down_sample_steps", fallback))


def _flatten(prefix: str, values: np.ndarray, n: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{i}": float(arr[i]) for i in range(min(n, len(arr)))}


def run(args: argparse.Namespace) -> None:
    robot_config_path = _resolve(args.robot_config)
    rc = _load_robot_config(robot_config_path)
    _set_robot_dataset_transform(rc.get("indy_robot_from_dataset_transform", None))

    bundle = _load_policy(args, rc)
    obs_res = get_real_obs_resolution(bundle.cfg.task.shape_meta)
    tcp_delta_scale_vec = _parse_tcp_delta_scales(args.tcp_delta_scales)
    dt = 1.0 / float(args.frequency)
    output_dir = _resolve(args.output, must_exist=False)
    run_dir = output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"

    print("mode: realtime closed loop (image source -> policy -> backend -> next image)")
    print("backend:", args.backend)
    print("checkpoint:", _resolve(args.input))
    print("robot_config:", robot_config_path)
    print("device:", bundle.device)
    print("obs_pose_repr:", bundle.obs_pose_repr)
    print("action_pose_repr:", bundle.action_pose_repr)
    print("obs_res:", obs_res)
    print("steps_per_inference:", args.steps_per_inference)
    print("image_source:", args.image_source)
    if args.image_source == "ros":
        print("image_topic:", args.image_topic)
    print("log:", log_path)
    if args.dry_run or args.plan_only:
        print("dry/plan mode: target is computed but robot/emulator command is not sent.")

    joint_names = [x.strip() for x in str(args.joint_names).split(",") if x.strip()]
    zarr_image_provider = None
    if args.image_source == "zarr":
        zarr_image_provider = ZarrImageEpisodeProvider(
            zarr_path=_resolve(args.zarr),
            episode=args.episode,
            obs_res=obs_res,
            horizon=_shape_horizon(bundle.cfg, "camera0_rgb"),
            down_sample_steps=_shape_downsample(bundle.cfg, "camera0_rgb"),
        )
        print(
            "zarr image replay:",
            f"path={zarr_image_provider.zarr_path}",
            f"episode={zarr_image_provider.episode}",
            f"frames=[{zarr_image_provider.start_idx}, {zarr_image_provider.end_idx})",
        )

    if args.backend == "neuromeka":
        if zarr_image_provider is None:
            raise ValueError("--backend neuromeka expects --image_source zarr")
        robot_ip = args.robot_ip or rc.get("robot_ip")
        if not robot_ip:
            raise ValueError("robot_ip is required for --backend neuromeka")
        print("neuromeka/emulator robot_ip:", robot_ip)
        bridge_cm = NeuromekaIndyRealtimeBridge(
            zarr_image_provider=zarr_image_provider,
            rc=rc,
            robot_ip=robot_ip,
            frequency=args.frequency,
            camera_horizon=_shape_horizon(bundle.cfg, "camera0_rgb"),
            robot_horizon=_shape_horizon(bundle.cfg, "robot0_eef_pos"),
            gripper_horizon=_shape_horizon(bundle.cfg, "robot0_gripper_width"),
            camera_down_sample_steps=_shape_downsample(bundle.cfg, "camera0_rgb"),
            robot_down_sample_steps=_shape_downsample(bundle.cfg, "robot0_eef_pos"),
            gripper_down_sample_steps=_shape_downsample(bundle.cfg, "robot0_gripper_width"),
            synthetic_gripper_width=args.synthetic_gripper_width,
            timeout_s=args.timeout_s,
            startup_timeout_s=args.neuromeka_startup_timeout_s,
            vel_ratio=args.neuromeka_vel_ratio,
            acc_ratio=args.neuromeka_acc_ratio,
        )
    else:
        bridge_cm = RosIndyRealtimeBridge(
            image_topic=args.image_topic,
            zarr_image_provider=zarr_image_provider,
            group_name=args.group_name,
            ik_link_name=args.ik_link_name,
            fk_link_name=args.fk_link_name,
            frame_id=args.frame_id,
            joint_names=joint_names,
            ik_service=args.ik_service,
            fk_service=args.fk_service,
            action_name=args.action_name,
            obs_res=obs_res,
            frequency=args.frequency,
            camera_horizon=_shape_horizon(bundle.cfg, "camera0_rgb"),
            robot_horizon=_shape_horizon(bundle.cfg, "robot0_eef_pos"),
            gripper_horizon=_shape_horizon(bundle.cfg, "robot0_gripper_width"),
            camera_down_sample_steps=_shape_downsample(bundle.cfg, "camera0_rgb"),
            robot_down_sample_steps=_shape_downsample(bundle.cfg, "robot0_eef_pos"),
            gripper_down_sample_steps=_shape_downsample(bundle.cfg, "robot0_gripper_width"),
            synthetic_gripper_width=args.synthetic_gripper_width,
            preprocessed_image=args.preprocessed_image,
            eval_image_mask=args.eval_image_mask,
            no_mirror=args.no_mirror,
            timeout_s=args.timeout_s,
            ik_timeout_s=args.ik_timeout_s,
            history_size=args.history_size,
        )

    try:
        with bridge_cm as bridge, open(log_path, "w", newline="") as log_f:
            obs = bridge.get_obs()
            episode_start_pose = [
                np.concatenate(
                    [obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"]],
                    axis=-1,
                )[-1]
            ]
            episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(
                episode_start_pose
            )
    
            print("warming up policy...")
            bundle.policy.reset()
            with torch.no_grad():
                obs_for_model = _apply_slam_frame_fix_to_obs(obs, 1)
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs_for_model,
                    shape_meta=bundle.cfg.task.shape_meta,
                    obs_pose_repr=bundle.obs_pose_repr,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose_for_model,
                )
                _check_policy_inputs_finite(obs_dict_np, "[warmup]")
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(bundle.device),
                )
                _ = bundle.policy.predict_action(obs_dict)
            print("started realtime policy loop.")
    
            writer = csv.DictWriter(log_f, fieldnames=[])
            header_written = False
            policy_iter = 0
            start_wall = time.time()
            while True:
                if args.max_policy_iters is not None and policy_iter >= int(args.max_policy_iters):
                    print(f"max_policy_iters={args.max_policy_iters} reached.")
                    break
                if args.max_duration_s is not None and (time.time() - start_wall) >= float(args.max_duration_s):
                    print(f"max_duration_s={args.max_duration_s} reached.")
                    break
    
                obs = bridge.get_obs()
                with torch.no_grad():
                    infer_t0 = time.time()
                    obs_for_model = _apply_slam_frame_fix_to_obs(obs, 1)
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=obs_for_model,
                        shape_meta=bundle.cfg.task.shape_meta,
                        obs_pose_repr=bundle.obs_pose_repr,
                        tx_robot1_robot0=None,
                        episode_start_pose=episode_start_pose_for_model,
                    )
                    _check_policy_inputs_finite(obs_dict_np, f"[iter={policy_iter}]")
                    obs_dict = dict_apply(
                        obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(bundle.device),
                    )
                    result = bundle.policy.predict_action(obs_dict)
                    raw_action = result["action_pred"][0].detach().to("cpu").numpy()
                    infer_latency = time.time() - infer_t0
    
                _check_finite_array(f"[iter={policy_iter}] raw_action", raw_action)
                raw_action = _apply_slam_frame_fix(raw_action, 1)
                action_dataset = _decode_real_umi_action_checked(
                    raw_action,
                    obs_for_model,
                    bundle.action_pose_repr,
                    f"[iter={policy_iter} dataset]",
                )
                action_robot = _transform_tcp7_action(
                    action_dataset,
                    eval_indy._ROBOT_FROM_DATASET_T,
                    1,
                )
                action_robot = _apply_policy_tcp7_rot_roundtrip(
                    action_robot,
                    enabled=bundle.policy_rot_rt,
                    euler_seq=bundle.policy_rot_seq,
                    euler_extrinsic=bundle.policy_rot_ext,
                    n_robots=1,
                )
    
                n_exec = min(int(args.steps_per_inference), len(action_robot))
                target_chunk = np.asarray(action_robot[:n_exec], dtype=np.float64)
                if (
                    tcp_delta_scale_vec is not None
                    or args.action_scale != 1.0
                    or args.freeze_rotation
                ):
                    target_chunk = _limit_policy_waypoints(
                        target_chunk,
                        obs,
                        n_robots=1,
                        tcp_delta_scales=tcp_delta_scale_vec,
                        action_scale=args.action_scale,
                        freeze_rotation=args.freeze_rotation,
                        freeze_rotation_ref_pose=episode_start_pose,
                    )
    
                if args.print_policy_output:
                    _print_policy_action_debug(
                        f"[policy iter={policy_iter}]",
                        raw_action,
                        action_robot,
                        submitted=target_chunk,
                    )
                if args.print_motion_debug:
                    action_timestamps = time.time() + np.arange(1, len(target_chunk) + 1) * dt
                    _print_motion_debug(
                        f"[motion iter={policy_iter}]",
                        obs,
                        target_chunk,
                        timestamps=action_timestamps,
                        n_robots=1,
                    )
    
                exec_t0 = time.time()
                joint_points = bridge.send_tcp7_chunk(
                    target_chunk,
                    dt=dt,
                    speed_scale=args.speed_scale,
                    dry_run=(args.dry_run or args.plan_only),
                )
                exec_latency = time.time() - exec_t0
    
                obs_tcp6 = np.concatenate(
                    [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]]
                )
                sent_tcp7 = np.asarray(target_chunk[0], dtype=np.float64)
                row: dict[str, Any] = {
                    "policy_iter": int(policy_iter),
                    "wall_time": float(time.time()),
                    "infer_latency_s": float(infer_latency),
                    "exec_latency_s": float(exec_latency),
                    "n_exec": int(len(target_chunk)),
                }
                row.update(_flatten("obs_tcp6", obs_tcp6, 6))
                row.update(_flatten("sent_tcp7", sent_tcp7, 7))
                row.update(_flatten("delta_pos", sent_tcp7[:3] - obs_tcp6[:3], 3))
                row.update(_flatten("joint0", joint_points[0], len(joint_names)))
                if not header_written:
                    writer = csv.DictWriter(log_f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)
                log_f.flush()
    
                print(
                    f"iter={policy_iter:04d} n_exec={len(target_chunk)} "
                    f"infer={infer_latency:.3f}s exec={exec_latency:.3f}s "
                    f"dpos={np.linalg.norm(sent_tcp7[:3] - obs_tcp6[:3]):.4f}m"
                )
                if zarr_image_provider is not None:
                    zarr_image_provider.advance(len(target_chunk))
                    if zarr_image_provider.is_done:
                        print("zarr episode reached the end.")
                        break
                policy_iter += 1
    finally:
        if zarr_image_provider is not None:
            zarr_image_provider.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run UMI Indy policy closed-loop on ROS2/Gazebo images and joints."
    )
    p.add_argument("-i", "--input", default=str(DEFAULT_CKPT), help="Checkpoint path")
    p.add_argument("-rc", "--robot_config", default=str(DEFAULT_ROBOT_CONFIG))
    p.add_argument("-d", "--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--backend",
        choices=["neuromeka", "moveit"],
        default="moveit",
        help=(
            "moveit uses /compute_ik + FJT for indy_gazebo. "
            "neuromeka only works if an IndyDCP3-compatible controller/emulator is running."
        ),
    )
    p.add_argument("--robot_ip", default=None, help="Neuromeka emulator/controller IP. Defaults to robot_config robot_ip.")
    p.add_argument("--neuromeka_vel_ratio", type=float, default=0.1)
    p.add_argument("--neuromeka_acc_ratio", type=float, default=0.5)
    p.add_argument("--neuromeka_startup_timeout_s", type=float, default=15.0)
    p.add_argument("--image_source", choices=["zarr", "ros"], default="zarr")
    p.add_argument("--zarr", default=str(DEFAULT_DATASET), help="Zarr used for camera0_rgb when --image_source=zarr.")
    p.add_argument("--episode", type=int, default=0, help="Zarr episode index for --image_source=zarr.")
    p.add_argument("--image_topic", default="/camera/image_raw")
    p.add_argument("--preprocessed_image", action="store_true", help="Image topic is already RGB policy size.")
    p.add_argument("--eval_image_mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_mirror", action="store_true")
    p.add_argument("--group_name", default="indy_manipulator")
    p.add_argument("--ik_link_name", default="tcp")
    p.add_argument("--fk_link_name", default="tcp")
    p.add_argument("--frame_id", default="link0")
    p.add_argument("--joint_names", default="joint0,joint1,joint2,joint3,joint4,joint5,joint6")
    p.add_argument("--ik_service", default="/compute_ik")
    p.add_argument("--fk_service", default="/compute_fk")
    p.add_argument("--action_name", default="/joint_trajectory_controller/follow_joint_trajectory")
    p.add_argument("--frequency", type=float, default=19.98)
    p.add_argument("--steps_per_inference", "-si", type=int, default=1)
    p.add_argument("--policy_num_inference_steps", type=int, default=16)
    p.add_argument("--synthetic_gripper_width", type=float, default=float(SYNTHETIC_GRIPPER_WIDTH))
    p.add_argument("--speed_scale", type=float, default=1.0, help=">1 slows each executed chunk.")
    p.add_argument("--timeout_s", type=float, default=20.0)
    p.add_argument("--ik_timeout_s", type=float, default=0.2)
    p.add_argument("--history_size", type=int, default=240)
    p.add_argument("--max_policy_iters", "-mpi", type=int, default=None)
    p.add_argument("--max_duration_s", type=float, default=None)
    p.add_argument("--dry_run", action="store_true", help="Solve IK but do not send trajectory goals.")
    p.add_argument("--plan_only", action="store_true", help="Alias-style safety mode: no trajectory goals.")
    p.add_argument("--print_policy_output", action="store_true")
    p.add_argument("--print_motion_debug", action="store_true")
    p.add_argument("--tcp_delta_scales", default=None, help="Scale xyz deltas, e.g. 1,0,0.")
    p.add_argument("--action_scale", type=float, default=1.0)
    p.add_argument("--freeze_rotation", action="store_true")
    p.add_argument("--disable_eval_image_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
