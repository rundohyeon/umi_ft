from __future__ import annotations

"""
Usage:
(umi): python3 scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data
python3 eval_real_indy_rg2.py --robot_config example/eval_robots_config_indy_rg2.yaml -i path/to.ckpt -o data/eval_rg2 --print_policy_output

Offline ckpt pose_repr / dataset z (no robot):
python3 scripts/indy_umi/inspect_ckpt_pose_eval.py -i path/to/latest.ckpt --zarr auto --stride 20 --episode_z 8

Live vs train z / raw model scale:
python3 eval_real_indy.py ... --pose_eval_audit --dataset_zarr auto

Current TCP vs next waypoint each policy step:
python3 eval_real_indy.py ... --print_motion_debug

Controller-connected planning check (no waypoint submission from this script):
python3 eval_real_indy.py ... --print_motion_debug --plan_only

One small step along +X only, then auto-stop:
python3 eval_real_indy.py ... -si 1 -mpi 1 --freeze_rotation --action_scale 0.2 --tcp_delta_scales 1,0,0

Overlay TCP / next waypoint on camera window:
python3 eval_real_indy.py ... --vis_pose

Print tensors fed to predict_action (vs raw env TCP):
python3 eval_real_indy.py ... --print_model_input

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the OpenCV window so it stays responsive. Motion keys (a/d/…) use a one-shot
latch so cv2.pollKey() stickiness does not repeat the same step every frame.
Press "C" to start evaluation (hand control over to policy).
Press "Esc" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

"""
"""

# %%
import csv
import atexit
import collections
import hashlib
import json
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
from diffusion_policy.common.dual_ft_contract import (
    inspect_dual_ft_checkpoint_payload as _inspect_dual_ft_checkpoint_payload,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics_file,
    FisheyeRectConverter
)
from umi.common.pose_util import rot6d_to_mat, mat_to_rot6d
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from umi.real_world.umi_env import UmiEnv
from umi.real_world.rg2ft_obs import (
    FTObservationStaleError,
    prepare_rg2ft_policy_obs,
)
from umi.real_world.valve_state_context import (
    RGBForceValveContextRuntime,
    ValveStateContextRuntime,
    ValveStateContextRuntimeV2,
    sha256_file as _sha256_file,
)
from diffusion_policy.common.valve_context_contract import (
    VALVE_CONTEXT_V1_SCHEMA,
    VALVE_CONTEXT_V2_SCHEMA,
    valve_context_spec,
)
from umi.real_world.grasp_force_width_feedback import (
    GraspForceWidthFeedbackConfig,
    GraspForceWidthFeedbackController,
)
from umi.real_world.rg2ft_startup_bias import (
    FTStartupBiasConfig,
    acquire_startup_bias,
    startup_residual_after_software_tare,
)
from umi.real_world.dual_ft_policy_safety import (
    FTSafetyConfig,
    PolicyMotionSafetyConfig,
    PolicySafetyError,
    read_and_validate_latest_ft,
    validate_policy_waypoints,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
_POSE_HUD_MAX_XY_RANGE_M = 0.35
# The 0709 training replay was generated without ``--out_fov``. Keep the
# live policy image on the same projection by default; ``--sim_fov VALUE``
# remains available for checkpoints trained with fisheye rectification.
_DEFAULT_SIM_FOV = None
_DEFAULT_CAMERA_INTRINSICS = str(
    _PROJECT_ROOT.joinpath(
        "gopro13_1080p_calib_export", "gopro13_1920x1080.yaml")
)
_DEFAULT_VALVE_CLASSIFIER_CHECKPOINT = str(
    _PROJECT_ROOT / "valve_state_classifier_v4" / "model" / "final.pt"
)
_SYNTHETIC_GRIPPER_WIDTH = np.float32(0.05651384)
_DEFAULT_GRIPPER_CALIB_ZARR = str(
    _PROJECT_ROOT.joinpath(
        "artifacts", "0709_robot_tcp", "dataset_robot_tcp.zarr.zip")
)
_SAVED_START_POSE_PATH = _PROJECT_ROOT.joinpath(
    "data", "saved_start_pose.yaml")
_DEFAULT_DYNAMIXEL_GRIPPER_CONFIG = str(
    _PROJECT_ROOT.joinpath("scripts", "waypoints", "rulebase_indy.yaml")
)
_DEFAULT_ARUCO_CONFIG = str(
    _PROJECT_ROOT.joinpath(
        "slam_pipeline_latest", "calibration", "aruco_config.yaml")
)
_RG2FT_WORKSPACE_TARGET = (
    "diffusion_policy.workspace.train_diffusion_unet_image_rg2ft_workspace."
    "TrainDiffusionUnetImageRg2ftWorkspace"
)


def _get_eval_workspace_class(target: str):
    """Resolve the training workspace needed only to restore model weights.

    The RG2 training checkpoint names a workspace wrapper that is not present
    in this deployment copy.  Its policy target and state dict are fully
    embedded in the checkpoint; the standard image workspace constructs the
    same model/EMA/optimizer objects required by BaseWorkspace.load_payload.
    Keep this compatibility mapping narrow so unrelated checkpoints still use
    their declared workspace class.
    """
    target = str(target)
    if target == _RG2FT_WORKSPACE_TARGET:
        from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
            TrainDiffusionUnetImageWorkspace,
        )

        print(
            "RG2 checkpoint workspace compatibility: "
            "TrainDiffusionUnetImageRg2ftWorkspace -> "
            "TrainDiffusionUnetImageWorkspace"
        )
        return TrainDiffusionUnetImageWorkspace
    return hydra.utils.get_class(target)


def inspect_dual_ft_checkpoint_payload(payload: dict) -> dict:
    """Compatibility export of the hardware-free checkpoint inspector."""

    return _inspect_dual_ft_checkpoint_payload(payload)


class _TerminalKeyPoller:
    """Non-blocking single-key input for Docker terminals."""

    def __init__(self):
        self.fd = None
        self.old_attrs = None
        self.enabled = False

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            self.fd = sys.stdin.fileno()
            self.old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
            return True
        except Exception as exc:
            print(f"terminal keyboard fallback disabled: {exc}")
            self.close()
            return False

    def close(self) -> None:
        if self.enabled and self.fd is not None and self.old_attrs is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
            except Exception:
                pass
        self.enabled = False

    def poll(self) -> int | None:
        if not self.enabled or self.fd is None:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return None
            ch = os.read(self.fd, 1)
        except Exception:
            return None
        if not ch:
            return None
        if ch == b"\x1b":
            # Ignore arrow/function-key escape sequences, but keep bare Esc.
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    os.read(self.fd, 8)
                    return None
            except Exception:
                return None
            return 27
        return ch[0]


def _poll_control_key(terminal_key_poller: _TerminalKeyPoller | None = None) -> int:
    cv_key = cv2.pollKey()
    if cv_key >= 0:
        key = cv_key & 0xFF
        if key != 255:
            return key
    if terminal_key_poller is not None:
        key = terminal_key_poller.poll()
        if key is not None:
            return key
    return -1


def _put_text_hud(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        img,
        text,
        org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        lineType=cv2.LINE_AA,
        thickness=3,
        color=(0, 0, 0),
    )
    cv2.putText(
        img,
        text,
        org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        lineType=cv2.LINE_AA,
        thickness=1,
        color=color,
    )


def _tcp6_from_obs(obs) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64),
            np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64),
        ]
    )


def _sanitize_gripper_width(width: float, fallback: float, *, tag: str) -> np.float32:
    width_f = float(width)
    if np.isfinite(width_f):
        return np.float32(width_f)
    fallback_f = float(fallback)
    if not np.isfinite(fallback_f):
        fallback_f = float(_SYNTHETIC_GRIPPER_WIDTH)
    print(
        f"[WARN] {tag}: non-finite gripper width {width_f}; "
        f"using fallback {fallback_f:.9f} m."
    )
    return np.float32(fallback_f)


def _with_synthetic_gripper_width(
    obs: dict,
    width: float,
    *,
    fallback: float = _SYNTHETIC_GRIPPER_WIDTH,
) -> dict:
    out = dict(obs)
    grip = np.asarray(out["robot0_gripper_width"])
    dtype = grip.dtype if np.issubdtype(grip.dtype, np.floating) else np.float32
    safe_width = _sanitize_gripper_width(
        width, fallback, tag="synthetic robot0_gripper_width"
    )
    out["robot0_gripper_width"] = np.full(grip.shape, safe_width, dtype=dtype)
    return out


def _disable_policy_image_transforms(policy) -> list[str]:
    obs_encoder = getattr(policy, "obs_encoder", None)
    if obs_encoder is None:
        return []
    candidates = [("obs_encoder", obs_encoder)]
    nested = getattr(obs_encoder, "vision_pose_encoder", None)
    if nested is not None:
        candidates.append(("obs_encoder.vision_pose_encoder", nested))
    disabled = []
    seen_maps = set()
    for prefix, encoder in candidates:
        transform_map = getattr(encoder, "key_transform_map", None)
        if transform_map is None or id(transform_map) in seen_maps:
            continue
        seen_maps.add(id(transform_map))
        for key in list(transform_map.keys()):
            transform_map[key] = torch.nn.Identity()
            disabled.append(f"{prefix}.{key}")
    return disabled


def _resolve_path_for_eval(path: str, *, must_exist: bool = True) -> pathlib.Path:
    raw = pathlib.Path(os.path.expanduser(str(path)))
    candidates = [raw]
    if not raw.is_absolute():
        script_dir = pathlib.Path(__file__).resolve().parent
        candidates = [
            pathlib.Path.cwd().joinpath(raw),
            script_dir.joinpath(raw),
            script_dir.parent.joinpath(raw),
        ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    if must_exist:
        raise FileNotFoundError(str(candidates[0]))
    return candidates[0].resolve()


def _load_gripper_width_range_from_zarr(zarr_path: str) -> tuple[float, float, pathlib.Path]:
    import zarr

    resolved = _resolve_path_for_eval(zarr_path)
    store = None
    try:
        if resolved.name.endswith(".zarr.zip") or resolved.suffix == ".zip":
            store = zarr.ZipStore(str(resolved), mode="r")
            root = zarr.open_group(store=store, mode="r")
        else:
            root = zarr.open_group(str(resolved), mode="r")
        if "robot0_gripper_width" not in root["data"]:
            raise KeyError(
                f"robot0_gripper_width not found in {resolved}; "
                f"available keys: {list(root['data'].keys())}"
            )
        width = np.asarray(root["data"]["robot0_gripper_width"][:], dtype=np.float64)
        width = width.reshape(-1)
        finite_width = width[np.isfinite(width)]
        if finite_width.size == 0:
            raise ValueError(f"robot0_gripper_width has no finite values in {resolved}")
        return float(np.min(finite_width)), float(np.max(finite_width)), resolved
    finally:
        if store is not None:
            store.close()


def _load_rulebase_gripper_config(config_path: str) -> tuple[dict, pathlib.Path]:
    resolved = _resolve_path_for_eval(config_path)
    with open(resolved, "r") as f:
        data = yaml.safe_load(f) or {}
    gripper = dict(data.get("gripper") or {})
    missing = [
        key
        for key in ("open_position", "close_position")
        if key not in gripper
    ]
    if missing:
        raise KeyError(f"{resolved} gripper config missing: {missing}")
    return gripper, resolved


def _gripper_width_to_tick(
    width_m: float,
    *,
    width_min_m: float,
    width_max_m: float,
    close_tick: int,
    open_tick: int,
) -> tuple[float, int]:
    span = max(float(width_max_m) - float(width_min_m), 1e-9)
    clipped = float(np.clip(width_m, width_min_m, width_max_m))
    ratio = (clipped - float(width_min_m)) / span
    tick = int(round(float(close_tick) + ratio * (float(open_tick) - float(close_tick))))
    return clipped, tick


def _gripper_tick_to_width(
    tick: int | float,
    *,
    width_min_m: float,
    width_max_m: float,
    close_tick: int,
    open_tick: int,
) -> tuple[int, float]:
    lo = min(int(close_tick), int(open_tick))
    hi = max(int(close_tick), int(open_tick))
    clipped_tick = int(round(np.clip(float(tick), lo, hi)))
    tick_span = max(float(open_tick) - float(close_tick), 1e-9)
    ratio = (float(clipped_tick) - float(close_tick)) / tick_span
    width = float(width_min_m) + ratio * (float(width_max_m) - float(width_min_m))
    width = float(np.clip(width, width_min_m, width_max_m))
    return clipped_tick, width


class _DirectDynamixelGripper:
    """Direct fallback for Indy eval when UmiEnv is running with no gripper."""

    def __init__(
        self,
        *,
        yaml_config: dict,
        yaml_path: pathlib.Path,
        width_min_m: float,
        width_max_m: float,
        zarr_path: pathlib.Path,
        print_debug: bool = False,
    ):
        self.yaml_config = yaml_config
        self.yaml_path = yaml_path
        self.width_min_m = float(width_min_m)
        self.width_max_m = float(width_max_m)
        self.zarr_path = zarr_path
        self.print_debug = bool(print_debug)
        self.dxl_id = int(yaml_config.get("id", yaml_config.get("dynamixel_id", 1)))
        self.open_tick = int(yaml_config["open_position"])
        self.close_tick = int(yaml_config["close_position"])
        self.keep_torque = bool(yaml_config.get("keep_torque", True))
        self.controller = None
        self.last_width_m: float | None = None
        self.last_tick: int | None = None
        self.present_tick: int | None = None
        self.initial_width_m: float = self.width_max_m

    def __enter__(self):
        from umi.real_world.dynamixel_controller import (
            PROTOCOL_2_0,
            DynamixelConfig,
            DynamixelPositionController,
        )

        cfg = DynamixelConfig(
            port=str(self.yaml_config.get("port", "/dev/ttyUSB0")),
            baudrate=int(self.yaml_config.get("baudrate", 57600)),
            protocol_version=float(self.yaml_config.get("protocol_version", PROTOCOL_2_0)),
            dxl_ids=(self.dxl_id,),
            profile_velocity=int(self.yaml_config.get("profile_velocity", 30)),
            profile_acceleration=int(self.yaml_config.get("profile_acceleration", 15)),
            current_limit=self.yaml_config.get("current_limit"),
            pwm_limit=self.yaml_config.get("pwm_limit"),
        )
        self.controller = DynamixelPositionController(cfg)
        self.controller.connect()
        self.controller.configure_position_mode()
        self.controller.enable_torque()
        try:
            present = self.controller.get_present_position(self.dxl_id)
        except Exception:
            present = None
        self.present_tick = present
        if present is not None:
            clipped_tick, initial_width_m = _gripper_tick_to_width(
                present,
                width_min_m=self.width_min_m,
                width_max_m=self.width_max_m,
                close_tick=self.close_tick,
                open_tick=self.open_tick,
            )
            self.last_tick = clipped_tick
            self.last_width_m = initial_width_m
            self.initial_width_m = initial_width_m
        print(
            "[direct_gripper] connected Dynamixel "
            f"id={self.dxl_id} port={cfg.port} present={present}"
        )
        print(
            "[direct_gripper] model width calibration: "
            f"{self.width_min_m:.9f}..{self.width_max_m:.9f} m "
            f"({self.zarr_path}) -> ticks close/open "
            f"{self.close_tick}/{self.open_tick} ({self.yaml_path})"
        )
        print(
            "[direct_gripper] initial model input width from present tick: "
            f"{self.initial_width_m:.9f} m"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.controller is not None:
            self.controller.disconnect(disable_torque=not self.keep_torque)
            self.controller = None

    def command_width(self, width_m: float, *, force: bool = False) -> tuple[float, int]:
        clipped, tick = _gripper_width_to_tick(
            width_m,
            width_min_m=self.width_min_m,
            width_max_m=self.width_max_m,
            close_tick=self.close_tick,
            open_tick=self.open_tick,
        )
        if self.controller is None:
            raise RuntimeError("direct Dynamixel gripper is not connected")
        if force or tick != self.last_tick:
            self.controller.set_goal_position(self.dxl_id, tick)
        self.last_width_m = clipped
        self.last_tick = tick
        if self.print_debug:
            print(f"[direct_gripper] model_width={float(width_m):.5f} m -> tick={tick}")
        return clipped, tick


def _draw_xy_localization_panel(
    img: np.ndarray,
    episode_origin_tcp6: np.ndarray,
    cur_tcp6: np.ndarray,
    target_tcp6: np.ndarray | None = None,
    *,
    panel_size: int = 150,
    margin: int = 12,
) -> None:
    """Top-down X-Y map relative to episode-start TCP (robot base frame, meters)."""
    h, w = img.shape[:2]
    x0 = w - panel_size - margin
    y0 = h - panel_size - margin - 24
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_size, y0 + panel_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    origin_xy = np.asarray(episode_origin_tcp6[:2], dtype=np.float64)
    center = np.array([x0 + panel_size // 2, y0 + panel_size // 2], dtype=np.float64)
    half = panel_size // 2 - 8
    max_range = _POSE_HUD_MAX_XY_RANGE_M

    def to_px(xy: np.ndarray) -> tuple[int, int]:
        delta = (np.asarray(xy[:2], dtype=np.float64) - origin_xy) / max_range * half
        px = center + np.array([delta[0], -delta[1]])
        px[0] = np.clip(px[0], x0 + 6, x0 + panel_size - 6)
        px[1] = np.clip(px[1], y0 + 6, y0 + panel_size - 6)
        return int(px[0]), int(px[1])

    cv2.rectangle(img, (x0, y0), (x0 + panel_size, y0 + panel_size), (180, 180, 180), 1)
    cv2.drawMarker(img, to_px(origin_xy), (160, 160, 160), cv2.MARKER_CROSS, 10, 1)
    cv2.circle(img, to_px(cur_tcp6), 5, (0, 255, 0), -1)
    if target_tcp6 is not None:
        cv2.circle(img, to_px(target_tcp6), 5, (0, 0, 255), -1)
        cv2.line(img, to_px(cur_tcp6), to_px(target_tcp6), (0, 180, 255), 1, cv2.LINE_AA)
    _put_text_hud(img, "XY vs episode start", (x0, y0 - 6), scale=0.42)
    _put_text_hud(img, "+ start  o cur  o next", (x0, y0 + panel_size + 4), scale=0.38)


def _overlay_pose_vis(
    vis_bgr: np.ndarray,
    *,
    header: str,
    cur_tcp6: np.ndarray,
    target_tcp6: np.ndarray | None = None,
    episode_origin_tcp6: np.ndarray | None = None,
) -> np.ndarray:
    """HUD: episode header, TCP xyz, delta vs start, next waypoint, XY map."""
    out = vis_bgr.copy()
    _put_text_hud(out, header, (10, 20), scale=0.6)

    cur = np.asarray(cur_tcp6, dtype=np.float64).reshape(-1)[:6]
    lines = [f"cur xyz(m): {cur[0]:+.3f} {cur[1]:+.3f} {cur[2]:+.3f}"]
    if episode_origin_tcp6 is not None:
        origin = np.asarray(episode_origin_tcp6, dtype=np.float64).reshape(-1)[:6]
        d0 = cur[:3] - origin[:3]
        lines.append(
            f"vs start dxyz(m): {d0[0]:+.3f} {d0[1]:+.3f} {d0[2]:+.3f}"
        )
    if target_tcp6 is not None:
        tgt = np.asarray(target_tcp6, dtype=np.float64).reshape(-1)[:6]
        d = tgt[:3] - cur[:3]
        lines.append(f"next xyz(m): {tgt[0]:+.3f} {tgt[1]:+.3f} {tgt[2]:+.3f}")
        lines.append(
            f"next dxyz(m): {d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}  |d|={np.linalg.norm(d):.3f}"
        )

    y = 46
    for line in lines:
        _put_text_hud(out, line, (10, y), scale=0.5)
        y += 22

    if episode_origin_tcp6 is not None:
        _draw_xy_localization_panel(out, episode_origin_tcp6, cur, target_tcp6)
    return out


def _scipy_euler_seq(seq: str, extrinsic: bool) -> str:
    """Match IndyInterpolationController: extrinsic -> lower, intrinsic -> UPPER."""
    es = str(seq)
    return es.lower() if extrinsic else es.upper()


def _get_live_display_bgr(env: UmiEnv, camera_idx: int = 0) -> np.ndarray:
    """Full-resolution unmasked camera frame for OpenCV (BGR uint8)."""
    vis_data = env.camera.get_vis()
    return vis_data["color"][camera_idx].copy()


def _rgb_uint8_from_any(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError(f"expected HWC RGB image, got shape={img.shape}")
    img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _policy_input_rgb_from_obs(obs) -> np.ndarray | None:
    img = obs.get("camera0_rgb")
    if img is None:
        return None
    img = np.asarray(img)
    while img.ndim > 3:
        img = img[-1]
    if img.ndim != 3 or img.shape[-1] < 3:
        return None
    return _rgb_uint8_from_any(img)


def _context_log_value_columns(context_schema: str) -> list[str]:
    columns = list(valve_context_spec(context_schema)["value_columns"])
    return ["context_warmed_up" if name == "warmed_up" else name for name in columns]


class _ValveContextInputCapture:
    """Persist the exact frozen-classifier input stream for later inspection.

    Images are lossless PNG copies of the final 224x224 RGB image passed to
    the classifier. The source-rate files are useful for an independent
    algorithm, while ``classifier_windows/window_*.npz`` preserves the exact
    temporal arrays selected by the frozen classifier for each prediction.
    Thus offline work never has to reproduce history sampling, startup
    clamping, rotation-6D conversion, F/T masking, or F/T scaling.

    Neither the frozen classifier nor the diffusion policy has a physical IMU
    input. Do not create a zero-filled IMU surrogate: it would be a fabricated
    feature, not deployment data.
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        episode_start_timestamp_s: float,
        context_schema: str = VALVE_CONTEXT_V1_SCHEMA,
    ):
        self.root = pathlib.Path(root)
        self.image_dir = self.root.joinpath("images")
        self.window_dir = self.root.joinpath("classifier_windows")
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.window_dir.mkdir(parents=True, exist_ok=True)
        self.episode_start_timestamp_s = float(episode_start_timestamp_s)
        self._last_wrench_timestamp = -np.inf
        self._frame_idx = 0
        self._wrench_idx = 0
        self._window_idx = 0
        self._image_idx = 0
        self._image_file_by_timestamp_key: dict[int, str] = {}
        self._image_fingerprint_by_timestamp_key: dict[int, bytes] = {}
        self._frame_timestamp_keys: set[int] = set()
        self._closed = False
        self.context_schema = str(context_schema)
        self.context_spec = valve_context_spec(self.context_schema)
        self.context_dim = int(self.context_spec["dim"])
        if self.context_schema == VALVE_CONTEXT_V1_SCHEMA:
            # warmed_up is already a dedicated CSV column in the legacy log.
            self._frame_context_columns = list(
                self.context_spec["value_columns"][:-1]
            )
            self._frame_context_slice = slice(0, 9)
        else:
            self._frame_context_columns = list(
                self.context_spec["value_columns"]
            )
            self._frame_context_slice = slice(0, self.context_dim)

        self._frame_file = open(self.root.joinpath("context_frames.csv"), "w", newline="")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow(
            [
                "frame_idx", "rgb_timestamp_s", "image_file",
                "latest_wrench_timestamp_s",
                "tcp_x_m", "tcp_y_m", "tcp_z_m",
                "tcp_rotvec_x_rad", "tcp_rotvec_y_rad", "tcp_rotvec_z_rad",
                "gripper_width_m",
                "phase", "error_reason", "warmed_up",
                *self._frame_context_columns,
            ]
        )

        self._wrench_file = open(self.root.joinpath("context_wrenches.csv"), "w", newline="")
        self._wrench_writer = csv.writer(self._wrench_file)
        self._wrench_writer.writerow(
            ["wrench_idx", "timestamp_s"]
            + [f"left_{name}_{unit}" for name, unit in zip(_FT_CHANNEL_LABELS, _FT_CHANNEL_UNITS)]
            + [f"right_{name}_{unit}" for name, unit in zip(_FT_CHANNEL_LABELS, _FT_CHANNEL_UNITS)]
        )

        self._window_index_file = open(
            self.window_dir.joinpath("index.csv"), "w", newline=""
        )
        self._window_index_writer = csv.writer(self._window_index_file)
        self._window_index_writer.writerow(
            [
                "window_idx", "classifier_timestamp_s", "npz_file",
                "temporal_steps", "force_history_samples",
            ]
        )

        self.root.joinpath("capture_manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "context_schema": self.context_schema,
                    "context_dim": self.context_dim,
                    "context_phase_names": list(self.context_spec["phase_names"]),
                    "image_encoding": "lossless PNG; RGB pixels exactly passed to classifier",
                    "image_resolution": [224, 224, 3],
                    "frame_csv": "context_frames.csv",
                    "wrench_csv": "context_wrenches.csv",
                    "wrench_contract": (
                        "startup-bias-corrected native left[Fx,Fy,Fz,Tx,Ty,Tz] + "
                        "right[Fx,Fy,Fz,Tx,Ty,Tz], N/Nm; no external scaling"
                    ),
                    "classifier_windows_index_csv": "classifier_windows/index.csv",
                    "classifier_window_archives": "classifier_windows/window_*.npz",
                    "classifier_window_variants": {
                        "legacy_valve_observer": (
                            "RGB references, lowdim, physical/scaled F/T, and mask"
                        ),
                        "rgb_force_observer": (
                            "two RGB references and one native causal F/T history; "
                            "no TCP pose or gripper-width model input"
                        ),
                    },
                    "classifier_window_contract": {
                        "rgb": (
                            "rgb_image_file references exact lossless PNG source frames; "
                            "the exact model tensor is reconstructed without loss as "
                            "moveaxis(read_rgb_png, -1, 1).astype(float32) / 255.0, "
                            "before ValveStateClassifier.forward's built-in ImageNet normalization"
                        ),
                        "lowdim": (
                            "[x,y,z,rotation_6d(6),gripper_width_m]; v1 uses "
                            "base-frame pose, v2 uses latest-TCP-relative pose"
                        ),
                        "wrench_history_physical": "[T,H,12] corrected physical N/Nm history passed to predict_window",
                        "wrench_history_model_scaled": (
                            "exact [T,H,12] model input after division by "
                            "[10,10,10,0.1,0.1,0.1] for each left/right finger"
                        ),
                        "wrench_mask": "[T,H] causal valid-sample mask passed to the model",
                        "normalization": (
                            "wrench_history_model_scaled records the exact frozen "
                            "observer force normalizer output"
                        ),
                    },
                    "no_imu_file": (
                        "No physical IMU is a direct input of this classifier, so no "
                        "placeholder or fabricated zero-valued IMU file is emitted."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @property
    def frame_count(self) -> int:
        return self._frame_idx

    @property
    def wrench_count(self) -> int:
        return self._wrench_idx

    @property
    def window_count(self) -> int:
        return self._window_idx

    @staticmethod
    def _timestamp_key(timestamp_s: float) -> int:
        """Stable image lookup key at a resolution above the timing contract."""
        return int(round(float(timestamp_s) * 1_000_000_000.0))

    def append_wrenches(
        self,
        timestamps,
        left_wrenches,
        right_wrenches,
        *,
        anchor_timestamp_s: float,
    ) -> float | None:
        """Record only new causal samples, matching runtime append semantics."""
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        left_wrenches = np.asarray(left_wrenches, dtype=np.float32)
        right_wrenches = np.asarray(right_wrenches, dtype=np.float32)
        if left_wrenches.shape != (len(timestamps), 6) or right_wrenches.shape != (
            len(timestamps), 6
        ):
            raise ValueError("context capture F/T requires matching [N,6] arrays")
        if not np.isfinite(timestamps).all() or not np.isfinite(left_wrenches).all() or not np.isfinite(right_wrenches).all():
            raise ValueError("context capture F/T contains NaN or Inf")
        if np.any(np.diff(timestamps) < 0.0):
            raise ValueError("context capture F/T timestamps must be sorted")
        if np.any(timestamps > float(anchor_timestamp_s) + 1e-6):
            raise ValueError("context capture received F/T newer than its RGB anchor")

        for timestamp_s, left, right in zip(timestamps, left_wrenches, right_wrenches):
            if timestamp_s < self.episode_start_timestamp_s:
                continue
            if timestamp_s <= self._last_wrench_timestamp:
                continue
            self._wrench_writer.writerow(
                [self._wrench_idx, float(timestamp_s)]
                + np.asarray(left, dtype=np.float64).tolist()
                + np.asarray(right, dtype=np.float64).tolist()
            )
            self._wrench_idx += 1
            self._last_wrench_timestamp = float(timestamp_s)
        return (
            float(self._last_wrench_timestamp)
            if np.isfinite(self._last_wrench_timestamp)
            else None
        )

    def _ensure_image(self, timestamp_s: float, rgb: np.ndarray) -> str:
        """Save an exact RGB source once and safely reuse it across windows."""
        rgb = _rgb_uint8_from_any(rgb)
        if rgb.shape != (224, 224, 3):
            raise ValueError(
                f"context capture RGB must be [224,224,3], got {rgb.shape}"
            )
        timestamp_key = self._timestamp_key(timestamp_s)
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(rgb).tobytes()
        ).digest()
        existing = self._image_file_by_timestamp_key.get(timestamp_key)
        if existing is not None:
            if (
                self._image_fingerprint_by_timestamp_key[timestamp_key]
                != fingerprint
            ):
                raise ValueError(
                    "context capture RGB pixels changed at an existing timestamp"
                )
            return existing
        image_relpath = pathlib.Path("images").joinpath(
            f"frame_{self._image_idx:08d}.png"
        )
        image_path = self.root.joinpath(image_relpath)
        if not cv2.imwrite(
            str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(
                f"failed to save lossless context image: {image_path}"
            )
        self._image_file_by_timestamp_key[timestamp_key] = str(image_relpath)
        self._image_fingerprint_by_timestamp_key[timestamp_key] = fingerprint
        self._image_idx += 1
        return str(image_relpath)

    def append_source_images(self, timestamps, images) -> None:
        """Persist RGB references selected by a stateless classifier window."""
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        images = np.asarray(images)
        if images.shape != (len(timestamps), 224, 224, 3):
            raise ValueError("context source RGB history has inconsistent shape")
        for timestamp_s, rgb in zip(timestamps, images):
            self._ensure_image(float(timestamp_s), rgb)

    def append_frame(
        self,
        *,
        timestamp_s: float,
        rgb: np.ndarray,
        position_m: np.ndarray,
        rotation_axis_angle_rad: np.ndarray,
        gripper_width_m: float,
        latest_wrench_timestamp_s: float | None,
        context_record,
    ) -> None:
        rgb = _rgb_uint8_from_any(rgb)
        if rgb.shape != (224, 224, 3):
            raise ValueError(f"context capture RGB must be [224,224,3], got {rgb.shape}")
        position_m = np.asarray(position_m, dtype=np.float64).reshape(3)
        rotation_axis_angle_rad = np.asarray(
            rotation_axis_angle_rad, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(position_m).all() or not np.isfinite(rotation_axis_angle_rad).all():
            raise ValueError("context capture TCP input contains NaN or Inf")
        if not np.isfinite(gripper_width_m):
            raise ValueError("context capture gripper width is not finite")
        if not np.isclose(float(context_record.timestamp_s), float(timestamp_s), atol=1e-6):
            raise ValueError("context capture record timestamp does not match RGB timestamp")

        timestamp_key = self._timestamp_key(timestamp_s)
        if timestamp_key in self._frame_timestamp_keys:
            raise ValueError("context capture received a duplicate output timestamp")
        image_relpath = self._ensure_image(timestamp_s, rgb)
        self._frame_timestamp_keys.add(timestamp_key)
        if str(getattr(context_record, "schema", self.context_schema)) != self.context_schema:
            raise ValueError("context record schema changed during capture")
        values = np.asarray(context_record.values, dtype=np.float64).reshape(
            self.context_dim
        )
        self._frame_writer.writerow(
            [
                self._frame_idx,
                float(timestamp_s),
                str(image_relpath),
                "" if latest_wrench_timestamp_s is None else float(latest_wrench_timestamp_s),
                *position_m.tolist(),
                *rotation_axis_angle_rad.tolist(),
                float(gripper_width_m),
                context_record.phase_name,
                context_record.error_reason_name,
                int(context_record.warmed_up),
                *values[self._frame_context_slice].tolist(),
            ]
        )
        self._frame_idx += 1

    def append_classifier_window(
        self,
        *,
        classifier_timestamp_s: float,
        model_inputs: dict[str, np.ndarray],
    ) -> None:
        """Save the exact classifier temporal window used for one prediction.

        RGB images are referenced rather than duplicated: each reference names
        one source image in ``images/`` written by ``append_frame``. All
        numerical arrays are self-contained in the compressed NPZ archive.
        """
        if self._closed:
            raise RuntimeError("context input capture is already closed")
        classifier_timestamp_s = float(classifier_timestamp_s)
        if not np.isfinite(classifier_timestamp_s):
            raise ValueError("classifier window timestamp is not finite")
        rgb_force_required = {
            "observer_input_schema",
            "rgb_timestamp_s",
            "camera0_rgb",
            "ft_timestamp_s",
            "robot0_ft_left",
            "robot0_ft_right",
        }
        if rgb_force_required.issubset(model_inputs):
            rgb_timestamps = np.asarray(
                model_inputs["rgb_timestamp_s"], dtype=np.float64
            ).reshape(-1)
            rgb = np.asarray(model_inputs["camera0_rgb"])
            ft_timestamps = np.asarray(
                model_inputs["ft_timestamp_s"], dtype=np.float64
            ).reshape(-1)
            ft_left = np.asarray(
                model_inputs["robot0_ft_left"], dtype=np.float32
            )
            ft_right = np.asarray(
                model_inputs["robot0_ft_right"], dtype=np.float32
            )
            if (
                rgb.shape != (len(rgb_timestamps), 224, 224, 3)
                or rgb.dtype != np.uint8
                or ft_left.shape != (len(ft_timestamps), 6)
                or ft_right.shape != (len(ft_timestamps), 6)
            ):
                raise ValueError(
                    "RGB/F-T classifier window has invalid shapes: "
                    f"rgb={rgb.shape} left={ft_left.shape} right={ft_right.shape}"
                )
            if (
                len(rgb_timestamps) != 2
                or len(ft_timestamps) == 0
                or not np.isfinite(rgb_timestamps).all()
                or not np.isfinite(ft_timestamps).all()
                or np.any(np.diff(rgb_timestamps) <= 0.0)
                or np.any(np.diff(ft_timestamps) <= 0.0)
                or not np.isfinite(ft_left).all()
                or not np.isfinite(ft_right).all()
            ):
                raise ValueError("RGB/F-T classifier window is nonfinite or noncausal")
            if not np.isclose(
                rgb_timestamps[-1], classifier_timestamp_s, atol=1e-6
            ):
                raise ValueError(
                    "RGB/F-T classifier latest RGB timestamp is not its anchor"
                )
            if ft_timestamps[-1] > classifier_timestamp_s + 1e-6:
                raise ValueError("RGB/F-T classifier window contains future F/T")
            image_files = []
            for timestamp_s in rgb_timestamps:
                image_file = self._image_file_by_timestamp_key.get(
                    self._timestamp_key(timestamp_s)
                )
                if image_file is None:
                    raise ValueError(
                        "RGB/F-T classifier window references an unsaved RGB frame"
                    )
                image_files.append(image_file)
            archive_relpath = pathlib.Path("classifier_windows").joinpath(
                f"window_{self._window_idx:08d}.npz"
            )
            np.savez(
                self.root.joinpath(archive_relpath),
                classifier_timestamp_s=np.asarray(
                    classifier_timestamp_s, dtype=np.float64
                ),
                observer_input_schema=np.asarray(
                    model_inputs["observer_input_schema"]
                ),
                rgb_timestamp_s=rgb_timestamps,
                rgb_image_file=np.asarray(image_files),
                ft_timestamp_s=ft_timestamps,
                robot0_ft_left=ft_left,
                robot0_ft_right=ft_right,
            )
            self._window_index_writer.writerow(
                [
                    self._window_idx,
                    classifier_timestamp_s,
                    str(archive_relpath),
                    len(rgb_timestamps),
                    len(ft_timestamps),
                ]
            )
            self._window_idx += 1
            return
        required = {
            "rgb_timestamp_s",
            "lowdim",
            "wrench_history_physical",
            "wrench_history_model_scaled",
            "wrench_mask",
        }
        missing = required - set(model_inputs)
        if missing:
            raise ValueError(
                "classifier model-input snapshot is incomplete: "
                + ", ".join(sorted(missing))
            )
        rgb_timestamps = np.asarray(
            model_inputs["rgb_timestamp_s"], dtype=np.float64
        ).reshape(-1)
        lowdim = np.asarray(model_inputs["lowdim"], dtype=np.float32)
        wrench_physical = np.asarray(
            model_inputs["wrench_history_physical"], dtype=np.float32
        )
        wrench_scaled = np.asarray(
            model_inputs["wrench_history_model_scaled"], dtype=np.float32
        )
        wrench_mask = np.asarray(model_inputs["wrench_mask"], dtype=np.float32)
        steps = len(rgb_timestamps)
        if (
            lowdim.shape != (steps, 10)
            or wrench_physical.ndim != 3
            or wrench_physical.shape[0] != steps
            or wrench_physical.shape[2] != 12
            or wrench_scaled.shape != wrench_physical.shape
            or wrench_mask.shape != wrench_physical.shape[:2]
        ):
            raise ValueError(
                "classifier temporal window has invalid shapes: "
                f"lowdim={lowdim.shape} physical={wrench_physical.shape} "
                f"scaled={wrench_scaled.shape} mask={wrench_mask.shape}"
            )
        if (
            not np.isfinite(rgb_timestamps).all()
            or np.any(np.diff(rgb_timestamps) < 0.0)
            or not np.isfinite(lowdim).all()
            or not np.isfinite(wrench_physical).all()
            or not np.isfinite(wrench_scaled).all()
            or not np.isfinite(wrench_mask).all()
        ):
            raise ValueError("classifier temporal window contains NaN or Inf")
        if not np.isclose(rgb_timestamps[-1], classifier_timestamp_s, atol=1e-6):
            raise ValueError("classifier window latest RGB timestamp is not its anchor")
        image_files = []
        for timestamp_s in rgb_timestamps:
            image_file = self._image_file_by_timestamp_key.get(
                self._timestamp_key(timestamp_s)
            )
            if image_file is None:
                raise ValueError(
                    "classifier temporal window references an RGB frame that "
                    "was not saved in context_inputs/images"
                )
            image_files.append(image_file)
        archive_relpath = pathlib.Path("classifier_windows").joinpath(
            f"window_{self._window_idx:08d}.npz"
        )
        # Do not zlib-compress one archive per camera frame on the context
        # worker thread. The saved arrays are small once RGB is referenced,
        # and uncompressed writes keep classifier latency bounded.
        np.savez(
            self.root.joinpath(archive_relpath),
            classifier_timestamp_s=np.asarray(classifier_timestamp_s, dtype=np.float64),
            rgb_timestamp_s=rgb_timestamps,
            rgb_image_file=np.asarray(image_files),
            lowdim=lowdim,
            wrench_history_physical=wrench_physical,
            wrench_history_model_scaled=wrench_scaled,
            wrench_mask=wrench_mask,
        )
        self._window_index_writer.writerow(
            [
                self._window_idx,
                classifier_timestamp_s,
                str(archive_relpath),
                steps,
                wrench_physical.shape[1],
            ]
        )
        self._window_idx += 1

    def flush(self) -> None:
        self._frame_file.flush()
        self._wrench_file.flush()
        self._window_index_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._frame_file.close()
        self._wrench_file.close()
        self._window_index_file.close()
        self._closed = True


class _PolicyInputCapture:
    """Persist the exact NumPy observations passed to ``policy.predict_action``.

    This is deliberately separate from ``_ValveContextInputCapture``.  The
    latter runs at camera rate for the frozen classifier; this capture writes
    once per diffusion-policy inference and stores the post-preprocessing,
    pre-normalizer arrays.  That means a different algorithm can consume the
    same policy observation contract without reconstructing relative poses,
    rotation-6D, image layout, or causal F/T history from display logs.
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        shape_meta: dict,
        episode_start_timestamp_s: float,
    ) -> None:
        self.root = pathlib.Path(root)
        self.image_dir = self.root.joinpath("images")
        self.sample_dir = self.root.joinpath("samples")
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.shape_meta = shape_meta
        self.episode_start_timestamp_s = float(episode_start_timestamp_s)
        self._sample_count = 0
        self._closed = False

        self._index_file = open(self.root.joinpath("index.csv"), "w", newline="")
        self._index_writer = csv.writer(self._index_file)
        self._index_writer.writerow(
            [
                "sample_idx", "policy_iter_idx", "policy_anchor_timestamp_s",
                "npz_file", "camera0_rgb_t0_file", "camera0_rgb_t1_file",
            ]
        )
        self._ft_file = open(self.root.joinpath("ft_history.csv"), "w", newline="")
        self._ft_writer = csv.writer(self._ft_file)
        self._ft_writer.writerow(
            [
                "sample_idx", "policy_iter_idx", "policy_anchor_timestamp_s",
                "finger", "history_idx_oldest_to_latest", "source_timestamp_s",
            ]
            + [f"{name}_{unit}" for name, unit in zip(_FT_CHANNEL_LABELS, _FT_CHANNEL_UNITS)]
        )
        self.root.joinpath("capture_manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "capture_stage": (
                        "exact NumPy obs_dict passed to policy.predict_action "
                        "after get_real_umi_obs_dict/_policy_obs_float32 and "
                        "before batch dimension, torch conversion, and policy normalizer"
                    ),
                    "shape_meta_obs_keys": list(shape_meta["obs"].keys()),
                    "sample_archives": "samples/sample_*.npz",
                    "index_csv": "index.csv",
                    "image_files": "images/sample_*_camera0_rgb_t*.png",
                    "ft_history_csv": "ft_history.csv",
                    "ft_contract": (
                        "exact policy input history: native left/right [Fx,Fy,Fz,Tx,Ty,Tz], "
                        "N/Nm, after live startup-bias correction; no policy normalizer applied"
                    ),
                    "input_key_rule": (
                        "Every and only shape_meta.obs key is archived. This checkpoint "
                        "does not define an IMU key, so no IMU placeholder is emitted."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @staticmethod
    def _validate_arrays(obs_dict_np: dict, expected_keys: set[str]) -> dict:
        actual_keys = set(obs_dict_np.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing or extra:
            raise ValueError(
                "policy input capture key mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        arrays = {}
        for key in sorted(expected_keys):
            value = np.asarray(obs_dict_np[key])
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"policy input {key} is empty or non-finite")
            arrays[key] = np.ascontiguousarray(value.copy())
        return arrays

    @staticmethod
    def _write_rgb_images(
        image_dir: pathlib.Path, sample_idx: int, rgb_tchw: np.ndarray
    ) -> list[str]:
        rgb_tchw = np.asarray(rgb_tchw)
        if rgb_tchw.ndim != 4 or rgb_tchw.shape[1:] != (3, 224, 224):
            raise ValueError(
                "policy camera0_rgb must be [T,3,224,224], got "
                f"{rgb_tchw.shape}"
            )
        relpaths = []
        for time_idx, rgb_chw in enumerate(rgb_tchw):
            rgb = _rgb_uint8_from_any(np.moveaxis(rgb_chw, 0, -1))
            relpath = pathlib.Path("images").joinpath(
                f"sample_{sample_idx:06d}_camera0_rgb_t{time_idx}.png"
            )
            if not cv2.imwrite(
                str(image_dir.parent.joinpath(relpath)),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            ):
                raise RuntimeError(f"failed to save policy input image {relpath}")
            relpaths.append(str(relpath))
        return relpaths

    def append(
        self,
        *,
        policy_iter_idx: int,
        policy_anchor_timestamp_s: float,
        obs_dict_np: dict,
        source_obs: dict,
    ) -> None:
        if self._closed:
            raise RuntimeError("policy input capture is already closed")
        policy_anchor_timestamp_s = float(policy_anchor_timestamp_s)
        if not np.isfinite(policy_anchor_timestamp_s):
            raise ValueError("policy input capture anchor timestamp is not finite")
        expected_keys = set(self.shape_meta["obs"].keys())
        arrays = self._validate_arrays(obs_dict_np, expected_keys)
        sample_idx = self._sample_count
        image_paths = self._write_rgb_images(
            self.image_dir, sample_idx, arrays["camera0_rgb"]
        )
        if len(image_paths) != 2:
            raise ValueError(
                "current policy contract requires exactly two camera0_rgb frames"
            )

        archive_relpath = pathlib.Path("samples").joinpath(
            f"sample_{sample_idx:06d}.npz"
        )
        np.savez_compressed(
            self.root.joinpath(archive_relpath),
            policy_iter_idx=np.asarray(int(policy_iter_idx), dtype=np.int64),
            policy_anchor_timestamp_s=np.asarray(
                policy_anchor_timestamp_s, dtype=np.float64
            ),
            **arrays,
        )
        self._index_writer.writerow(
            [
                sample_idx,
                int(policy_iter_idx),
                policy_anchor_timestamp_s,
                str(archive_relpath),
                *image_paths,
            ]
        )

        for finger in ("left", "right"):
            key = f"robot0_ft_{finger}"
            if key not in arrays:
                continue
            values = arrays[key]
            if values.ndim != 2 or values.shape[1] != 6:
                raise ValueError(f"policy input {key} must be [T,6], got {values.shape}")
            source_timestamps = np.asarray(
                source_obs.get(f"{key}_timestamps", []), dtype=np.float64
            )
            if source_timestamps.shape != (len(values),):
                raise ValueError(
                    f"policy input source timestamps for {key} do not match history"
                )
            if np.any(source_timestamps > policy_anchor_timestamp_s + 1e-6):
                raise ValueError(f"policy F/T {finger} history is newer than its RGB anchor")
            for history_idx, (timestamp_s, wrench) in enumerate(
                zip(source_timestamps, values)
            ):
                self._ft_writer.writerow(
                    [
                        sample_idx,
                        int(policy_iter_idx),
                        policy_anchor_timestamp_s,
                        finger,
                        history_idx,
                        float(timestamp_s),
                        *np.asarray(wrench, dtype=np.float64).tolist(),
                    ]
                )
        self._sample_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._index_file.flush()
        self._ft_file.flush()
        self._index_file.close()
        self._ft_file.close()
        self._closed = True


class _ValveContextWorker:
    """Own the stateful frozen classifier at camera rate, off the policy loop.

    The diffusion policy replans at about 20 Hz but the frozen context model
    was trained on every camera observation.  This worker polls an independent
    long camera history, replays each previously unseen frame in timestamp
    order, and retains a short timestamp-indexed record cache.  The policy
    thread waits only for the record with its exact RGB anchor and never feeds
    a newer context value into an older policy image.
    """

    def __init__(
        self,
        runtime: ValveStateContextRuntime,
        stream_provider,
        *,
        input_capture: _ValveContextInputCapture | None = None,
        poll_hz: float = 60.0,
        max_cached_records: int = 256,
        history_frames: int = 120,
        initial_history_frames: int = 1,
        anchor_recovery_history_frames: int = 32,
    ):
        if not np.isfinite(poll_hz) or poll_hz <= 0.0:
            raise ValueError("context worker poll_hz must be positive")
        if int(max_cached_records) < 2:
            raise ValueError("context worker needs at least two cached records")
        if int(history_frames) < 2:
            raise ValueError("context worker recovery needs at least two frames")
        if not 1 <= int(initial_history_frames) <= int(history_frames):
            raise ValueError("initial context history must be within worker history")
        if not 1 <= int(anchor_recovery_history_frames) <= int(history_frames):
            raise ValueError("anchor recovery history must be within worker history")
        self.runtime = runtime
        self.stream_provider = stream_provider
        self.input_capture = input_capture
        self.poll_period_s = 1.0 / float(poll_hz)
        self.max_cached_records = int(max_cached_records)
        self.history_frames = int(history_frames)
        self.initial_history_frames = int(initial_history_frames)
        self.anchor_recovery_history_frames = int(anchor_recovery_history_frames)
        self._records = collections.OrderedDict()
        self._condition = threading.Condition()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._error = None
        self._initial_history_pending = True
        self._anchor_recovery_request = None
        self._episode_start_timestamp_s = -np.inf
        self._last_frame_timestamp_s = -np.inf
        self._last_camera_timestamp_s = None
        self._last_camera_step_idx = None
        # Estimate the physical camera period from timestamps.  ``step_idx``
        # is calculated against the requested put rate (60 Hz), so a healthy
        # 40 Hz UVC source naturally advances it by two on some frames.
        self._recent_camera_periods_s = collections.deque(maxlen=12)
        self._metrics = {
            "processed_frames": 0,
            "max_frame_gap_s": 0.0,
            "sum_frame_gap_s": 0.0,
            "frame_gap_count": 0,
            "poll_count": 0,
            "recovery_polls": 0,
            "anchor_recovery_polls": 0,
            "transient_stream_timeouts": 0,
            "last_transient_stream_timeout": None,
            "last_record_timestamp_s": None,
        }

    @staticmethod
    def _timestamp_key(timestamp_s: float) -> int:
        return int(round(float(timestamp_s) * 1_000_000.0))

    @staticmethod
    def _rgb_fingerprint(rgb: np.ndarray) -> bytes:
        array = np.ascontiguousarray(_rgb_uint8_from_any(rgb))
        return hashlib.sha256(array.tobytes()).digest()

    def _get_stream(self, *, history_frames: int) -> dict | None:
        """Read one camera-rate stream without killing the worker on a hiccup.

        UVC/shared-memory reads use a short internal timeout. A single delayed
        frame must not turn into a permanent classifier failure; the policy
        still requires an exact subsequent anchor and will safety-stop if one
        cannot be obtained within its separate bounded wait.
        """
        try:
            return self.stream_provider(history_frames=history_frames)
        except TimeoutError as exc:
            with self._condition:
                self._metrics["transient_stream_timeouts"] += 1
                self._metrics["last_transient_stream_timeout"] = repr(exc)
                self._condition.notify_all()
            return None

    def start(self, *, episode_start_timestamp_s: float) -> None:
        if self._thread is not None:
            raise RuntimeError("context worker may only be started once")
        episode_start_timestamp_s = float(episode_start_timestamp_s)
        if not np.isfinite(episode_start_timestamp_s):
            raise ValueError("context worker episode start must be finite")
        self.runtime.reset(episode_start_timestamp_s=episode_start_timestamp_s)
        self._episode_start_timestamp_s = episode_start_timestamp_s
        self._thread = threading.Thread(
            target=self._run,
            name="ValveContextWorker",
            daemon=True,
        )
        self._thread.start()
        self._wake_event.set()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_s)))
            if self._thread.is_alive():
                raise RuntimeError("context worker did not stop within timeout")
        with self._condition:
            self._condition.notify_all()

    def _store_record(self, timestamp_s: float, rgb: np.ndarray, record) -> None:
        key = self._timestamp_key(timestamp_s)
        fingerprint = self._rgb_fingerprint(rgb)
        with self._condition:
            self._records[key] = (float(timestamp_s), record, fingerprint)
            self._records.move_to_end(key)
            while len(self._records) > self.max_cached_records:
                self._records.popitem(last=False)
            self._metrics["processed_frames"] += 1
            if np.isfinite(self._last_frame_timestamp_s):
                gap_s = float(timestamp_s) - self._last_frame_timestamp_s
                if gap_s > 0.0:
                    self._metrics["frame_gap_count"] += 1
                    self._metrics["sum_frame_gap_s"] += gap_s
                    self._metrics["max_frame_gap_s"] = max(
                        self._metrics["max_frame_gap_s"], gap_s
                    )
            self._last_frame_timestamp_s = float(timestamp_s)
            self._metrics["last_record_timestamp_s"] = float(timestamp_s)
            self._condition.notify_all()

    def _process_stream(self, stream: dict) -> None:
        timestamps = np.asarray(stream.get("rgb_timestamp_s"), dtype=np.float64)
        camera_step_indices = np.asarray(
            stream.get("camera_step_idx"), dtype=np.int64
        )
        rgb_frames = np.asarray(stream.get("camera0_rgb"))
        positions = np.asarray(stream.get("robot0_eef_pos"), dtype=np.float64)
        rotations = np.asarray(
            stream.get("robot0_eef_rot_axis_angle"), dtype=np.float64
        )
        widths = np.asarray(stream.get("robot0_gripper_width"), dtype=np.float64)
        ft_timestamps = np.asarray(stream.get("ft_timestamp_s"), dtype=np.float64)
        ft_left = np.asarray(stream.get("robot0_ft_left"), dtype=np.float32)
        ft_right = np.asarray(stream.get("robot0_ft_right"), dtype=np.float32)
        n = len(timestamps)
        if (
            rgb_frames.shape[0] != n
            or camera_step_indices.shape != (n,)
            or positions.shape != (n, 3)
            or rotations.shape != (n, 3)
            or widths.shape[0] != n
        ):
            raise ValueError("context worker received inconsistent camera-rate stream")
        if n == 0 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
            raise ValueError("context worker RGB timestamps must be finite and sorted")
        # UVC ``step_idx`` is an arrival-rate estimate used only to notice a
        # missed poll.  A shared-memory recovery read can include frames from
        # either side of a capture-process restart, so it is not a reliable
        # chronological clock.  ``rgb_timestamp_s`` above is the authoritative
        # classifier timeline and remains strictly validated/sorted.
        if (
            ft_left.shape != (len(ft_timestamps), 6)
            or ft_right.shape != (len(ft_timestamps), 6)
            or not np.isfinite(ft_timestamps).all()
            or np.any(np.diff(ft_timestamps) < 0.0)
        ):
            raise ValueError("context worker received invalid F/T stream")

        for idx, timestamp_s in enumerate(timestamps):
            # A history-recovery stream can contain many frames. Do not make
            # shutdown wait for every queued classifier/logging operation when
            # the policy has already stopped or failed its safety check.
            if self._stop_event.is_set():
                break
            timestamp_s = float(timestamp_s)
            if timestamp_s < self._episode_start_timestamp_s:
                continue
            if timestamp_s <= self.runtime.last_prediction_timestamp:
                continue
            ft_end = int(np.searchsorted(ft_timestamps, timestamp_s, side="right"))
            latest_wrench_timestamp_s = None
            if self.input_capture is not None:
                latest_wrench_timestamp_s = self.input_capture.append_wrenches(
                    ft_timestamps[:ft_end],
                    ft_left[:ft_end],
                    ft_right[:ft_end],
                    anchor_timestamp_s=timestamp_s,
                )
            rgb = _rgb_uint8_from_any(rgb_frames[idx])
            record = self.runtime.predict(
                timestamp_s=timestamp_s,
                rgb=rgb,
                position_m=positions[idx],
                rotation_axis_angle_rad=rotations[idx],
                gripper_width_m=float(np.asarray(widths[idx]).reshape(-1)[0]),
                ft_timestamps=ft_timestamps[:ft_end],
                ft_left=ft_left[:ft_end],
                ft_right=ft_right[:ft_end],
            )
            if self.input_capture is not None:
                self.input_capture.append_frame(
                    timestamp_s=timestamp_s,
                    rgb=rgb,
                    position_m=positions[idx],
                    rotation_axis_angle_rad=rotations[idx],
                    gripper_width_m=float(np.asarray(widths[idx]).reshape(-1)[0]),
                    latest_wrench_timestamp_s=latest_wrench_timestamp_s,
                    context_record=record,
                )
                if not (
                    isinstance(self.runtime, RGBForceValveContextRuntime)
                    and not record.warmed_up
                ):
                    self.input_capture.append_classifier_window(
                        classifier_timestamp_s=timestamp_s,
                        model_inputs=self.runtime.get_last_classifier_model_inputs(),
                    )
            self._store_record(timestamp_s, rgb, record)
        if self.input_capture is not None:
            self.input_capture.flush()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._wake_event.wait(timeout=self.poll_period_s)
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                recover_history = False
                anchor_recovery = False
                with self._condition:
                    anchor_request = self._anchor_recovery_request
                    self._anchor_recovery_request = None
                if anchor_request is not None:
                    # A policy anchor can be older than the worker's newest
                    # one-frame poll because camera timestamps are latency
                    # compensated.  Recover it by timestamp/RGB, never by
                    # substituting a newer classifier result.
                    stream = self._get_stream(
                        history_frames=self.anchor_recovery_history_frames
                    )
                    anchor_recovery = True
                    with self._condition:
                        self._metrics["anchor_recovery_polls"] += 1
                elif self._initial_history_pending:
                    # At episode handoff, the first policy RGB can predate
                    # eval_t_start by one camera-latency interval.  Seed from
                    # a short retained history so that exact first anchor is
                    # available without using a future context value.
                    stream = self._get_stream(
                        history_frames=self.initial_history_frames
                    )
                    self._initial_history_pending = False
                else:
                    # The common path transfers one newest frame.  Detect
                    # actual capture gaps from camera timestamps, not UVC
                    # ``step_idx``: the latter is based on requested 60 Hz and
                    # can skip values normally when source is slower.
                    stream = self._get_stream(history_frames=1)
                    if stream is None:
                        continue
                    timestamps = np.asarray(
                        stream.get("rgb_timestamp_s"), dtype=np.float64
                    )
                    step_indices = np.asarray(
                        stream.get("camera_step_idx"), dtype=np.int64
                    )
                    if timestamps.shape != (1,) or step_indices.shape != (1,):
                        raise ValueError(
                            "context worker newest-frame stream must contain one "
                            "RGB timestamp and camera step index"
                        )
                    newest_timestamp_s = float(timestamps[0])
                    newest_step_idx = int(step_indices[0])
                    if (
                        self._last_camera_timestamp_s is not None
                        and newest_timestamp_s > self._last_camera_timestamp_s
                    ):
                        period_s = newest_timestamp_s - self._last_camera_timestamp_s
                        if len(self._recent_camera_periods_s) >= 3:
                            expected_period_s = float(
                                np.median(self._recent_camera_periods_s)
                            )
                            # A one-frame loss is ~2x normal period.  Leave
                            # room for small UVC timestamp jitter.
                            recover_history = period_s > 1.75 * expected_period_s
                        elif (
                            self._last_camera_step_idx is not None
                            and newest_step_idx > self._last_camera_step_idx + 2
                        ):
                            # Before a timestamp-rate estimate exists, recover
                            # only a clearly large initial jump.  A two-step
                            # jump is normal for a healthy ~40 Hz source
                            # requested at 60 Hz.
                            recover_history = True
                    if recover_history:
                        stream = self._get_stream(
                            history_frames=self.history_frames
                        )
                        with self._condition:
                            self._metrics["recovery_polls"] += 1
                if stream is None:
                    continue
                self._process_stream(stream)
                processed_timestamps = np.asarray(
                    stream.get("rgb_timestamp_s"), dtype=np.float64
                )
                processed_indices = np.asarray(
                    stream.get("camera_step_idx"), dtype=np.int64
                )
                if (
                    processed_timestamps.ndim != 1
                    or len(processed_timestamps) == 0
                    or processed_indices.shape != processed_timestamps.shape
                ):
                    raise ValueError("context worker stream has no camera timestamps")
                newest_timestamp_s = float(processed_timestamps[-1])
                if (
                    self._last_camera_timestamp_s is not None
                    and newest_timestamp_s > self._last_camera_timestamp_s
                    and not recover_history
                    and not anchor_recovery
                ):
                    self._recent_camera_periods_s.append(
                        newest_timestamp_s - self._last_camera_timestamp_s
                    )
                self._last_camera_timestamp_s = newest_timestamp_s
                self._last_camera_step_idx = int(processed_indices[-1])
                with self._condition:
                    self._metrics["poll_count"] += 1
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def record_for_policy_anchor(
        self,
        *,
        timestamp_s: float,
        rgb: np.ndarray,
        timeout_s: float = 0.10,
    ):
        """Return the classifier result for exactly this policy RGB frame."""
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("policy anchor timestamp must be finite")
        desired_key = self._timestamp_key(timestamp_s)
        desired_fingerprint = self._rgb_fingerprint(rgb)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        self._wake_event.set()
        requested_history_recovery = False
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("context worker failed") from self._error
                item = self._records.get(desired_key)
                if item is not None:
                    record_timestamp_s, record, fingerprint = item
                    if not np.isclose(record_timestamp_s, timestamp_s, atol=1e-6):
                        raise RuntimeError("context worker timestamp-key collision")
                    if fingerprint != desired_fingerprint:
                        raise RuntimeError(
                            "context worker RGB differs from the policy RGB at the same timestamp"
                        )
                    return record
                if not requested_history_recovery:
                    self._anchor_recovery_request = (
                        timestamp_s, desired_key, desired_fingerprint
                    )
                    requested_history_recovery = True
                    self._wake_event.set()
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    last_timestamp_s = self._metrics["last_record_timestamp_s"]
                    raise TimeoutError(
                        "context worker did not produce the policy anchor within "
                        f"{timeout_s:.3f}s (anchor={timestamp_s:.6f}, "
                        f"latest={last_timestamp_s})"
                    )
                self._condition.wait(timeout=remaining_s)

    def summary(self) -> dict:
        with self._condition:
            summary = dict(self._metrics)
            gaps = int(summary.pop("frame_gap_count"))
            total_gap_s = float(summary.pop("sum_frame_gap_s"))
            summary["mean_frame_period_s"] = (
                total_gap_s / gaps if gaps > 0 else None
            )
            summary["episode_start_timestamp_s"] = self._episode_start_timestamp_s
            summary["cached_records"] = len(self._records)
            summary["error"] = None if self._error is None else repr(self._error)
            return summary


def _add_valve_context_to_policy_observation(
    obs: dict,
    *,
    runtime: ValveStateContextRuntime,
    input_capture: _ValveContextInputCapture | None = None,
) -> tuple[dict, object]:
    """Replay every unseen policy-preprocessed RGB frame through classifier v4.

    ``UmiEnv`` supplies a short overlapping camera-rate stream on each
    replanning call.  Replaying unseen frames makes the classifier run at the
    camera rate even though diffusion action inference is slower.  Each call
    receives only startup-bias-corrected physical F/T samples at or before its
    RGB timestamp; classifier-side wrench scaling remains untouched.
    """

    stream = obs.get("valve_context_stream")
    if isinstance(stream, dict):
        timestamps = np.asarray(stream.get("rgb_timestamp_s"), dtype=np.float64)
        rgb_frames = np.asarray(stream.get("camera0_rgb"))
        positions = np.asarray(stream.get("robot0_eef_pos"), dtype=np.float64)
        rotations = np.asarray(
            stream.get("robot0_eef_rot_axis_angle"), dtype=np.float64
        )
        widths = np.asarray(stream.get("robot0_gripper_width"), dtype=np.float64)
        ft_timestamps = np.asarray(stream.get("ft_timestamp_s"), dtype=np.float64)
        ft_left = np.asarray(stream.get("robot0_ft_left"), dtype=np.float32)
        ft_right = np.asarray(stream.get("robot0_ft_right"), dtype=np.float32)
        n = len(timestamps)
        if (
            rgb_frames.shape[0] != n
            or positions.shape != (n, 3)
            or rotations.shape != (n, 3)
            or widths.shape[0] != n
        ):
            raise ValueError("valve_context_stream has inconsistent camera-rate shapes")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
            raise ValueError("valve_context_stream RGB timestamps must be finite/sorted")
        if n == 0:
            raise ValueError("valve_context_stream contains no camera frames")
        policy_rgb = _policy_input_rgb_from_obs(obs)
        if policy_rgb is None or not np.array_equal(
            _rgb_uint8_from_any(rgb_frames[-1]), policy_rgb
        ):
            raise ValueError(
                "classifier RGB must exactly equal the final latest policy image"
            )
        if not np.isclose(
            timestamps[-1], float(np.asarray(obs["timestamp"])[-1]), atol=1e-6
        ):
            raise ValueError(
                "valve_context_stream latest RGB timestamp must equal policy anchor"
            )
        for idx, timestamp_s in enumerate(timestamps):
            if timestamp_s <= runtime.last_prediction_timestamp:
                continue
            ft_end = int(np.searchsorted(ft_timestamps, timestamp_s, side="right"))
            latest_wrench_timestamp_s = None
            if input_capture is not None:
                latest_wrench_timestamp_s = input_capture.append_wrenches(
                    ft_timestamps[:ft_end],
                    ft_left[:ft_end],
                    ft_right[:ft_end],
                    anchor_timestamp_s=float(timestamp_s),
                )
            record = runtime.predict(
                timestamp_s=float(timestamp_s),
                rgb=_rgb_uint8_from_any(rgb_frames[idx]),
                position_m=positions[idx],
                rotation_axis_angle_rad=rotations[idx],
                gripper_width_m=float(np.asarray(widths[idx]).reshape(-1)[0]),
                ft_timestamps=ft_timestamps[:ft_end],
                ft_left=ft_left[:ft_end],
                ft_right=ft_right[:ft_end],
            )
            if input_capture is not None:
                input_capture.append_frame(
                    timestamp_s=float(timestamp_s),
                    rgb=_rgb_uint8_from_any(rgb_frames[idx]),
                    position_m=positions[idx],
                    rotation_axis_angle_rad=rotations[idx],
                    gripper_width_m=float(np.asarray(widths[idx]).reshape(-1)[0]),
                    latest_wrench_timestamp_s=latest_wrench_timestamp_s,
                    context_record=record,
                )
                if not (
                    isinstance(runtime, RGBForceValveContextRuntime)
                    and not record.warmed_up
                ):
                    input_capture.append_classifier_window(
                        classifier_timestamp_s=float(timestamp_s),
                        model_inputs=runtime.get_last_classifier_model_inputs(),
                    )
        if input_capture is not None:
            input_capture.flush()
    if runtime.last_record is None:
        # This is intentionally a strict fallback for test/offline callers;
        # live UmiEnv always provides the camera-rate stream above.
        anchor = float(np.asarray(obs["timestamp"])[-1])
        if input_capture is not None:
            latest_wrench_timestamp_s = input_capture.append_wrenches(
                np.asarray(obs["robot0_ft_left_timestamps"]),
                np.asarray(obs["robot0_ft_left"]),
                np.asarray(obs["robot0_ft_right"]),
                anchor_timestamp_s=anchor,
            )
        record = runtime.predict(
            timestamp_s=anchor,
            rgb=_policy_input_rgb_from_obs(obs),
            position_m=np.asarray(obs["robot0_eef_pos"])[-1],
            rotation_axis_angle_rad=np.asarray(obs["robot0_eef_rot_axis_angle"])[-1],
            gripper_width_m=float(np.asarray(obs["robot0_gripper_width"])[-1].reshape(-1)[0]),
            ft_timestamps=np.asarray(obs["robot0_ft_left_timestamps"]),
            ft_left=np.asarray(obs["robot0_ft_left"]),
            ft_right=np.asarray(obs["robot0_ft_right"]),
        )
        if input_capture is not None:
            input_capture.append_frame(
                timestamp_s=anchor,
                rgb=_policy_input_rgb_from_obs(obs),
                position_m=np.asarray(obs["robot0_eef_pos"])[-1],
                rotation_axis_angle_rad=np.asarray(obs["robot0_eef_rot_axis_angle"])[-1],
                gripper_width_m=float(
                    np.asarray(obs["robot0_gripper_width"])[-1].reshape(-1)[0]
                ),
                latest_wrench_timestamp_s=latest_wrench_timestamp_s,
                context_record=record,
            )
            if not (
                isinstance(runtime, RGBForceValveContextRuntime)
                and not record.warmed_up
            ):
                input_capture.append_classifier_window(
                    classifier_timestamp_s=anchor,
                    model_inputs=runtime.get_last_classifier_model_inputs(),
                )
            input_capture.flush()
    latest = runtime.last_record
    if latest.timestamp_s > float(np.asarray(obs["timestamp"])[-1]) + 1e-6:
        raise ValueError("valve classifier context is newer than policy RGB anchor")
    if not np.isclose(
        latest.timestamp_s, float(np.asarray(obs["timestamp"])[-1]), atol=1e-6
    ):
        raise ValueError(
            "classifier prediction is not aligned to the current policy RGB anchor"
        )
    obs_for_model = dict(obs)
    obs_for_model["valve_context"] = latest.values.reshape(1, -1)
    return obs_for_model, latest


def _add_rgb_force_context_to_policy_observation(
    obs: dict,
    *,
    runtime: RGBForceValveContextRuntime,
    stream: dict,
    input_capture: _ValveContextInputCapture | None = None,
) -> tuple[dict, object]:
    """Run the stateless RGB/F-T observer once for this policy anchor."""
    if not isinstance(runtime, RGBForceValveContextRuntime):
        raise TypeError("policy-rate context path requires RGBForceValveContextRuntime")
    policy_rgb = _policy_input_rgb_from_obs(obs)
    if policy_rgb is None:
        raise ValueError("policy observation has no final camera0_rgb frame")
    anchor = float(np.asarray(obs["timestamp"], dtype=np.float64)[-1])
    rgb_timestamps = np.asarray(
        stream.get("rgb_timestamp_s"), dtype=np.float64
    )
    rgb_frames_raw = np.asarray(stream.get("camera0_rgb"))
    if rgb_frames_raw.ndim != 4 or rgb_frames_raw.shape[-1] < 3:
        raise ValueError(
            "retained camera history must be HWC RGB frames, "
            f"got {rgb_frames_raw.shape}"
        )
    rgb_frames = np.stack(
        [_rgb_uint8_from_any(frame) for frame in rgb_frames_raw]
    )
    ft_timestamps = np.asarray(
        stream.get("ft_timestamp_s"), dtype=np.float64
    )
    ft_left = np.asarray(stream.get("robot0_ft_left"), dtype=np.float32)
    ft_right = np.asarray(stream.get("robot0_ft_right"), dtype=np.float32)

    record = runtime.predict_policy_anchor(
        timestamp_s=anchor,
        rgb=policy_rgb,
        rgb_timestamps=rgb_timestamps,
        rgb_frames=rgb_frames,
        ft_timestamps=ft_timestamps,
        ft_left=ft_left,
        ft_right=ft_right,
    )
    if not np.isclose(record.timestamp_s, anchor, atol=1e-6):
        raise ValueError(
            "RGB/F-T classifier prediction is not aligned to policy RGB anchor"
        )

    if input_capture is not None:
        latest_wrench_timestamp_s = None
        if record.warmed_up:
            model_inputs = runtime.get_last_classifier_model_inputs()
            input_capture.append_source_images(
                model_inputs["rgb_timestamp_s"],
                model_inputs["camera0_rgb"],
            )
            latest_wrench_timestamp_s = input_capture.append_wrenches(
                model_inputs["ft_timestamp_s"],
                model_inputs["robot0_ft_left"],
                model_inputs["robot0_ft_right"],
                anchor_timestamp_s=anchor,
            )
        else:
            ft_end = int(
                np.searchsorted(ft_timestamps, anchor, side="right")
            )
            latest_wrench_timestamp_s = input_capture.append_wrenches(
                ft_timestamps[:ft_end],
                ft_left[:ft_end],
                ft_right[:ft_end],
                anchor_timestamp_s=anchor,
            )
        input_capture.append_frame(
            timestamp_s=anchor,
            rgb=policy_rgb,
            position_m=np.asarray(obs["robot0_eef_pos"])[-1],
            rotation_axis_angle_rad=np.asarray(
                obs["robot0_eef_rot_axis_angle"]
            )[-1],
            gripper_width_m=float(
                np.asarray(obs["robot0_gripper_width"])[-1].reshape(-1)[0]
            ),
            latest_wrench_timestamp_s=latest_wrench_timestamp_s,
            context_record=record,
        )
        if record.warmed_up:
            input_capture.append_classifier_window(
                classifier_timestamp_s=anchor,
                model_inputs=model_inputs,
            )
        input_capture.flush()

    obs_for_model = dict(obs)
    obs_for_model["valve_context"] = record.values.reshape(1, -1)
    return obs_for_model, record


def _policy_input_bgr_from_obs(obs) -> np.ndarray | None:
    img = _policy_input_rgb_from_obs(obs)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _resize_rgb_like_policy(match_rgb, out_hw: tuple[int, int]) -> np.ndarray:
    rgb = _rgb_uint8_from_any(match_rgb)
    oh, ow = out_hw
    ih, iw = rgb.shape[:2]
    if (ih, iw) == (oh, ow):
        return rgb
    tf = get_image_transform(
        input_res=(iw, ih),
        output_res=(ow, oh),
        bgr_to_rgb=False,
    )
    return np.ascontiguousarray(tf(rgb))


def _blend_match_rgb_on_live_bgr(
    live_bgr: np.ndarray,
    match_rgb: np.ndarray,
) -> np.ndarray:
    """Overlay a training RGB frame on a live OpenCV BGR frame at 50/50."""
    resized_match_rgb = _resize_rgb_like_policy(match_rgb, live_bgr.shape[:2])
    match_bgr = cv2.cvtColor(resized_match_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(live_bgr, 0.5, match_bgr, 0.5, 0)


def _show_policy_input_window(obs, label: str, match_rgb=None) -> None:
    live_rgb = _policy_input_rgb_from_obs(obs)
    if live_rgb is None:
        return
    live_bgr = cv2.cvtColor(live_rgb, cv2.COLOR_RGB2BGR)

    if match_rgb is None:
        panel = cv2.resize(live_bgr, (448, 448), interpolation=cv2.INTER_NEAREST)
        _put_text_hud(panel, label, (10, 22), scale=0.5)
        cv2.imshow("policy_input", panel)
        return

    match_rgb = _resize_rgb_like_policy(match_rgb, live_rgb.shape[:2])
    match_bgr = cv2.cvtColor(match_rgb, cv2.COLOR_RGB2BGR)
    overlap = cv2.addWeighted(live_bgr, 0.5, match_bgr, 0.5, 0)

    panels = []
    for title, img in (
        ("train zarr policy image", match_bgr),
        ("live policy input", live_bgr),
        ("overlap 50/50", overlap),
    ):
        this = cv2.resize(img, (336, 336), interpolation=cv2.INTER_NEAREST)
        _put_text_hud(this, title, (8, 22), scale=0.45)
        panels.append(this)
    panel = np.concatenate(panels, axis=1)
    _put_text_hud(panel, label, (8, panel.shape[0] - 10), scale=0.45)
    cv2.imshow("policy_input", panel)


def _overlay_episode_text(vis_bgr: np.ndarray, text: str) -> np.ndarray:
    out = vis_bgr.copy()
    cv2.putText(
        out,
        text,
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        lineType=cv2.LINE_AA,
        thickness=3,
        color=(0, 0, 0),
    )
    cv2.putText(
        out,
        text,
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        thickness=1,
        color=(255, 255, 255),
    )
    return out


# Robot <-> training-dataset frame alignment. _SLAM_FRAME_FIX_P is the old
# sign/axis-only debug path and remains identity. The 4x4 transform below is
# the real robot-from-dataset/tag calibration:
#     T_robot_tcp = T_robot_dataset @ T_dataset_tcp
_SLAM_FRAME_FIX_P = np.eye(3, dtype=np.float64)
_ROBOT_FROM_DATASET_T = np.eye(4, dtype=np.float64)
_DATASET_FROM_ROBOT_T = np.eye(4, dtype=np.float64)


def _set_robot_dataset_transform(transform) -> None:
    global _ROBOT_FROM_DATASET_T, _DATASET_FROM_ROBOT_T
    if transform is None:
        _ROBOT_FROM_DATASET_T = np.eye(4, dtype=np.float64)
        _DATASET_FROM_ROBOT_T = np.eye(4, dtype=np.float64)
        return
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("indy_robot_from_dataset_transform must be a 4x4 matrix")
    if not np.all(np.isfinite(T)):
        raise ValueError("indy_robot_from_dataset_transform contains non-finite values")
    _ROBOT_FROM_DATASET_T = T
    _DATASET_FROM_ROBOT_T = np.linalg.inv(T)


def _transform_pos_rot_with_T(pos, rot, T: np.ndarray):
    pos = np.asarray(pos, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    R_base = np.asarray(T[:3, :3], dtype=np.float64)
    t_base = np.asarray(T[:3, 3], dtype=np.float64)
    new_pos = pos @ R_base.T + t_base

    orig_shape = rot.shape
    rot_mat = st.Rotation.from_rotvec(rot.reshape(-1, 3)).as_matrix()
    rot_mat = np.einsum("ij,tjk->tik", R_base, rot_mat)
    new_rot = st.Rotation.from_matrix(rot_mat).as_rotvec().reshape(orig_shape)
    return new_pos, new_rot


def _transform_tcp7_action(action: np.ndarray, T: np.ndarray, n_robots: int) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(1, -1)
    for r in range(n_robots):
        base = r * 7
        out[:, base:base + 3], out[:, base + 3:base + 6] = (
            _transform_pos_rot_with_T(out[:, base:base + 3], out[:, base + 3:base + 6], T)
        )
    return out


def _match_episode_to_robot_tcp7(
    episode: dict,
    *,
    fallback_gripper_width: float,
    stride: int = 1,
    max_samples: int | None = None,
) -> np.ndarray:
    stride = max(1, int(stride))
    pos = np.asarray(episode["robot0_eef_pos"], dtype=np.float64)
    rot = np.asarray(episode["robot0_eef_rot_axis_angle"], dtype=np.float64)
    n = min(len(pos), len(rot))
    if n <= 0:
        raise ValueError("selected match episode has no TCP samples")
    idx = np.arange(0, n, stride, dtype=np.int64)
    if max_samples is not None and int(max_samples) > 0:
        idx = idx[:int(max_samples)]
    if len(idx) <= 0:
        raise ValueError("selected match episode has no samples after stride/max_samples")

    pos_robot, rot_robot = _transform_pos_rot_with_T(
        pos[idx], rot[idx], _ROBOT_FROM_DATASET_T
    )
    if "robot0_gripper_width" in episode:
        grip = np.asarray(episode["robot0_gripper_width"], dtype=np.float64)
        grip = grip.reshape((len(grip), -1))[idx, :1]
        finite = np.isfinite(grip[:, 0])
        if not np.all(finite):
            grip[~finite, 0] = float(fallback_gripper_width)
    else:
        grip = np.full((len(idx), 1), float(fallback_gripper_width), dtype=np.float64)
    return np.concatenate([pos_robot, rot_robot, grip], axis=-1)


def _apply_slam_frame_fix(raw_action: np.ndarray, n_robots: int) -> np.ndarray:
    """Remap raw model output (pose cols 0:9 of each 10/11-D block) from the
    SLAM training frame to the robot frame via _SLAM_FRAME_FIX_P.
    Position transforms as v' = P @ v. Rotation is encoded as rot6d (the
    first two ROWS of the 3x3 rotation matrix, see pose_util.rot6d_to_mat/
    mat_to_rot6d) and must transform by conjugation R' = P @ R @ P.T so the
    encoded orientation stays consistent with the remapped position frame."""
    P = _SLAM_FRAME_FIX_P
    if raw_action.shape[-1] % n_robots != 0:
        raise ValueError(
            f"action dimension {raw_action.shape[-1]} is not divisible by "
            f"n_robots={n_robots}"
        )
    block_dim = raw_action.shape[-1] // n_robots
    if block_dim not in (10, 11):
        raise ValueError(f"expected a 10-D or 11-D robot action block, got {block_dim}")
    for r in range(n_robots):
        b = r * block_dim
        xyz = raw_action[:, b:b + 3]
        raw_action[:, b:b + 3] = xyz @ P.T

        rot_mat = rot6d_to_mat(raw_action[:, b + 3:b + 9])
        rot_mat = np.einsum('ij,tjk,kl->til', P, rot_mat, P.T)
        raw_action[:, b + 3:b + 9] = mat_to_rot6d(rot_mat)
    return raw_action


def _slam_frame_fix_pos_rot(pos, rot):
    """Convert robot-frame TCP pose to dataset/tag-frame TCP pose."""
    return _transform_pos_rot_with_T(pos, rot, _DATASET_FROM_ROBOT_T)


def _apply_slam_frame_fix_to_obs(obs: dict, n_robots: int) -> dict:
    """Mirror of _apply_slam_frame_fix for the model's INPUT side: convert
    the absolute obs pose (robot frame, from the real controller) into the
    SLAM training frame before computing relative obs, so obs and action
    use the same coordinate convention end to end. Returns a shallow copy;
    the original obs dict (used for logging/visualization/exec baseline)
    is left untouched."""
    obs_fixed = dict(obs)
    for r in range(n_robots):
        pk, rk = f'robot{r}_eef_pos', f'robot{r}_eef_rot_axis_angle'
        obs_fixed[pk], obs_fixed[rk] = _slam_frame_fix_pos_rot(obs[pk], obs[rk])
    return obs_fixed


def _apply_slam_frame_fix_to_start_pose(episode_start_pose):
    """Same fix applied to episode_start_pose (list of pos3+rotvec3 per
    robot) used by get_real_umi_obs_dict's 'wrt_start' relative pose."""
    fixed = []
    for sp in episode_start_pose:
        sp = np.asarray(sp, dtype=np.float64)
        pos, rot = _slam_frame_fix_pos_rot(sp[:3], sp[3:6])
        fixed.append(np.concatenate([pos, rot]))
    return fixed


def _apply_policy_tcp7_rot_roundtrip(
    action: np.ndarray,
    *,
    enabled: bool,
    euler_seq: str,
    euler_extrinsic: bool,
    n_robots: int,
) -> np.ndarray:
    """Per tcp7 block (7 = xyz + rotvec + grip), remap rotvec through task Euler chart."""
    if not enabled:
        return action
    a = np.asarray(action, dtype=np.float64).copy()
    squeeze = a.ndim == 1
    if squeeze:
        a = a[None, :]
    scipy_seq = _scipy_euler_seq(euler_seq, euler_extrinsic)
    for row in range(a.shape[0]):
        for r in range(n_robots):
            b = r * 7
            rv = a[row, b + 3 : b + 6]
            rot = st.Rotation.from_rotvec(rv)
            euler = rot.as_euler(scipy_seq, degrees=False)
            a[row, b + 3 : b + 6] = st.Rotation.from_euler(
                scipy_seq, euler, degrees=False
            ).as_rotvec()
    return a[0] if squeeze else a


def _human_teleop_compose_rotvec(
    prev_rotvec: np.ndarray,
    drot_xyz: np.ndarray,
    euler_seq: str,
    euler_extrinsic: bool,
) -> np.ndarray:
    """Same composition as keyboard / SpaceMouse (was hard-coded 'xyz')."""
    scipy_seq = _scipy_euler_seq(euler_seq, euler_extrinsic)
    drot = st.Rotation.from_euler(scipy_seq, drot_xyz, degrees=False)
    return (drot * st.Rotation.from_rotvec(prev_rotvec)).as_rotvec()


def _print_ckpt_pose_eval_contract(cfg):
    """Item (2)(3): what this ckpt commits to for obs/action pose decoding."""
    print("[pose_eval_audit] cfg.task.pose_repr (from checkpoint):")
    if hasattr(cfg.task, "pose_repr"):
        print(OmegaConf.to_yaml(cfg.task.pose_repr))
    else:
        print("  (missing cfg.task.pose_repr)")


def _print_pose_z_audit(
    obs,
    action_tcp7,
    action_pose_repr: str,
    iter_idx,
    n_robots: int,
    tag: str,
    *,
    dataset_z_stats=None,
    raw_action_pred=None,
):
    """
    Item (1): compare live obs TCP (m) to decoded policy waypoints (m).
    Large |xyz| suggests mm/m confusion; monotonic +delta_z suggests model bias.
    """
    obs_tcp = np.concatenate(
        [
            obs["robot0_eef_pos"][-1],
            obs["robot0_eef_rot_axis_angle"][-1],
        ]
    )
    obs_xyz = obs_tcp[:3]
    a = np.asarray(action_tcp7, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    print(f"{tag} action_pose_repr={action_pose_repr!r}")
    print(
        "  obs_tcp xyz(m) [last in horizon]:",
        np.array2string(obs_xyz, precision=5),
    )
    print(
        "  obs |xyz|_inf (m):",
        float(np.max(np.abs(obs_xyz))),
        "(typical single-arm workspace < ~1.5 m; >>3 may hint wrong units)",
    )
    if dataset_z_stats is not None:
        oz = float(obs_xyz[2])
        p5 = dataset_z_stats["pos_z_p5"]
        p50 = dataset_z_stats["pos_z_p50"]
        p95 = dataset_z_stats["pos_z_p95"]
        print(
            "  vs train robot0_eef_pos z (subsampled): "
            f"obs_z - train_p50 = {oz - p50:.5f} m; train p5/p50/p95 = "
            f"{p5:.5f} / {p50:.5f} / {p95:.5f}"
        )
    if raw_action_pred is not None:
        ra = np.asarray(raw_action_pred, dtype=np.float64)
        if ra.ndim == 2 and ra.shape[-1] >= 3:
            rz = ra[:, 2]
            print(
                "  model action_pred[:,2] (pose10d; not SI tcp7): "
                f"min {float(rz.min()):.5f}, max {float(rz.max()):.5f}, mean {float(rz.mean()):.5f}"
            )
            if ra.shape[-1] >= 10:
                rg = ra[:, 9]
                print(
                    "  model action_pred[:,9] (grip channel raw): "
                    f"min {float(rg.min()):.5f}, max {float(rg.max()):.5f}"
                )
    for r in range(n_robots):
        blk = a[:, r * 7 : r * 7 + 3]
        z = blk[:, 2]
        dz = z - float(obs_xyz[2])
        print(f"  robot{r} action chunk rows={a.shape[0]}")
        print(
            "    action z (m): min",
            f"{float(z.min()):.5f}, max {float(z.max()):.5f}, mean {float(z.mean()):.5f}",
        )
        print(
            "    delta z vs obs (m): min",
            f"{float(dz.min()):.5f}, max {float(dz.max()):.5f}, mean {float(dz.mean()):.5f}",
        )
        if blk.shape[0] >= 2:
            step = np.diff(blk[:, 2])
            print(
                "    per-row dz along horizon:",
                np.array2string(step, precision=5),
            )
        if dataset_z_stats is not None and blk.size:
            zm = float(np.mean(blk[:, 2]))
            p50 = dataset_z_stats["pos_z_p50"]
            print(
                f"    decoded mean z vs train_p50: {zm - p50:.5f} m "
                "(SI tcp after get_real_umi_action)"
            )


def _print_model_input_debug(
    obs_dict_np,
    env_obs,
    episode_start_pose,
    obs_pose_repr: str,
    tag: str,
):
    """
    Raw env TCP vs dict passed to policy.predict_action (output of get_real_umi_obs_dict).

    For obs_pose_repr='relative', each horizon row is expressed w.r.t. **the last**
    robot sample in that horizon (see real_inference_util.get_real_umi_obs_dict).
    The last row's pose10d *position* slice is therefore ~0 by construction, not
    "wrong model input" and not comparable to world-frame demo z from the dataset.
    """
    print(f"[model_input] {tag} obs_pose_repr={obs_pose_repr!r}")
    rawp = np.asarray(env_obs["robot0_eef_pos"][-1], dtype=np.float64)
    rawr = np.asarray(env_obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64)
    print(
        "  env raw robot0_eef_pos[-1] xyz (m, TCP):",
        np.array2string(rawp, precision=5),
        f"| z={float(rawp[2]):.5f}",
    )
    print(
        "  env raw robot0_eef_rot_axis_angle[-1]:",
        np.array2string(rawr, precision=5),
    )
    if episode_start_pose is not None and len(episode_start_pose) > 0:
        sp = np.asarray(episode_start_pose[0], dtype=np.float64).ravel()
        print(
            "  episode_start_pose tcp6 (for wrt_start):",
            np.array2string(sp, precision=5),
            f"| z={float(sp[2]):.5f}",
        )
    if str(obs_pose_repr).lower() == "relative":
        print(
            "  note: policy_obs['robot0_eef_pos'] is pose10d *position* after "
            "inv(T_last) @ T_t per horizon row. Last row ≈ 0 is expected; "
            "earlier rows show motion within the obs window vs current pose."
        )
    for key in sorted(obs_dict_np.keys()):
        v = obs_dict_np[key]
        va = np.asarray(v)
        if "rgb" in key.lower() or va.ndim >= 4:
            print(f"  policy_obs[{key!r}]: shape={va.shape} dtype={va.dtype} (tensor omitted)")
            continue
        print(f"  policy_obs[{key!r}]: shape={va.shape} dtype={va.dtype}")
        if va.ndim >= 2:
            for ti in range(va.shape[0]):
                row = np.asarray(va[ti], dtype=np.float64).ravel()
                print(f"      row[{ti}]:", np.array2string(row, precision=5, max_line_width=120))
        else:
            print("      ", np.array2string(va.ravel(), precision=5))
        if va.ndim >= 2 and "eef_pos" in key and va.shape[-1] >= 3:
            zcol = np.asarray(va[:, 2], dtype=np.float64)
            print(
                "      col[2] over time dim:",
                f"min {float(zcol.min()):.5f} max {float(zcol.max()):.5f}",
            )
    if str(obs_pose_repr).lower() == "relative":
        print(
            "  compare to training: use the same pipeline on zarr rows "
            "(robot0_eef_pos after get_real_umi_obs_dict), not raw demo_start_pose z alone."
        )


def _check_finite_array(name: str, arr, *, max_rows: int = 3) -> None:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if np.all(finite):
        return

    bad = np.argwhere(~finite)
    lines = [
        f"{name} contains non-finite values: shape={a.shape} dtype={a.dtype}",
        f"  first bad indices: {bad[:10].tolist()}",
    ]
    if a.ndim >= 2:
        for i in range(min(max_rows, a.shape[0])):
            lines.append(
                f"  row[{i}]: "
                + np.array2string(
                    np.asarray(a[i]).ravel(),
                    precision=5,
                    max_line_width=160,
                )
            )
    else:
        lines.append(
            "  values: "
            + np.array2string(a.ravel()[:40], precision=5, max_line_width=160)
        )
    raise click.ClickException("\n".join(lines))


def _check_policy_inputs_finite(obs_dict_np, tag: str) -> None:
    for key, value in obs_dict_np.items():
        _check_finite_array(f"{tag} policy input {key!r}", value)


def _policy_obs_float32(obs_dict_np: dict) -> dict:
    """Match the dataset's float32 model-input dtype without normalizing."""
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in obs_dict_np.items()
    }


def _format_timing_stats_ms(samples) -> str:
    """Compact p50/p95/max timing report without retaining raw log samples."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return "n/a"
    return (
        f"p50={np.percentile(values, 50) * 1000.0:.3f}ms "
        f"p95={np.percentile(values, 95) * 1000.0:.3f}ms "
        f"max={np.max(values) * 1000.0:.3f}ms"
    )


def _runtime_cycle_counts(runtime_metrics: dict) -> dict:
    """Return cycle counters with non-overlapping, non-negative meanings."""
    attempted = int(runtime_metrics["attempted_cycles"])
    completed = int(runtime_metrics["completed_cycles"])
    valid_observations = int(runtime_metrics["valid_observations"])
    safety_rejections = int(runtime_metrics["safety_rejections"])
    context_recovery_skips = int(runtime_metrics.get("context_recovery_skips", 0))
    if not 0 <= completed <= valid_observations <= attempted:
        raise ValueError(
            "runtime cycle counters must satisfy "
            "completed <= valid_observations <= attempted"
        )
    if not 0 <= safety_rejections <= attempted - completed:
        raise ValueError(
            "runtime safety rejections cannot exceed incomplete attempts"
        )
    if not 0 <= context_recovery_skips <= attempted - completed:
        raise ValueError(
            "runtime context recovery skips cannot exceed incomplete attempts"
        )
    if safety_rejections + context_recovery_skips > attempted - completed:
        raise ValueError(
            "runtime rejected/skipped cycles exceed incomplete attempts"
        )
    return {
        "cycles": attempted,
        "completed_cycles": completed,
        "valid_observation_cycles": valid_observations,
        "dropped_cycles": attempted - valid_observations,
        "safety_rejected_cycles": safety_rejections,
        "context_recovery_skipped_cycles": context_recovery_skips,
    }


def _register_context_recovery_skip(
    env,
    runtime_metrics: dict,
    *,
    plan_only: bool,
    consecutive_skips: int,
    reason: str,
    max_consecutive_skips: int = 3,
) -> int:
    """Hold pending motion and bound retries without using a stale policy anchor."""
    next_consecutive = int(consecutive_skips) + 1
    if not plan_only:
        env.hold_robot()
    if next_consecutive > int(max_consecutive_skips):
        raise PolicySafetyError(
            "valve-context recovery exceeded "
            f"{max_consecutive_skips} consecutive policy cycles: {reason}"
        )
    runtime_metrics["context_recovery_skips"] += 1
    print(
        "[context recovery] "
        f"{reason}; policy inference/command skipped "
        f"({next_consecutive}/{max_consecutive_skips})."
    )
    return next_consecutive


def _exec_actions_with_fresh_ft_guard(
    env,
    actions,
    action_timestamps,
    startup_bias_12d,
    ft_safety_cfg,
):
    """Validate a newly read F/T snapshot, then submit without intervening I/O."""
    _, measured_grasp_force_n, sample_age_s = read_and_validate_latest_ft(
        env,
        startup_bias_12d,
        ft_safety_cfg,
    )
    env.exec_actions(
        actions=actions,
        timestamps=action_timestamps,
        compensate_latency=False,
    )
    return measured_grasp_force_n, sample_age_s


def _get_warmup_observation_with_retry(
    env,
    *,
    timeout_s=3.0,
    retry_interval_s=0.02,
    monotonic_func=None,
    sleep_func=None,
):
    """Wait briefly for a fresh startup observation before any policy motion."""
    timeout_s = float(timeout_s)
    retry_interval_s = float(retry_interval_s)
    if timeout_s <= 0.0 or retry_interval_s <= 0.0:
        raise ValueError("warmup retry timeout and interval must be positive")
    if monotonic_func is None:
        monotonic_func = time.monotonic
    if sleep_func is None:
        sleep_func = time.sleep

    deadline = monotonic_func() + timeout_s
    stale_retries = 0
    while True:
        try:
            obs = env.get_obs(include_valve_context_stream=False)
            if stale_retries:
                print(
                    "[warmup] fresh dual-F/T observation recovered after "
                    f"{stale_retries} stale retries."
                )
            return obs
        except FTObservationStaleError as exc:
            stale_retries += 1
            remaining_s = deadline - monotonic_func()
            if remaining_s <= 0.0:
                raise FTObservationStaleError(
                    "policy warmup did not receive a fresh dual-F/T observation "
                    f"within {timeout_s:.2f}s after {stale_retries} retries; "
                    f"last error: {exc}"
                ) from exc
            if stale_retries == 1:
                print(
                    "[warmup] transient stale dual-F/T observation; waiting up to "
                    f"{timeout_s:.2f}s for a fresh camera-aligned sample."
                )
            sleep_func(min(retry_interval_s, remaining_s))


def _array_minmax_str(arr) -> str:
    a = np.asarray(arr)
    if a.size == 0:
        return "empty"
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return "all non-finite"
    return f"{float(finite.min()):.6g}..{float(finite.max()):.6g}"


def _expected_policy_rgb_tchw_from_env(env_obs, shape_meta, key="camera0_rgb"):
    imgs = np.asarray(env_obs[key])
    t, hi, wi, ci = imgs.shape
    co, ho, wo = shape_meta["obs"][key]["shape"]
    if ci != co:
        raise ValueError(f"{key} channel mismatch: env={ci}, shape_meta={co}")
    out_imgs = imgs
    if (ho != hi) or (wo != wi) or (imgs.dtype == np.uint8):
        tf = get_image_transform(
            input_res=(wi, hi),
            output_res=(wo, ho),
            bgr_to_rgb=False,
        )
        out_imgs = np.stack([tf(x) for x in imgs])
        if imgs.dtype == np.uint8:
            out_imgs = out_imgs.astype(np.float32) / 255.0
    return np.moveaxis(out_imgs, -1, 1)


def _print_policy_image_audit(
    env_obs,
    obs_dict_np,
    shape_meta,
    tag: str,
    *,
    train_rgb=None,
    train_info=None,
) -> None:
    key = "camera0_rgb"
    if key not in env_obs or key not in obs_dict_np:
        print(f"[policy_image_audit] {tag}: camera0_rgb unavailable")
        return
    env_img = np.asarray(env_obs[key])
    policy_img = np.asarray(obs_dict_np[key])
    expected = _expected_policy_rgb_tchw_from_env(env_obs, shape_meta, key=key)
    diff = np.asarray(policy_img, dtype=np.float32) - np.asarray(expected, dtype=np.float32)
    shape_cfg = tuple(int(x) for x in shape_meta["obs"][key]["shape"])
    horizon_cfg = int(shape_meta["obs"][key].get("horizon", env_img.shape[0]))
    print(f"[policy_image_audit] {tag}")
    print(
        f"  shape_meta {key}: CHW={shape_cfg} horizon={horizon_cfg}"
    )
    print(
        f"  env_obs {key}: THWC shape={env_img.shape} dtype={env_img.dtype} "
        f"range={_array_minmax_str(env_img)}"
    )
    print(
        f"  policy_obs {key}: TCHW shape={policy_img.shape} dtype={policy_img.dtype} "
        f"range={_array_minmax_str(policy_img)}"
    )
    print(
        "  env_obs -> policy_obs max_abs_diff:",
        f"{float(np.max(np.abs(diff))):.9g}",
    )
    if train_info is not None:
        print(
            "  train zarr camera0_rgb:",
            f"shape={train_info.get('shape')} dtype={train_info.get('dtype')} "
            f"episodes={train_info.get('n_episodes')}",
        )
    if train_rgb is not None:
        train_rgb_u8 = _rgb_uint8_from_any(train_rgb)
        print(
            "  selected train policy frame:",
            f"HWC shape={train_rgb_u8.shape} dtype={train_rgb_u8.dtype} "
            f"range={_array_minmax_str(train_rgb_u8)}"
        )
        train_env = {key: train_rgb_u8[None]}
        train_tchw = _expected_policy_rgb_tchw_from_env(train_env, shape_meta, key=key)
        print(
            "  selected train frame as model tensor:",
            f"TCHW shape={train_tchw.shape} dtype={train_tchw.dtype} "
            f"range={_array_minmax_str(train_tchw)}"
        )


def _rotvec_distances(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    return (st.Rotation.from_rotvec(a) * st.Rotation.from_rotvec(b).inv()).magnitude()


def _print_coord_transform_audit(
    tag: str,
    obs,
    obs_for_model,
    *,
    action_dataset=None,
    action_robot=None,
    match_debug_data=None,
    match_source_idx: int | None = None,
) -> None:
    live_tcp6 = np.concatenate(
        [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]]
    ).astype(np.float64)
    model_tcp6 = np.concatenate(
        [
            obs_for_model["robot0_eef_pos"][-1],
            obs_for_model["robot0_eef_rot_axis_angle"][-1],
        ]
    ).astype(np.float64)
    rt_pos, rt_rot = _transform_pos_rot_with_T(
        model_tcp6[:3], model_tcp6[3:6], _ROBOT_FROM_DATASET_T
    )
    rt_tcp6 = np.concatenate([rt_pos, rt_rot])
    pos_err = float(np.linalg.norm(rt_tcp6[:3] - live_tcp6[:3]))
    rot_err = float(_rotvec_distances(rt_tcp6[3:6], live_tcp6[3:6])[0])

    print(f"[coord_transform_audit] {tag}")
    print(
        "  T_robot_from_dataset:",
        np.array2string(_ROBOT_FROM_DATASET_T, precision=5, max_line_width=160),
    )
    print(
        "  live robot tcp6:",
        np.array2string(live_tcp6, precision=5, max_line_width=160),
    )
    print(
        "  model-side dataset tcp6:",
        np.array2string(model_tcp6, precision=5, max_line_width=160),
    )
    print(
        "  dataset->robot roundtrip tcp6:",
        np.array2string(rt_tcp6, precision=5, max_line_width=160),
    )
    print(
        "  obs transform roundtrip error:",
        f"pos={pos_err:.9g} m rot={rot_err:.9g} rad",
    )

    if match_debug_data is not None and match_source_idx is not None:
        idx = int(np.clip(match_source_idx, 0, len(match_debug_data["raw_pose6"]) - 1))
        raw_tcp6 = np.asarray(match_debug_data["raw_pose6"][idx], dtype=np.float64)
        robot_tcp6 = np.asarray(match_debug_data["robot_pose6"][idx], dtype=np.float64)
        dpos = robot_tcp6[:3] - live_tcp6[:3]
        drot = float(_rotvec_distances(robot_tcp6[3:6], live_tcp6[3:6])[0])
        print(
            f"  zarr sample ep={match_debug_data['episode']} frame={idx} dataset tcp6:",
            np.array2string(raw_tcp6, precision=5, max_line_width=160),
        )
        print(
            "  zarr sample mapped robot tcp6:",
            np.array2string(robot_tcp6, precision=5, max_line_width=160),
        )
        print(
            "  mapped zarr vs live robot gap:",
            np.array2string(dpos, precision=5, max_line_width=160),
            f"|d|={float(np.linalg.norm(dpos)):.9g} m rot={drot:.9g} rad",
        )

    if action_dataset is not None and action_robot is not None:
        ad = np.asarray(action_dataset, dtype=np.float64)
        ar = np.asarray(action_robot, dtype=np.float64)
        if ad.ndim == 1:
            ad = ad[None]
        if ar.ndim == 1:
            ar = ar[None]
        n = min(len(ad), len(ar))
        back_pos, back_rot = _transform_pos_rot_with_T(
            ar[:n, :3], ar[:n, 3:6], _DATASET_FROM_ROBOT_T
        )
        pos_e = np.linalg.norm(back_pos - ad[:n, :3], axis=1)
        rot_e = _rotvec_distances(back_rot, ad[:n, 3:6])
        print(
            "  action dataset->robot->dataset roundtrip max error:",
            f"pos={float(pos_e.max()):.9g} m rot={float(rot_e.max()):.9g} rad",
        )


def _decode_real_umi_action_checked(raw_action, obs, action_pose_repr: str, tag: str):
    _check_finite_array(f"{tag} raw action_pred", raw_action)
    decoder_action = np.asarray(raw_action)
    if decoder_action.shape[-1] % 11 == 0:
        n_robot_blocks = decoder_action.shape[-1] // 11
        blocks = decoder_action.reshape(*decoder_action.shape[:-1], n_robot_blocks, 11)
        # get_real_umi_action understands pose9 + width1. The predicted force
        # remains available in raw_action for the bounded width-feedback path,
        # and is never passed to the width-only hardware scheduler.
        decoder_action = blocks[..., :10].reshape(
            *decoder_action.shape[:-1], n_robot_blocks * 10
        )
    try:
        action = get_real_umi_action(decoder_action, obs, action_pose_repr)
    except np.linalg.LinAlgError as exc:
        raw = np.asarray(raw_action, dtype=np.float64)
        lines = [
            f"{tag} failed to decode raw action_pred into TCP action: {exc}",
            f"  shape={raw.shape} action_pose_repr={action_pose_repr!r}",
        ]
        if raw.ndim >= 2 and raw.shape[-1] >= 9:
            for i in range(min(3, raw.shape[0])):
                row = raw[i]
                pos = row[:3]
                rot6d = row[3:9]
                lines.append(
                    f"  row[{i}] pos="
                    + np.array2string(pos, precision=5, max_line_width=120)
                    + " rot6d="
                    + np.array2string(rot6d, precision=5, max_line_width=120)
                    + f" rot6d_norm={float(np.linalg.norm(rot6d)):.5g}"
                )
        raise click.ClickException("\n".join(lines)) from exc
    _check_finite_array(f"{tag} decoded tcp7 action", action)
    return action


def _print_motion_debug(
    tag: str,
    obs,
    target_poses: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    n_robots: int = 1,
):
    """Current robot TCP vs waypoints about to be sent to exec_actions."""
    cur = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64),
            np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64),
        ]
    )
    targets = np.asarray(target_poses, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    print(f"{tag} motion debug (world TCP, meters + rotvec rad)")
    print(
        "  current xyz(m):",
        np.array2string(cur[:3], precision=5),
        " rotvec:",
        np.array2string(cur[3:6], precision=5),
    )
    r_cur = st.Rotation.from_rotvec(cur[3:6])
    n_show = min(3, targets.shape[0])
    for i in range(n_show):
        row = targets[i]
        tcp6 = row[:6]
        grip = float(row[6]) if row.size > 6 else 0.0
        dpos = tcp6[:3] - cur[:3]
        r_tgt = st.Rotation.from_rotvec(tcp6[3:6])
        drot_rad = (r_tgt * r_cur.inv()).magnitude()
        when = ""
        if timestamps is not None and len(timestamps) > i:
            when = f"  sched_in={float(timestamps[i]) - time.time():.3f}s"
        print(
            f"  next[{i}] xyz(m):",
            np.array2string(tcp6[:3], precision=5),
            " rotvec:",
            np.array2string(tcp6[3:6], precision=5),
            f" grip(m)={grip:.5f}{when}",
        )
        print(
            "    delta xyz(m):",
            np.array2string(dpos, precision=5),
            f"|d|={float(np.linalg.norm(dpos)):.5f}",
            f" delta_rot={drot_rad:.5f} rad",
        )
    if targets.shape[0] > n_show:
        print(f"  ... {targets.shape[0]} waypoints total")
    if n_robots > 1:
        print(f"  (n_robots={n_robots}; only robot0 shown)")


def _resolve_match_dataset_paths(match_dataset: str) -> tuple[str, pathlib.Path]:
    """Resolve zarr path + session dir for videos.

    Accepts:
    - a .zarr.zip file (training dataset, e.g. dataset.zarr.zip)
    - a session folder with replay_buffer.zarr/ or dataset.zarr.zip inside
    """
    match_path = pathlib.Path(os.path.expanduser(match_dataset)).resolve()
    if match_path.is_file():
        name = match_path.name.lower()
        if name.endswith(".zarr.zip") or name.endswith(".zip"):
            return str(match_path), match_path.parent
        raise FileNotFoundError(
            f"--match_dataset file must be .zarr.zip, got: {match_path}"
        )
    if match_path.is_dir():
        for name in ("replay_buffer.zarr", "dataset.zarr.zip"):
            cand = match_path.joinpath(name)
            if cand.exists():
                return str(cand), match_path
        raise FileNotFoundError(
            f"--match_dataset folder needs replay_buffer.zarr or dataset.zarr.zip: {match_path}"
        )
    raise FileNotFoundError(f"--match_dataset not found: {match_path}")


_MATCH_POSE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
)


class _MatchPoseReplayBuffer:
    """Pose-only zarr reader for --match_dataset (`g` key).

    Training zarr stores camera frames as JPEG-XL, which needs imagecodecs.
    For matching start pose we only need TCP + gripper lowdim arrays.
    """

    def __init__(self, zarr_path: str):
        import zarr

        zarr_path = os.path.expanduser(zarr_path)
        self._zip_store = None
        if zarr_path.endswith(".zarr.zip") or (
            zarr_path.endswith(".zip") and not zarr_path.endswith(".zarr")
        ):
            self._zip_store = zarr.ZipStore(zarr_path, mode="r")
            root = zarr.open_group(store=self._zip_store, mode="r")
        else:
            root = zarr.open_group(zarr_path, mode="r")
        self.episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        self._arrays: dict[str, object] = {}
        for key in _MATCH_POSE_KEYS:
            if key in root["data"]:
                self._arrays[key] = root["data"][key]
        missing = [
            k for k in ("robot0_eef_pos", "robot0_eef_rot_axis_angle")
            if k not in self._arrays
        ]
        if missing:
            avail = list(root["data"].keys())
            raise KeyError(
                f"match_dataset missing required pose keys {missing}; "
                f"available data keys: {avail}"
            )

    @property
    def n_episodes(self) -> int:
        return int(len(self.episode_ends))

    def get_episode(self, idx: int, copy: bool = False) -> dict:
        if idx < 0 or idx >= self.n_episodes:
            raise IndexError(
                f"episode idx {idx} out of range [0, {self.n_episodes})"
            )
        start = 0 if idx == 0 else int(self.episode_ends[idx - 1])
        end = int(self.episode_ends[idx])
        result = {}
        for key, arr in self._arrays.items():
            x = np.asarray(arr[start:end])
            if copy:
                x = x.copy()
            result[key] = x
        return result

    def close(self) -> None:
        if self._zip_store is not None:
            self._zip_store.close()
            self._zip_store = None


def _load_match_replay_buffer(zarr_path: str) -> _MatchPoseReplayBuffer:
    buf = _MatchPoseReplayBuffer(zarr_path)
    print(
        f"[match_dataset] pose-only load OK: {buf.n_episodes} episodes "
        "(skipped JPEG-XL image arrays; imagecodecs not required for g key)"
    )
    if str(zarr_path).endswith(".zarr.zip"):
        print(
            "[match_dataset] NOTE: training zarr poses are SLAM/tag-frame "
            "(GoPro+ORB-SLAM), not Indy robot TCP. Do not expect g to move "
            "the robot to a physically correct pose unless you use "
            "--match_g_move_robot (usually wrong for Indy eval)."
        )
    return buf


def _load_zarr_episode_first_policy_frames(zarr_path: str):
    try:
        import imagecodecs.numcodecs as _icn
        _icn.register_codecs()
    except Exception:
        pass
    import zarr as _zarr

    store = None
    if str(zarr_path).endswith(".zip"):
        store = _zarr.ZipStore(str(zarr_path), mode="r")
        root = _zarr.open_group(store=store, mode="r")
    else:
        root = _zarr.open_group(str(zarr_path), mode="r")

    try:
        if "camera0_rgb" not in root["data"]:
            return {}, None
        ee = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        ep_starts = np.concatenate([[0], ee[:-1]])
        img_arr = root["data"]["camera0_rgb"]
        frames = {
            int(ep_idx): np.asarray(img_arr[int(start)])
            for ep_idx, start in enumerate(ep_starts)
        }
        info = {
            "shape": tuple(int(x) for x in img_arr.shape),
            "dtype": str(img_arr.dtype),
            "n_episodes": int(len(ep_starts)),
        }
        print(
            "[match_dataset] training policy images:",
            f"camera0_rgb shape={info['shape']} dtype={info['dtype']} "
            f"episodes={info['n_episodes']}",
        )
        return frames, info
    finally:
        if store is not None:
            store.close()


def _make_match_pose_debug_data(match_replay_buffer, episode_idx: int | None):
    if match_replay_buffer is None or episode_idx is None:
        return None
    ep_idx = int(episode_idx)
    ep = match_replay_buffer.get_episode(ep_idx)
    raw_pose6 = np.concatenate(
        [
            np.asarray(ep["robot0_eef_pos"], dtype=np.float64),
            np.asarray(ep["robot0_eef_rot_axis_angle"], dtype=np.float64),
        ],
        axis=-1,
    )
    robot_pos, robot_rot = _transform_pos_rot_with_T(
        raw_pose6[:, :3], raw_pose6[:, 3:6], _ROBOT_FROM_DATASET_T
    )
    return {
        "episode": ep_idx,
        "raw_pose6": raw_pose6,
        "robot_pose6": np.concatenate([robot_pos, robot_rot], axis=-1),
    }


def _print_match_pose_compare(
    episode_idx: int,
    zarr_tcp6: np.ndarray,
    live_tcp6: np.ndarray,
    *,
    will_move: bool,
) -> None:
    zarr_tcp6 = np.asarray(zarr_tcp6, dtype=np.float64).ravel()[:6]
    live_tcp6 = np.asarray(live_tcp6, dtype=np.float64).ravel()[:6]
    dpos = zarr_tcp6[:3] - live_tcp6[:3]
    r_live = st.Rotation.from_rotvec(live_tcp6[3:6])
    r_zarr = st.Rotation.from_rotvec(zarr_tcp6[3:6])
    drot = (r_zarr * r_live.inv()).magnitude()
    zarr_robot_pos, zarr_robot_rot = _transform_pos_rot_with_T(
        zarr_tcp6[:3], zarr_tcp6[3:6], _ROBOT_FROM_DATASET_T
    )
    zarr_robot_tcp6 = np.concatenate([zarr_robot_pos, zarr_robot_rot])
    dpos_cal = zarr_robot_tcp6[:3] - live_tcp6[:3]
    r_zarr_robot = st.Rotation.from_rotvec(zarr_robot_tcp6[3:6])
    drot_cal = (r_zarr_robot * r_live.inv()).magnitude()
    print(f"[match g] episode={episode_idx}")
    print(
        "  zarr tcp6 (SLAM training frame):",
        np.array2string(zarr_tcp6, precision=5),
    )
    print(
        "  live tcp6 (Indy robot now):",
        np.array2string(live_tcp6, precision=5),
    )
    print(
        "  gap xyz(m):",
        np.array2string(dpos, precision=5),
        f"|d|={float(np.linalg.norm(dpos)):.5f}",
        f" gap_rot={drot:.5f} rad",
    )
    if not np.allclose(_ROBOT_FROM_DATASET_T, np.eye(4)):
        print(
            "  zarr mapped to robot by indy_robot_from_dataset_transform:",
            np.array2string(zarr_robot_tcp6, precision=5),
        )
        print(
            "  calibrated gap xyz(m):",
            np.array2string(dpos_cal, precision=5),
            f"|d|={float(np.linalg.norm(dpos_cal)):.5f}",
            f" gap_rot={drot_cal:.5f} rad",
        )
    if will_move:
        print(
            "  → moving robot to zarr pose (--match_g_move_robot). "
            "Often wrong for SLAM-trained ckpt on Indy."
        )
    else:
        print(
            "  → robot NOT moved. Align with keyboard teleop + live camera, then press c. "
            "Add --match_g_move_robot only if you know zarr poses are robot TCP."
        )


def _parse_tcp_delta_scales(spec: str | None) -> np.ndarray | None:
    if spec is None:
        return None
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError("--tcp_delta_scales expects three comma-separated values, e.g. 1,0,0")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _limit_policy_waypoints(
    target_poses: np.ndarray,
    obs,
    *,
    n_robots: int = 1,
    tcp_delta_scales: np.ndarray | None = None,
    action_scale: float = 1.0,
    freeze_rotation: bool = False,
    freeze_rotation_ref_pose=None,
) -> np.ndarray:
    """Shrink / axis-mask policy waypoints relative to current TCP (debug only)."""
    out = np.asarray(target_poses, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(1, -1)
    scale = float(action_scale)
    for r in range(n_robots):
        base = r * 7
        cur_pos = np.asarray(obs[f"robot{r}_eef_pos"][-1], dtype=np.float64)
        cur_rot = np.asarray(obs[f"robot{r}_eef_rot_axis_angle"][-1], dtype=np.float64)
        freeze_rot = cur_rot
        if freeze_rotation_ref_pose is not None:
            freeze_rot = np.asarray(freeze_rotation_ref_pose[r], dtype=np.float64)[3:6]
        r_cur = st.Rotation.from_rotvec(cur_rot)
        axis_mask = tcp_delta_scales if tcp_delta_scales is not None else np.ones(3)
        for i in range(out.shape[0]):
            delta = (out[i, base:base + 3] - cur_pos) * axis_mask * scale
            out[i, base:base + 3] = cur_pos + delta
            if freeze_rotation:
                out[i, base + 3:base + 6] = freeze_rot
            elif scale != 1.0:
                r_tgt = st.Rotation.from_rotvec(out[i, base + 3:base + 6])
                drot = (r_tgt * r_cur.inv()).as_rotvec() * scale
                out[i, base + 3:base + 6] = (st.Rotation.from_rotvec(drot) * r_cur).as_rotvec()
    return out


def _apply_policy_motion_momentum(
    target_poses: np.ndarray,
    *,
    current_pose_width7: np.ndarray,
    previous_weight: float,
) -> np.ndarray:
    """Smooth one policy horizon from the latest measured TCP/gripper state.

    ``previous_weight=1/3`` implements the requested 1:2 ratio: one part of
    the previous value and two parts of the newly predicted value. TCP
    position increments are blended sequentially within this horizon, and
    rotation increments use the local SO(3) tangent (rotation-vector) space
    before composition.

    TCP state always starts from the current measured pose with zero retained
    increment. A final waypoint from the preceding horizon is only scheduled,
    not guaranteed to have been executed; carrying it into this cycle can run
    the software pose ahead of the physical arm and amplify a tiny policy
    rotation into a safety violation.

    Gripper width instead blends the measured width at this policy cycle with
    each F/T-corrected target. The measured anchor is intentionally fixed
    across the horizon: a future scheduled width may not have been executed
    when the next policy cycle starts, so retaining it would make the command
    run ahead of the physical gripper. This smooths without velocity
    extrapolation or target overshoot. Waypoint safety validates the result.
    """
    weight = float(previous_weight)
    if not 0.0 <= weight < 1.0:
        raise ValueError("motion momentum previous_weight must be in [0, 1)")
    out = np.asarray(target_poses, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 7:
        raise ValueError(
            "motion momentum requires single-arm TCP7 targets, got "
            f"{out.shape}"
        )
    current = np.asarray(current_pose_width7, dtype=np.float64).reshape(7)
    if not np.isfinite(current).all() or not np.isfinite(out).all():
        raise ValueError("motion momentum requires finite TCP/gripper inputs")

    reference = current.copy()
    previous_delta = np.zeros(7, dtype=np.float64)
    current_weight = 1.0 - weight
    measured_width = float(current[6])
    for waypoint_idx in range(len(out)):
        policy_target = out[waypoint_idx]
        policy_delta_pos = policy_target[:3] - reference[:3]
        policy_delta_rotvec = (
            st.Rotation.from_rotvec(policy_target[3:6])
            * st.Rotation.from_rotvec(reference[3:6]).inv()
        ).as_rotvec()
        previous_width = float(reference[6])
        blended_width = (
            weight * measured_width + current_weight * policy_target[6]
        )
        blended_delta = np.concatenate(
            [
                weight * previous_delta[:3] + current_weight * policy_delta_pos,
                weight * previous_delta[3:6]
                + current_weight * policy_delta_rotvec,
                [blended_width - previous_width],
            ]
        )
        reference[:3] += blended_delta[:3]
        reference[3:6] = (
            st.Rotation.from_rotvec(blended_delta[3:6])
            * st.Rotation.from_rotvec(reference[3:6])
        ).as_rotvec()
        reference[6] = blended_width
        out[waypoint_idx] = reference
        previous_delta = blended_delta
    return out


def _print_policy_action_debug(tag, raw_action, action_7d, submitted=None):
    """raw_action: model action_pred (e.g. T x 10). action_7d: after get_real_umi_action (T x 7)."""
    print(f"{tag} raw_action_pred shape={raw_action.shape} dtype={raw_action.dtype}")
    r = np.asarray(raw_action)
    if r.ndim >= 2:
        for i in range(min(3, r.shape[0])):
            print(f"  raw[{i}]:", np.array2string(r[i], precision=5))
        if r.shape[0] > 3:
            print(f"  ... ({r.shape[0]} rows total)")
    else:
        print("  raw:", np.array2string(r, precision=5))
    a = np.asarray(action_7d)
    print(f"{tag} after get_real_umi_action shape={a.shape} (xyz m + rotvec rad + grip m)")
    if a.ndim >= 2:
        for i in range(min(3, a.shape[0])):
            print(f"  tcp7[{i}]:", np.array2string(a[i], precision=5))
        if a.shape[0] > 3:
            print(f"  ... ({a.shape[0]} rows total)")
    else:
        print("  tcp7:", np.array2string(a, precision=5))
    if submitted is not None:
        s = np.asarray(submitted)
        print(f"{tag} submitted to exec_actions shape={s.shape}")
        for i in range(min(3, s.shape[0])):
            print(f"  exec[{i}]:", np.array2string(s[i], precision=5))
        if s.shape[0] > 3:
            print(f"  ... ({s.shape[0]} rows total)")


_POSE10D_LABELS = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5", "grip", "grasp_N"]
_FT_CHANNEL_LABELS = ("fx", "fy", "fz", "tx", "ty", "tz")
_FT_CHANNEL_UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
_PANEL_W = 320
_PANEL_H = 420   # taller than wide: text panels have ~19-22 lines, 320 clipped them


def _tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _normalized_ft_policy_inputs(policy, obs_dict) -> dict[str, np.ndarray]:
    """Return the two F/T histories after the checkpoint normalizer.

    ``obs_dict`` is the same tensor dictionary passed to ``predict_action``.
    The operation is cheap (no vision/diffusion forward pass) and makes the
    recorded CSV explicitly distinguish physical sensor units from model input.
    """
    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None:
        return {}
    try:
        normalized = normalizer.normalize(obs_dict)
    except Exception as exc:
        print(f"[eval_log] could not normalize F/T diagnostic input: {exc}")
        return {}
    return {
        key: _tensor_to_numpy(normalized[key])[0]
        for key in ("robot0_ft_left", "robot0_ft_right")
        if key in normalized
    }


def _write_ft_input_history(
    writer,
    *,
    iter_idx: int,
    wall_time: float,
    obs: dict,
    obs_dict_np: dict,
    normalized_ft: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Save every causal 6-D F/T sample used by one policy decision."""
    histories = []
    for side in ("left", "right"):
        key = f"robot0_ft_{side}"
        values = np.asarray(obs_dict_np.get(key), dtype=np.float64)
        if values.ndim != 2 or values.shape[-1] != 6:
            return None
        normalized = np.asarray(normalized_ft.get(key), dtype=np.float64)
        if normalized.shape != values.shape:
            normalized = np.full_like(values, np.nan, dtype=np.float64)
        timestamps = np.asarray(
            obs.get(f"robot0_ft_{side}_timestamps", np.full(len(values), np.nan)),
            dtype=np.float64,
        ).reshape(-1)
        if len(timestamps) != len(values):
            timestamps = np.full(len(values), np.nan, dtype=np.float64)
        for sample_idx, (timestamp, raw_row, normalized_row) in enumerate(
            zip(timestamps, values, normalized)
        ):
            writer.writerow(
                [iter_idx, wall_time, side, sample_idx, int(sample_idx == len(values) - 1), timestamp]
                + raw_row.tolist()
                + normalized_row.tolist()
            )
        histories.append(values)
    return tuple(histories)


def _render_ft_input_timeline(path: pathlib.Path, rows: list[dict]) -> bool:
    """Render latest causal F/T sample per policy iteration without matplotlib."""
    if not rows:
        return False
    left = np.asarray([row["left"] for row in rows], dtype=np.float64)
    right = np.asarray([row["right"] for row in rows], dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 6 or right.shape != left.shape:
        return False

    width, height = 1440, 900
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Policy F/T input timeline (latest causal sample per inference)",
        (28, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "blue: left finger   red: right finger   values before checkpoint normalization",
        (28, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    n_rows = len(rows)
    pad_x, top, gap_x, gap_y = 52, 90, 34, 55
    panel_w = (width - 2 * pad_x - 2 * gap_x) // 3
    panel_h = (height - top - 52 - gap_y) // 2
    for channel_idx, (label, unit) in enumerate(zip(_FT_CHANNEL_LABELS, _FT_CHANNEL_UNITS)):
        col, row = channel_idx % 3, channel_idx // 3
        x0 = pad_x + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        x1, y1 = x0 + panel_w, y0 + panel_h
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (160, 160, 160), 1)
        values = np.concatenate([left[:, channel_idx], right[:, channel_idx]])
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = float(finite.min()), float(finite.max())
            margin = max(1e-6, 0.08 * max(hi - lo, 1e-3))
            lo, hi = lo - margin, hi + margin
        cv2.putText(
            canvas, f"{label} [{unit}]  {lo:.3g} .. {hi:.3g}",
            (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (30, 30, 30), 1, cv2.LINE_AA,
        )
        for frac in (0.25, 0.5, 0.75):
            y = int(y0 + frac * panel_h)
            cv2.line(canvas, (x0, y), (x1, y), (225, 225, 225), 1)
        def point(i, value):
            x = x0 if n_rows <= 1 else int(x0 + i * panel_w / (n_rows - 1))
            y = int(y1 - (float(value) - lo) * panel_h / (hi - lo))
            return x, int(np.clip(y, y0, y1))
        for series, color in ((left[:, channel_idx], (210, 80, 30)), (right[:, channel_idx], (35, 35, 210))):
            valid_idx = np.flatnonzero(np.isfinite(series))
            for i0, i1 in zip(valid_idx[:-1], valid_idx[1:]):
                if i1 == i0 + 1:
                    cv2.line(canvas, point(i0, series[i0]), point(i1, series[i1]), color, 2)
    return bool(cv2.imwrite(str(path), canvas))


def _render_fusion_attention_heatmap(
    path: pathlib.Path, token_names: list[str], attention: np.ndarray
) -> bool:
    """Render mean query-to-key fusion attention; rows=query, columns=key."""
    matrix = np.asarray(attention, dtype=np.float64)
    n_tokens = len(token_names)
    if matrix.shape != (n_tokens, n_tokens) or not np.all(np.isfinite(matrix)):
        return False
    cell, left, top = 145, 220, 125
    canvas = np.full((top + cell * n_tokens + 90, left + cell * n_tokens + 45, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Mean fusion self-attention (query row -> key column)", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "descriptive attention only; not causal feature attribution", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)
    for idx, name in enumerate(token_names):
        cv2.putText(canvas, name, (left + idx * cell + 4, top - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (8, top + idx * cell + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    for q_idx in range(n_tokens):
        for k_idx in range(n_tokens):
            value = float(np.clip(matrix[q_idx, k_idx], 0.0, 1.0))
            color = cv2.applyColorMap(
                np.array([[int(round(value * 255.0))]], dtype=np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )[0, 0].tolist()
            x0, y0 = left + k_idx * cell, top + q_idx * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (210, 210, 210), 1)
            text_color = (0, 0, 0) if value > 0.55 else (255, 255, 255)
            cv2.putText(canvas, f"{value:.3f}", (x0 + 35, y0 + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    return bool(cv2.imwrite(str(path), canvas))


def _render_text_panel(lines, width=_PANEL_W, height=_PANEL_H, bg_color=(30, 30, 30)):
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    y = 16
    line_h = 15
    for line in lines:
        if y > height - 4:
            break  # safety net; line budget below is sized to not hit this
        cv2.putText(panel, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return panel


def _tcp6_to_xyzrpy_lines(prefix: str, tcp6) -> list[str]:
    tcp6 = np.asarray(tcp6, dtype=np.float64).reshape(-1)[:6]
    rpy = st.Rotation.from_rotvec(tcp6[3:6]).as_euler("xyz", degrees=True)
    return [
        prefix,
        f"  x: {tcp6[0]:+.5f} m",
        f"  y: {tcp6[1]:+.5f} m",
        f"  z: {tcp6[2]:+.5f} m",
        f"  roll : {rpy[0]:+.2f} deg",
        f"  pitch: {rpy[1]:+.2f} deg",
        f"  yaw  : {rpy[2]:+.2f} deg",
    ]


def _load_match_episode_debug_data(zarr_path: str | None, episode_idx: int | None):
    if zarr_path is None or episode_idx is None:
        return None
    try:
        try:
            import imagecodecs.numcodecs as _icn
            _icn.register_codecs()
        except Exception:
            pass
        import zarr as _zarr
        store = None
        if str(zarr_path).endswith(".zip"):
            store = _zarr.ZipStore(str(zarr_path), mode="r")
            root = _zarr.open_group(store=store, mode="r")
        else:
            root = _zarr.open_group(str(zarr_path), mode="r")
        ee = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        if not (0 <= int(episode_idx) < len(ee)):
            raise IndexError(f"episode {episode_idx} out of range [0, {len(ee)})")
        start = 0 if int(episode_idx) == 0 else int(ee[int(episode_idx) - 1])
        end = int(ee[int(episode_idx)])
        raw_pose6 = np.concatenate(
            [
                np.asarray(root["data"]["robot0_eef_pos"][start:end], dtype=np.float64),
                np.asarray(root["data"]["robot0_eef_rot_axis_angle"][start:end], dtype=np.float64),
            ],
            axis=-1,
        )
        robot_pos, robot_rot = _transform_pos_rot_with_T(
            raw_pose6[:, :3], raw_pose6[:, 3:6], _ROBOT_FROM_DATASET_T
        )
        robot_pose6 = np.concatenate([robot_pos, robot_rot], axis=-1)
        rgb = None
        if "camera0_rgb" in root["data"]:
            try:
                rgb = np.asarray(root["data"]["camera0_rgb"][start:end])
            except Exception as exc:
                print(
                    "[eval_log] original zarr video frames unavailable "
                    f"(coordinates will still be shown): {exc}"
                )
        if store is not None:
            store.close()
        return {
            "episode": int(episode_idx),
            "raw_pose6": raw_pose6,
            "robot_pose6": robot_pose6,
            "rgb": rgb,
            "fps": 59.94,
        }
    except Exception as exc:
        print(f"[eval_log] failed to load original match episode debug data: {exc}")
        return None


def _render_policy_output_video_panel(
    raw_action_h0,
    decoded_action_h0,
    scheduled_action_h0,
    *,
    predicted_force_n: float | None,
    measured_force_n: float | None,
    width_correction_m: float | None,
    valve_context_record=None,
    width: int,
    height: int,
) -> np.ndarray:
    """Compact first-waypoint output readout for the comparison video."""
    lines = ["4. policy output, horizon 0 (pre-safety)"]
    raw = None if raw_action_h0 is None else np.asarray(raw_action_h0, dtype=np.float64).ravel()
    decoded = (
        None if decoded_action_h0 is None
        else np.asarray(decoded_action_h0, dtype=np.float64).ravel()
    )
    scheduled = (
        None if scheduled_action_h0 is None
        else np.asarray(scheduled_action_h0, dtype=np.float64).ravel()
    )
    if raw is None or raw.size < 11:
        lines.append("model output: waiting for first inference")
    else:
        lines.extend([
            f"raw xyz: {raw[0]:+.4f} {raw[1]:+.4f} {raw[2]:+.4f}",
            "raw R6D: " + " ".join(f"{x:+.3f}" for x in raw[3:9]),
            f"raw grip={raw[9]:+.4f}  grasp_N={raw[10]:+.3f}",
        ])
    if decoded is not None and decoded.size >= 7:
        lines.extend([
            "decoded TCP (before scale/F/T):",
            f" xyz: {decoded[0]:+.4f} {decoded[1]:+.4f} {decoded[2]:+.4f} m",
            " rotvec: " + " ".join(f"{x:+.3f}" for x in decoded[3:6]),
            f" width: {decoded[6]:+.4f} m",
        ])
    if scheduled is not None and scheduled.size >= 7:
        lines.extend([
            "candidate after scale/F/T:",
            f" xyz: {scheduled[0]:+.4f} {scheduled[1]:+.4f} {scheduled[2]:+.4f} m",
            " rotvec: " + " ".join(f"{x:+.3f}" for x in scheduled[3:6]),
            f" width: {scheduled[6]:+.4f} m",
        ])
    if predicted_force_n is not None or measured_force_n is not None:
        force_text = "F/T grasp N: "
        force_text += "pred=" + (
            "n/a" if predicted_force_n is None else f"{predicted_force_n:+.3f}"
        )
        force_text += " meas=" + (
            "n/a" if measured_force_n is None else f"{measured_force_n:+.3f}"
        )
        lines.append(force_text)
    if width_correction_m is not None:
        lines.append(f"F/T width correction: {width_correction_m * 1000.0:+.3f} mm")
    if valve_context_record is not None:
        context = np.asarray(valve_context_record.values, dtype=np.float64)
        schema = str(
            getattr(valve_context_record, "schema", VALVE_CONTEXT_V1_SCHEMA)
        )
        spec = valve_context_spec(schema)
        lines.extend([
            f"valve context ({schema}):",
            f" phase={valve_context_record.phase_name} "
            f"reason={valve_context_record.error_reason_name} "
            f"warm={int(valve_context_record.warmed_up)}",
            " p=" + " ".join(
                f"{value:.2f}" for value in context[:len(spec["phase_names"])]
            ),
        ])
    return _render_text_panel(lines, width=width, height=height)


def _render_fusion_attention_video_panel(
    attention,
    token_names: list[str],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Small live query-to-key attention heatmap for comparison.mp4."""
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(
        panel, "5. live fusion attention (heads mean)", (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, "row=query, column=key; descriptive, not causal", (6, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA,
    )
    names = [str(name).replace("camera0_", "").replace("robot0_", "") for name in token_names]
    matrix = None if attention is None else np.asarray(attention, dtype=np.float64)
    n_tokens = len(names)
    if (
        n_tokens == 0
        or matrix is None
        or matrix.shape != (n_tokens, n_tokens)
        or not np.all(np.isfinite(matrix))
    ):
        cv2.putText(
            panel, "attention: waiting / capture disabled", (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1, cv2.LINE_AA,
        )
        return panel

    left, top = 96, 57
    cell = min(72, max(38, (width - left - 8) // n_tokens), max(32, (height - top - 50) // n_tokens))
    for idx, name in enumerate(names):
        short_name = name.replace("_", " ")[:11]
        cv2.putText(
            panel, short_name, (left + idx * cell + 2, top - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1, cv2.LINE_AA,
        )
        cv2.putText(
            panel, short_name, (4, top + idx * cell + cell // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1, cv2.LINE_AA,
        )
    for query_idx in range(n_tokens):
        for key_idx in range(n_tokens):
            value = float(np.clip(matrix[query_idx, key_idx], 0.0, 1.0))
            color = cv2.applyColorMap(
                np.array([[int(round(value * 255.0))]], dtype=np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )[0, 0].tolist()
            x0, y0 = left + key_idx * cell, top + query_idx * cell
            cv2.rectangle(panel, (x0, y0), (x0 + cell, y0 + cell), color, -1)
            cv2.rectangle(panel, (x0, y0), (x0 + cell, y0 + cell), (235, 235, 235), 1)
            cv2.putText(
                panel, f"{value:.2f}", (x0 + 4, y0 + cell // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (20, 20, 20) if value > 0.55 else (255, 255, 255),
                1, cv2.LINE_AA,
            )
    key_mean = matrix.mean(axis=0)
    cv2.putText(
        panel,
        "key mean: " + " ".join(
            f"{name[:6]}={value:.2f}" for name, value in zip(names, key_mean)
        ),
        (6, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
        (200, 255, 200), 1, cv2.LINE_AA,
    )
    return panel


def _render_valve_context_video_panel(
    valve_context_record,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the exact frozen-classifier context supplied to the policy."""
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(
        panel, "6. valve context fed to policy", (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if valve_context_record is None:
        cv2.putText(
            panel, "context: waiting for classifier", (8, 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1, cv2.LINE_AA,
        )
        return panel

    values = np.asarray(valve_context_record.values, dtype=np.float64).reshape(-1)
    schema = str(
        getattr(valve_context_record, "schema", VALVE_CONTEXT_V1_SCHEMA)
    )
    try:
        spec = valve_context_spec(schema)
    except ValueError:
        spec = None
    if (
        spec is None
        or values.shape != (int(spec["dim"]),)
        or not np.all(np.isfinite(values))
    ):
        cv2.putText(
            panel, "context: invalid diagnostic value", (8, 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA,
        )
        return panel
    cv2.putText(
        panel,
        f"phase={valve_context_record.phase_name}  "
        f"reason={valve_context_record.error_reason_name}  "
        f"warm={int(valve_context_record.warmed_up)}",
        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (200, 255, 200), 1, cv2.LINE_AA,
    )
    phases = tuple(spec["phase_names"])
    reasons = tuple(spec["reason_names"])
    bar_x, bar_w, bar_h = 92, max(40, width - 106), 11
    y = 61
    for label, value in zip(phases, values[:len(phases)]):
        cv2.putText(panel, label, (7, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (65, 65, 65), -1)
        cv2.rectangle(
            panel, (bar_x, y),
            (bar_x + int(round(bar_w * float(np.clip(value, 0.0, 1.0)))), y + bar_h),
            (70, 185, 255), -1,
        )
        cv2.putText(panel, f"{value:.3f}", (bar_x + 4, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 0), 1, cv2.LINE_AA)
        y += 14
    if reasons:
        cv2.putText(panel, "error reason probabilities", (7, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 150), 1, cv2.LINE_AA)
        y += 16
        phase_count = len(phases)
        for label, value in zip(
            reasons, values[phase_count:phase_count + len(reasons)]
        ):
            cv2.putText(panel, label, (7, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (65, 65, 65), -1)
            cv2.rectangle(
                panel, (bar_x, y),
                (bar_x + int(round(bar_w * float(np.clip(value, 0.0, 1.0)))), y + bar_h),
                (120, 120, 255), -1,
            )
            cv2.putText(panel, f"{value:.3f}", (bar_x + 4, y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 0), 1, cv2.LINE_AA)
            y += 14
    return panel


def _draw_ft_force_history(
    panel: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    y0: int,
    height: int,
) -> None:
    """Draw physical Fx/Fy/Fz traces from the exact 32-sample policy input."""
    values = np.asarray(values, dtype=np.float64)
    panel_h, panel_w = panel.shape[:2]
    x0, x1 = 40, panel_w - 7
    y1 = min(panel_h - 5, y0 + height)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 6:
        cv2.putText(panel, f"{title}: unavailable", (7, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (100, 180, 255), 1, cv2.LINE_AA)
        return
    force = values[:, :3]
    if not np.all(np.isfinite(force)):
        cv2.putText(panel, f"{title}: non-finite", (7, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80, 80, 255), 1, cv2.LINE_AA)
        return
    low, high = float(force.min()), float(force.max())
    pad = max(0.25, (high - low) * 0.10)
    low, high = low - pad, high + pad
    if high - low < 1e-9:
        low, high = low - 0.25, high + 0.25
    cv2.putText(panel, f"{title} force N   oldest -> latest", (7, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)
    graph_top, graph_bottom = y0 + 18, y1 - 23
    cv2.rectangle(panel, (x0, graph_top), (x1, graph_bottom), (105, 105, 105), 1)
    zero_y = int(round(graph_bottom - (0.0 - low) / (high - low) * (graph_bottom - graph_top)))
    if graph_top <= zero_y <= graph_bottom:
        cv2.line(panel, (x0, zero_y), (x1, zero_y), (85, 85, 85), 1)
    colors = ((80, 80, 255), (80, 255, 80), (255, 180, 50))  # Fx, Fy, Fz (BGR)
    for channel_idx, (label, color) in enumerate(zip(("Fx", "Fy", "Fz"), colors)):
        points = []
        for index, value in enumerate(force[:, channel_idx]):
            x = x0 + int(round((x1 - x0) * index / max(len(force) - 1, 1)))
            y = graph_bottom - int(round((value - low) / (high - low) * (graph_bottom - graph_top)))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        elif points:
            cv2.circle(panel, points[0], 2, color, -1)
        cv2.putText(panel, f"{label}={force[-1, channel_idx]:+.2f}",
                    (x0 + channel_idx * max(58, (x1 - x0) // 3), y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{high:+.1f}", (2, graph_top + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{low:+.1f}", (2, graph_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 180, 180), 1, cv2.LINE_AA)


def _render_ft_input_video_panel(obs, *, width: int, height: int) -> np.ndarray:
    """Render physical, startup-bias-corrected F/T values fed to the policy."""
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(
        panel, "7. policy F/T input (physical, corrected)", (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if obs is None:
        return panel
    left = np.asarray(obs.get("robot0_ft_left", []), dtype=np.float64)
    right = np.asarray(obs.get("robot0_ft_right", []), dtype=np.float64)
    graph_height = max(52, (height - 27) // 2)
    _draw_ft_force_history(panel, left, title="left", y0=23, height=graph_height)
    _draw_ft_force_history(
        panel, right, title="right", y0=23 + graph_height, height=graph_height
    )
    return panel


def _render_eval_video_frame(
    original_rgb,
    current_bgr,
    original_tcp6,
    robot_tcp6,
    live_tcp6,
    *,
    source_idx: int | None,
    match_episode_id: int | None,
    raw_action_h0=None,
    decoded_action_h0=None,
    scheduled_action_h0=None,
    predicted_force_n: float | None = None,
    measured_force_n: float | None = None,
    width_correction_m: float | None = None,
    valve_context_record=None,
    ft_observation=None,
    fusion_attention=None,
    fusion_attention_tokens: list[str] | None = None,
):
    """Video with matching images, coordinates, live output, and attention."""
    if original_rgb is None:
        left = np.full((_PANEL_H, _PANEL_W, 3), (20, 20, 20), dtype=np.uint8)
        left = _overlay_episode_text(left, "1. original video unavailable")
    else:
        img = np.asarray(original_rgb)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        left = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (_PANEL_W, _PANEL_H))
        title = "1. original video"
        if match_episode_id is not None:
            title += f" ep={match_episode_id}"
        if source_idx is not None:
            title += f" frame={source_idx}"
        left = _overlay_episode_text(left, title)

    if current_bgr is None:
        middle = np.full((_PANEL_H, _PANEL_W, 3), (20, 20, 20), dtype=np.uint8)
        middle = _overlay_episode_text(middle, "2. current policy input unavailable")
    else:
        middle = cv2.resize(current_bgr, (_PANEL_W, _PANEL_H))
        middle = _overlay_episode_text(middle, "2. current policy input")

    coord_lines = ["3. original coordinate (xyz/rpy)"]
    if original_tcp6 is not None:
        coord_lines += _tcp6_to_xyzrpy_lines("raw zarr/tag frame:", original_tcp6)
    else:
        coord_lines.append("raw zarr/tag frame: unavailable")
    if robot_tcp6 is not None:
        coord_lines += _tcp6_to_xyzrpy_lines("mapped robot frame:", robot_tcp6)
    if live_tcp6 is not None:
        coord_lines += _tcp6_to_xyzrpy_lines("live robot now:", live_tcp6)
    right = _render_text_panel(coord_lines)

    sep = np.full((_PANEL_H, 4, 3), (255, 255, 255), dtype=np.uint8)
    top_row = np.concatenate([left, sep, middle, sep, right], axis=1)

    bottom_h = 250
    output_w = top_row.shape[1] // 2 - 2
    attention_w = top_row.shape[1] - output_w - 4
    output_panel = _render_policy_output_video_panel(
        raw_action_h0,
        decoded_action_h0,
        scheduled_action_h0,
        predicted_force_n=predicted_force_n,
        measured_force_n=measured_force_n,
        width_correction_m=width_correction_m,
        valve_context_record=valve_context_record,
        width=output_w,
        height=bottom_h,
    )
    attention_panel = _render_fusion_attention_video_panel(
        fusion_attention,
        [] if fusion_attention_tokens is None else fusion_attention_tokens,
        width=attention_w,
        height=bottom_h,
    )
    horizontal_sep = np.full((4, top_row.shape[1], 3), (255, 255, 255), dtype=np.uint8)
    bottom_sep = np.full((bottom_h, 4, 3), (255, 255, 255), dtype=np.uint8)
    bottom_row = np.concatenate([output_panel, bottom_sep, attention_panel], axis=1)

    # Keep output/attention intact, then add the actual context and physical
    # F/T histories below so one comparison frame is sufficient for diagnosis.
    diagnostic_h = 190
    context_w = top_row.shape[1] // 2 - 2
    ft_w = top_row.shape[1] - context_w - 4
    context_panel = _render_valve_context_video_panel(
        valve_context_record, width=context_w, height=diagnostic_h
    )
    ft_panel = _render_ft_input_video_panel(
        ft_observation, width=ft_w, height=diagnostic_h
    )
    diagnostic_sep = np.full((diagnostic_h, 4, 3), (255, 255, 255), dtype=np.uint8)
    diagnostic_row = np.concatenate([context_panel, diagnostic_sep, ft_panel], axis=1)
    second_horizontal_sep = np.full(
        (4, top_row.shape[1], 3), (255, 255, 255), dtype=np.uint8
    )
    return np.concatenate(
        [top_row, horizontal_sep, bottom_row, second_horizontal_sep, diagnostic_row],
        axis=0,
    )


def _make_eval_comparison_frame(
    obs,
    match_debug_data,
    source_idx: int | None,
    *,
    env: UmiEnv | None = None,
    camera_idx: int = 0,
    raw_action_h0=None,
    decoded_action_h0=None,
    scheduled_action_h0=None,
    predicted_force_n: float | None = None,
    measured_force_n: float | None = None,
    width_correction_m: float | None = None,
    valve_context_record=None,
    fusion_attention=None,
    fusion_attention_tokens: list[str] | None = None,
):
    """Build one fixed-size comparison-video frame from a live observation."""
    original_rgb = None
    original_tcp6 = None
    robot_tcp6 = None
    match_episode_id = None
    if match_debug_data is not None:
        n_src = len(match_debug_data["raw_pose6"])
        if n_src > 0:
            source_idx = int(np.clip(
                0 if source_idx is None else source_idx, 0, n_src - 1
            ))
            original_tcp6 = match_debug_data["raw_pose6"][source_idx]
            robot_tcp6 = match_debug_data["robot_pose6"][source_idx]
            if match_debug_data.get("rgb") is not None:
                original_rgb = match_debug_data["rgb"][source_idx]
            match_episode_id = match_debug_data["episode"]
        else:
            source_idx = None
    else:
        source_idx = None
    current_bgr = _policy_input_bgr_from_obs(obs)
    if current_bgr is None and env is not None:
        try:
            current_bgr = _get_live_display_bgr(env, camera_idx=camera_idx)
        except Exception as exc:
            print(f"[eval_log] could not obtain live fallback video frame: {exc}")
    return _render_eval_video_frame(
        original_rgb,
        current_bgr,
        original_tcp6,
        robot_tcp6,
        _tcp6_from_obs(obs),
        source_idx=source_idx,
        match_episode_id=match_episode_id,
        raw_action_h0=raw_action_h0,
        decoded_action_h0=decoded_action_h0,
        scheduled_action_h0=scheduled_action_h0,
        predicted_force_n=predicted_force_n,
        measured_force_n=measured_force_n,
        width_correction_m=width_correction_m,
        valve_context_record=valve_context_record,
        ft_observation=obs,
        fusion_attention=fusion_attention,
        fusion_attention_tokens=fusion_attention_tokens,
    )


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Training session folder, or path to dataset.zarr.zip / replay_buffer.zarr (for g / episode overlay)')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option(
    "--match_replay_stride",
    default=1,
    type=int,
    show_default=True,
    help="On v: replay every Nth sample from the selected match episode.",
)
@click.option(
    "--match_replay_max_samples",
    default=0,
    type=int,
    show_default=True,
    help="On v: maximum selected-episode samples to replay; <=0 means all.",
)
@click.option(
    "--match_replay_duration_scale",
    default=3.0,
    type=float,
    show_default=True,
    help="On v: slow down selected-episode replay by this factor.",
)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option(
    '--steps_per_inference', '-si', default=None, type=int,
    help=(
        "Number of predicted actions to execute before replanning. Defaults "
        "to checkpoint execution.n_action_steps (dual-F/T default: 2)."
    ),
)
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=None, type=float,
    help="Control frequency in Hz. Defaults to checkpoint action frequency.")
@click.option(
    '--device',
    default='auto',
    show_default=True,
    help="Inference device: auto, cpu, cuda, or cuda:N.",
)
@click.option(
    "--valve_classifier_checkpoint",
    default=None,
    help=(
        "Frozen valve context observer checkpoint. By default, use the path "
        "serialized by the action-policy checkpoint (v1 falls back to final.pt)."
    ),
)
@click.option(
    "--valve_classifier_device",
    default="auto",
    show_default=True,
    help="Classifier device: auto (same as policy), cpu, cuda, or cuda:N.",
)
@click.option(
    "--allow_valve_classifier_override",
    is_flag=True,
    default=False,
    help=(
        "Explicitly replace the observer serialized by a v2 action checkpoint "
        "with a compatible-output RGB/native-F/T checkpoint. This bypasses only "
        "the observer SHA identity check; phase order and 5-D policy contract "
        "remain strict."
    ),
)
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--no_spacemouse', is_flag=True, default=True, help="Disable SpaceMouse and use keyboard teleop only.")
@click.option('--no_gripper', is_flag=True, default=False, help="Run without connecting to gripper hardware.")
@click.option(
    "--direct_dynamixel_gripper/--no_direct_dynamixel_gripper",
    default=False,
    show_default=True,
    help=(
        "Legacy compatibility only. RG2-FT eval should leave this disabled."
    ),
)
@click.option(
    "--dynamixel_gripper_config",
    default=_DEFAULT_DYNAMIXEL_GRIPPER_CONFIG,
    show_default=True,
    help="Waypoint YAML containing the current Dynamixel gripper open/close ticks.",
)
@click.option(
    "--gripper_calib_zarr",
    default=_DEFAULT_GRIPPER_CALIB_ZARR,
    show_default=True,
    help="Training zarr used to calibrate model gripper width min/max.",
)
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=_DEFAULT_SIM_FOV)
@click.option('-ci', '--camera_intrinsics', type=str, default=_DEFAULT_CAMERA_INTRINSICS)
@click.option(
    "--eval_image_mask/--no_eval_image_mask",
    default=True,
    show_default=True,
    help="Apply the same predefined gripper/mirror mask used when generating zarr images.",
)
@click.option(
    "--policy_image_crop_ratio",
    default=1.0,
    type=float,
    show_default=True,
    help=(
        "Center-crop ratio before resizing live policy image to the checkpoint "
        "resolution. 1.0 matches the zarr generator; <1 zooms in when live FOV is too wide."
    ),
)
@click.option(
    "--inpaint_aruco_tags/--no_inpaint_aruco_tags",
    default=True,
    show_default=True,
    help="Detect ArUco tags in live camera frames and inpaint them before policy input.",
)
@click.option(
    "--aruco_config",
    default=_DEFAULT_ARUCO_CONFIG,
    show_default=True,
    help="Aruco config YAML used for live eval tag inpainting.",
)
@click.option(
    "--disable_eval_image_aug/--keep_eval_image_aug",
    default=True,
    show_default=True,
    help="Replace policy image RandomCrop/ColorJitter transforms with Identity during eval.",
)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option(
    "--print_policy_output",
    is_flag=True,
    default=False,
    help="Print policy action_pred (raw) and get_real_umi_action output each inference.",
)
@click.option(
    "--pose_eval_audit",
    is_flag=True,
    default=False,
    help=(
        "Print ckpt pose_repr once, then each policy step: obs TCP xyz vs "
        "decoded action xyz (esp. z deltas) to check frame/units vs model bias."
    ),
)
@click.option(
    "--dataset_zarr",
    default=None,
    type=str,
    help=(
        "Training replay zarr path, or 'auto' to resolve cfg.task.dataset.dataset_path "
        "near the ckpt/cwd. If omitted but --pose_eval_audit is set, auto-resolve is tried."
    ),
)
@click.option(
    "--dataset_z_stride",
    default=20,
    type=int,
    show_default=True,
    help="Stride when subsampling zarr for dataset z / action stats.",
)
@click.option(
    "--print_motion_debug",
    is_flag=True,
    default=False,
    help=(
        "Each policy step: print current TCP pose and the next waypoint(s) "
        "sent to exec_actions, with xyz/rot deltas."
    ),
)
@click.option(
    "--vis_pose",
    is_flag=True,
    default=False,
    help=(
        "On the OpenCV camera window: overlay current TCP xyz, next waypoint, "
        "delta vs episode start, and a small top-down XY localization map."
    ),
)
@click.option(
    "--print_model_input",
    is_flag=True,
    default=False,
    help=(
        "After get_real_umi_obs_dict: print raw env TCP vs policy_obs tensors "
        "(input to predict_action)."
    ),
)
@click.option(
    "--show_policy_image",
    is_flag=True,
    default=False,
    help=(
        "Show the exact camera0_rgb frame used by the policy in a separate "
        "OpenCV window for crop/FOV/lens checks."
    ),
)
@click.option(
    "--policy_input_audit",
    is_flag=True,
    default=False,
    help=(
        "Print camera0_rgb shape/range and verify env_obs image equals the "
        "tensor passed to policy.predict_action."
    ),
)
@click.option(
    "--save_fusion_attention/--no_save_fusion_attention",
    default=False,
    show_default=True,
    help=(
        "Save the Dual-F/T fusion layer's per-head 4x4 self-attention to the "
        "eval log. This is descriptive attention, not causal attribution."
    ),
)
@click.option(
    "--save_context_inputs/--no_save_context_inputs",
    default=True,
    show_default=True,
    help=(
        "For a valve-context checkpoint, save every exact classifier RGB frame "
        "(lossless PNG), corrected F/T sample, TCP lowdim input, and classifier "
        "output under eval_logs/ep*/context_inputs/."
    ),
)
@click.option(
    "--save_policy_inputs/--no_save_policy_inputs",
    default=True,
    show_default=True,
    help=(
        "Save the exact post-preprocessing NumPy observation dictionary passed "
        "to policy.predict_action: RGB, relative TCP/rotation-6D, causal "
        "left/right F/T, and valve_context when required by the checkpoint."
    ),
)
@click.option(
    "--coord_transform_audit",
    is_flag=True,
    default=False,
    help=(
        "Print dataset<->robot TCP transform roundtrip checks for obs/actions "
        "and the selected zarr match episode."
    ),
)
@click.option(
    "--zero_ft_on_start/--no_zero_ft_on_start",
    default=True,
    show_default=True,
    help=(
        "Software-tare both RG2-FT finger sensors at startup from recent raw "
        "samples. Keep the unloaded gripper still while the program starts."
    ),
)
@click.option(
    "--ft_zero_samples",
    default=25,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of recent 100 Hz RG2-FT samples averaged for startup tare.",
)
@click.option(
    "--ft_max_age_sec",
    default=0.020,
    type=float,
    show_default=True,
    help=(
        "Maximum age of the latest causal RG2-FT sample at an RGB anchor. "
        "20 ms permits normal 100 Hz transport jitter; values must be positive."
    ),
)
@click.option(
    "--max_policy_iters",
    "-mpi",
    default=None,
    type=int,
    help="Stop policy after this many inference cycles (e.g. 1 for a single step).",
)
@click.option(
    "--plan_only",
    is_flag=True,
    default=False,
    help=(
        "Suppress waypoint submission from this script. This still starts the "
        "robot controller and is not a passive/no-protocol sensor mode."
    ),
)
@click.option(
    "--tcp_delta_scales",
    default=None,
    type=str,
    help="Scale each axis of position delta vs current TCP, e.g. 1,0,0 for +X only.",
)
@click.option(
    "--action_scale",
    default=1.0,
    type=float,
    show_default=True,
    help="Scale position delta magnitude. Rotation is enabled unless --freeze_rotation is set.",
)
@click.option(
    "--motion_momentum_previous_weight",
    default=0.0,
    type=float,
    show_default=True,
    help=(
        "Blend TCP increments within each policy horizon and gripper targets as "
        "previous*weight + policy*(1-weight). "
        "Use 0.333333 for the requested 1:2 previous:current momentum; "
        "each policy cycle is re-anchored to measured robot state."
    ),
)
@click.option(
    "--freeze_rotation/--allow_rotation",
    default=False,
    show_default=True,
    help="Freeze or execute policy-predicted TCP orientation; rotation is enabled by default.",
)
@click.option(
    "--match_g_move_robot",
    is_flag=True,
    default=False,
    help=(
        "On g: actually move the robot to the zarr episode start TCP. "
        "Default off — SLAM training poses are not Indy absolute TCP."
    ),
)
@click.option(
    "--auto_start_policy",
    is_flag=True,
    default=False,
    help="Start policy automatically after warmup instead of waiting for key 'c'.",
)
def main(input, output, robot_config,
    match_dataset, match_episode, match_camera,
    match_replay_stride, match_replay_max_samples, match_replay_duration_scale,
    camera_reorder,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, device, valve_classifier_checkpoint, valve_classifier_device,
    allow_valve_classifier_override,
    command_latency, no_spacemouse, no_gripper,
    direct_dynamixel_gripper, dynamixel_gripper_config, gripper_calib_zarr,
    no_mirror, sim_fov, camera_intrinsics, eval_image_mask, policy_image_crop_ratio,
    inpaint_aruco_tags, aruco_config, disable_eval_image_aug,
    mirror_swap, print_policy_output,
    pose_eval_audit, dataset_zarr, dataset_z_stride, print_model_input,
    show_policy_image, policy_input_audit, save_fusion_attention, save_context_inputs,
    save_policy_inputs,
    coord_transform_audit,
    zero_ft_on_start, ft_zero_samples, ft_max_age_sec,
    print_motion_debug, vis_pose, max_policy_iters, plan_only,
    tcp_delta_scales, action_scale, motion_momentum_previous_weight,
    freeze_rotation, match_g_move_robot,
    auto_start_policy):
    max_gripper_width = 0.1
    gripper_speed = 0.2
    no_gripper_obs_width = _SYNTHETIC_GRIPPER_WIDTH
    
    # load robot config file (single arm: one robot + one gripper)
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data.get('grippers', [])
    auto_no_gripper = False
    if len(robots_config) != 1:
        raise ValueError('eval_real_indy expects exactly one robot in robot_config YAML.')
    if not no_gripper:
        if len(grippers_config) == 0:
            no_gripper = True
            auto_no_gripper = True
            print(
                "No gripper config found; running without gripper hardware and "
                "feeding synthetic robot0_gripper_width."
            )
        elif len(grippers_config) != 1:
            raise ValueError(
                'When --no_gripper is not set, eval_real_indy expects exactly one gripper in robot_config YAML.'
            )
    rc = robots_config[0]
    gc = grippers_config[0] if len(grippers_config) > 0 else {}
    if direct_dynamixel_gripper:
        raise click.ClickException(
            "eval_real_indy_rg2.py does not support direct Dynamixel control. "
            "Use eval_real_indy_dynamixel.py for that hardware."
        )
    if not no_gripper:
        gripper_type = str(gc.get("gripper_type", "rg2ft")).lower()
        if gripper_type != "rg2ft":
            raise click.ClickException(
                "eval_real_indy_rg2.py requires gripper_type: rg2ft; "
                f"got {gripper_type!r}."
            )
        if not gc.get("gripper_ip"):
            raise click.ClickException(
                "RG2-FT config requires gripper_ip (OnRobot Compute Box IP)."
            )
        max_gripper_width = float(gc.get("max_gripper_width", 0.1))
        if not (0.0 < max_gripper_width <= 0.1):
            raise click.ClickException(
                "RG2-FT max_gripper_width must be in (0, 0.1] metres."
            )
        print(
            "RG2-FT config:",
            f"{gc['gripper_ip']}:{int(gc.get('gripper_port', 502))}",
            f"slave={int(gc.get('gripper_slave_id', 65))}",
            f"width=0..{max_gripper_width:.3f}m",
            f"force={float(gc.get('rg2ft_force', 20.0)):.1f}N",
            f"home_to_open={bool(gc.get('rg2ft_home_to_open', False))}",
        )
    if gc.get("gripper_type") == "dynamixel":
        max_gripper_width = float(
            gc.get("dynamixel_max_gripper_width", gc.get("max_gripper_width", 0.09))
        )
    n_robots = 1
    direct_gripper_cm = nullcontext(None)
    if no_gripper:
        print(
            "no_gripper: feeding synthetic robot0_gripper_width "
            f"{float(no_gripper_obs_width):.9f}."
        )
        if auto_no_gripper and direct_dynamixel_gripper and not plan_only:
            try:
                width_min_m, width_max_m, calib_zarr_path = _load_gripper_width_range_from_zarr(
                    gripper_calib_zarr
                )
                yaml_gripper_cfg, yaml_gripper_path = _load_rulebase_gripper_config(
                    dynamixel_gripper_config
                )
            except Exception as exc:
                raise click.ClickException(
                    "Failed to prepare direct Dynamixel gripper calibration. "
                    "Use --no_direct_dynamixel_gripper to disable it, or pass "
                    "a zarr with finite robot0_gripper_width via --gripper_calib_zarr. "
                    "Note: -m/--match_dataset is only for first-scene overlay, "
                    "not gripper calibration. "
                    f"Also check --dynamixel_gripper_config. Error: {exc}"
                ) from exc
            direct_gripper_cm = _DirectDynamixelGripper(
                yaml_config=yaml_gripper_cfg,
                yaml_path=yaml_gripper_path,
                width_min_m=width_min_m,
                width_max_m=width_max_m,
                zarr_path=calib_zarr_path,
                print_debug=print_motion_debug,
            )
            no_gripper_obs_width = _sanitize_gripper_width(
                width_max_m,
                _SYNTHETIC_GRIPPER_WIDTH,
                tag="zarr gripper max/open",
            )
            max_gripper_width = float(width_max_m)
            print(
                "direct_dynamixel_gripper: enabled. Synthetic obs gripper width "
                f"starts at zarr max/open {float(no_gripper_obs_width):.9f} m."
            )

    # Human teleop rotation deltas (keyboard / SpaceMouse) use this Euler order.
    teleop_euler_seq = rc.get("indy_teleop_rot_euler_seq", "xyz")
    teleop_euler_extrinsic = rc.get("indy_teleop_rot_euler_extrinsic", False)
    robot_from_dataset_T_cfg = rc.get("indy_robot_from_dataset_transform", None)
    _set_robot_dataset_transform(robot_from_dataset_T_cfg)
    if robot_from_dataset_T_cfg is not None:
        offset_rv = st.Rotation.from_matrix(
            _ROBOT_FROM_DATASET_T[:3, :3]
        ).as_rotvec()
        print(
            "robot_from_dataset/tag offset enabled: "
            f"t={np.array2string(_ROBOT_FROM_DATASET_T[:3, 3], precision=5)} "
            f"rotvec={np.array2string(offset_rv, precision=5)}"
        )
    # Policy tcp7 rotvec: optional round-trip through Indy's task Euler chart
    # (indy_task_rot_*) so commands align with movetelel_abs / ActualTCPPose.
    policy_rot_rt = rc.get("indy_policy_tcp7_rot_euler_roundtrip", False)
    policy_rot_seq = rc.get("indy_policy_tcp7_rot_euler_seq")
    policy_rot_ext = rc.get("indy_policy_tcp7_rot_euler_extrinsic")
    if policy_rot_seq is None:
        policy_rot_seq = rc.get("indy_task_rot_euler_seq", "xyz")
    if policy_rot_ext is None:
        policy_rot_ext = rc.get("indy_task_rot_euler_extrinsic", True)

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    checkpoint_contract = inspect_dual_ft_checkpoint_payload(payload)
    cfg = checkpoint_contract["cfg"]
    valve_context_enabled = bool(checkpoint_contract.get("valve_context_enabled", False))
    valve_context_schema = checkpoint_contract.get("valve_context_schema")
    expected_valve_classifier_sha256 = None
    if valve_context_enabled:
        expected_valve_classifier_sha256 = str(
            OmegaConf.select(
                cfg, "task.valve_context.checkpoint_sha256", default=""
            )
        ).lower()
        if len(expected_valve_classifier_sha256) != 64:
            raise click.ClickException(
                "context checkpoint is missing task.valve_context.checkpoint_sha256"
            )
        if no_gripper:
            raise click.ClickException(
                "the valve-context checkpoint requires live RG2-FT, so --no_gripper "
                "is not permitted"
            )
    print(
        "dual-F/T checkpoint contract: "
        f"condition=[1,{checkpoint_contract['condition_dim']}], "
        f"action=[1,{checkpoint_contract['action_horizon']},"
        f"{checkpoint_contract['action_dim']}], "
        f"FT=[1,{checkpoint_contract['ft_horizon']},"
        f"{checkpoint_contract['ft_dim']}], normalizer="
        f"{checkpoint_contract['normalizer_owner']}"
    )
    if valve_context_enabled:
        print(
            "valve context contract: "
            f"key={checkpoint_contract['valve_context_key']} "
            f"dim={checkpoint_contract['valve_context_dim']} "
            f"schema={valve_context_schema} "
            f"expected_sha256={expected_valve_classifier_sha256}"
        )
    # The checkpoint payload restores all trained weights immediately after
    # workspace construction. Avoid an unnecessary online pretrained-weight
    # download when this self-contained folder is moved to the robot PC.
    if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is not None:
        cfg.policy.obs_encoder.pretrained = False
        if bool(OmegaConf.select(cfg, "policy.obs_encoder.frozen", default=False)):
            OmegaConf.update(
                cfg,
                "policy.obs_encoder.allow_frozen_without_pretrained",
                True,
                force_add=True,
            )
    print("model_name:", cfg.policy.obs_encoder.model_name)
    left_ft_meta = OmegaConf.select(
        cfg, "task.shape_meta.obs.robot0_ft_left", default=None
    )
    right_ft_meta = OmegaConf.select(
        cfg, "task.shape_meta.obs.robot0_ft_right", default=None
    )
    if (left_ft_meta is None) != (right_ft_meta is None):
        raise click.ClickException(
            "checkpoint must request both robot0_ft_left and robot0_ft_right"
        )
    dual_ft_enabled = left_ft_meta is not None
    if dual_ft_enabled:
        left_ft_horizon = int(left_ft_meta.horizon)
        right_ft_horizon = int(right_ft_meta.horizon)
        if left_ft_horizon != right_ft_horizon:
            raise click.ClickException("left/right F/T horizons must match")
        ft_obs_horizon = left_ft_horizon
        ft_obs_stride = int(left_ft_meta.get("down_sample_steps", 1))
        if checkpoint_contract["condition_dim"] != 786:
            raise click.ClickException(
                "dual-F/T checkpoint condition must be [1,786], got "
                f"[1,{checkpoint_contract['condition_dim']}]"
            )
    else:
        ft_obs_horizon = 0
        ft_obs_stride = 1

    ft_bias_removal = str(
        OmegaConf.select(cfg, "task.ft.bias_removal", default="none")
    )
    if dual_ft_enabled and ft_bias_removal not in {
        "precomputed_in_sidecar",
        "episode_start_mean",
    }:
        raise click.ClickException(
            "dual-F/T checkpoint must have been trained with bias removal; got "
            f"task.ft.bias_removal={ft_bias_removal!r}"
        )
    if dual_ft_enabled and not np.allclose(
        _ROBOT_FROM_DATASET_T, np.eye(4), atol=1e-9
    ):
        raise click.ClickException(
            "dual-F/T deployment forbids an extra dataset->robot coordinate "
            "transform. Set indy_robot_from_dataset_transform to identity; the "
            "Indy mm/Euler/flange-to-TCP protocol conversion remains enabled."
        )
    if dual_ft_enabled and policy_rot_rt:
        raise click.ClickException(
            "dual-F/T deployment requires indy_policy_tcp7_rot_euler_roundtrip=false"
        )
    if dual_ft_enabled and match_g_move_robot:
        raise click.ClickException(
            "--match_g_move_robot is disabled for dual-F/T deployment. Use the "
            "training first frame only as a visual alignment reference and move "
            "to the saved/live initial pose through teleop."
        )
    if dual_ft_enabled:
        image_contract_errors = []
        if sim_fov is not None:
            image_contract_errors.append("--sim_fov must be omitted (no distortion correction)")
        if not eval_image_mask:
            image_contract_errors.append("--eval_image_mask must be enabled")
        if no_mirror:
            image_contract_errors.append("--no_mirror must be disabled")
        if mirror_swap:
            image_contract_errors.append("--mirror_swap must be disabled")
        if abs(float(policy_image_crop_ratio) - 1.0) > 1e-9:
            image_contract_errors.append("--policy_image_crop_ratio must be 1.0")
        if not inpaint_aruco_tags:
            image_contract_errors.append("--inpaint_aruco_tags must be enabled")
        if not disable_eval_image_aug:
            image_contract_errors.append("--disable_eval_image_aug must be enabled")
        if image_contract_errors:
            raise click.ClickException(
                "live image preprocessing differs from session_260827 training: "
                + "; ".join(image_contract_errors)
            )
        print(
            "training image contract: ArUco inpaint -> gripper mask "
            "(finger mask off) -> resize/crop ratio 1.0; distortion correction off"
        )

    startup_bias_cfg = FTStartupBiasConfig.from_mapping(
        robot_config_data.get("ft_startup_bias", {})
    )
    motion_safety_cfg = PolicyMotionSafetyConfig.from_mapping(
        robot_config_data.get("policy_safety", {})
    )
    ft_safety_cfg = FTSafetyConfig.from_mapping(
        robot_config_data.get("ft_safety", {})
    )
    force_feedback = None
    if int(checkpoint_contract["action_dim"]) == 11:
        feedback_mapping = OmegaConf.select(
            cfg, "task.grasp_force_feedback", default=None
        )
        if feedback_mapping is None:
            raise click.ClickException(
                "11-D checkpoint has no task.grasp_force_feedback configuration"
            )
        feedback_cfg = GraspForceWidthFeedbackConfig.from_mapping(
            OmegaConf.to_container(feedback_mapping, resolve=True)
        )
        force_feedback = GraspForceWidthFeedbackController(feedback_cfg)
        print(
            "grasp-force control: action[10] is a bounded width-feedback "
            "reference; RG2 force register remains fixed by robot YAML"
        )

    if steps_per_inference is None:
        steps_per_inference = int(
            OmegaConf.select(cfg, "execution.n_action_steps", default=6)
        )
    if dual_ft_enabled:
        allowed_steps = list(
            OmegaConf.select(
                cfg,
                "execution.allowed_n_action_steps",
                default=[1, 2, 4, 6, 8],
            )
        )
        # Existing dual-F/T checkpoints predate the validated 6-step runtime
        # option and therefore serialize [1, 2, 4, 8]. The prediction horizon
        # is 16, so executing six predicted rows is supported without retraining.
        allowed_steps = sorted(set(map(int, allowed_steps)) | {6})
        if int(steps_per_inference) not in set(map(int, allowed_steps)):
            raise click.ClickException(
                f"dual-F/T steps_per_inference must be one of {allowed_steps}"
            )
    if frequency is None:
        frequency = float(
            OmegaConf.select(cfg, "execution.action_frequency", default=19.98)
        )
    if frequency <= 0:
        raise click.ClickException("frequency must be positive")
    ft_sample_frequency = float(
        OmegaConf.select(
            cfg,
            "task.ft_frequency",
            default=gc.get('rg2ft_frequency', 100.0),
        )
    )
    configured_ft_max_age = OmegaConf.select(
        cfg, "task.ft_max_age_sec", default=None
    )
    # The August checkpoint predates the explicit config field. Its 0814
    # alignment audit measured 11.182 ms max age at 100 Hz, which is 1.118
    # samples; 1.2 sample periods preserves that evidence-based limit.
    if not np.isfinite(ft_max_age_sec) or ft_max_age_sec <= 0:
        raise click.ClickException("--ft_max_age_sec must be a finite positive value")
    ft_max_age = float(ft_max_age_sec)
    if dual_ft_enabled:
        print(
            "dual-F/T freshness limit: "
            f"{ft_max_age * 1000.0:.3f} ms (runtime override; checkpoint "
            f"metadata={float(configured_ft_max_age) * 1000.0:.3f} ms)"
            if configured_ft_max_age is not None
            else f"{ft_max_age * 1000.0:.3f} ms (runtime override)"
        )
    if dual_ft_enabled and no_gripper:
        raise click.ClickException(
            "dual-F/T deployment requires the live RG2-FT sensor; "
            "--no_gripper is not supported"
        )
    embedded_dataset_path = str(cfg.task.dataset.dataset_path)
    print("checkpoint dataset_path metadata:", embedded_dataset_path)
    print(
        "runtime dataset_path: checkpoint metadata retained "
        "(training zarr is optional unless an audit explicitly requests it)"
    )

    dataset_z_stats = None
    if pose_eval_audit or dataset_zarr:
        try:
            from eval_pose_audit_util import (
                format_dataset_z_block,
                load_tcp_z_stats_from_replay,
                resolve_zarr_dataset_path,
            )

            ckpt_abs = str(pathlib.Path(ckpt_path).expanduser().resolve())
            zpath = None
            if dataset_zarr:
                ds_arg = str(dataset_zarr).strip()
                if ds_arg.lower() == "auto":
                    zpath = resolve_zarr_dataset_path(
                        str(cfg.task.dataset.dataset_path), ckpt_abs
                    )
                else:
                    cand = pathlib.Path(os.path.expanduser(dataset_zarr))
                    zpath = str(cand.resolve()) if cand.exists() else None
            else:
                zpath = resolve_zarr_dataset_path(
                    str(cfg.task.dataset.dataset_path), ckpt_abs
                )
            if zpath:
                dataset_z_stats = load_tcp_z_stats_from_replay(
                    zpath,
                    stride=max(1, int(dataset_z_stride)),
                    action_key="action",
                )
                print(
                    "[dataset tcp z / magnitude benchmark]\n"
                    + format_dataset_z_block(dataset_z_stats)
                )
            else:
                print(
                    "[dataset tcp z / magnitude benchmark] skipped "
                    f"(no file on disk for dataset_path={cfg.task.dataset.dataset_path!r}); "
                    "use --dataset_zarr /path/to/replay_buffer.zarr or place zarr next to ckpt."
                )
        except Exception as exc:
            print(f"[dataset tcp z / magnitude benchmark] failed: {exc}")

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        if camera_intrinsics is None:
            raise click.ClickException(
                "--camera_intrinsics is required when --sim_fov is set.")
        opencv_intr_dict = parse_fisheye_intrinsics_file(camera_intrinsics)
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )
        print(
            "fisheye rectification:",
            f"intrinsics={camera_intrinsics}",
            f"source_dim={opencv_intr_dict['DIM'].tolist()}",
            f"out_res={obs_res}",
            f"out_fov={sim_fov}",
        )
    else:
        print("fisheye rectification: off (matches 0709 no-out_fov training images)")
    if not (0.0 < float(policy_image_crop_ratio) <= 1.0):
        raise click.ClickException(
            "--policy_image_crop_ratio must be in (0, 1]. "
            "Use <1 to zoom in; if live FOV is already too narrow, switch camera/lens mode."
        )

    print("steps_per_inference:", steps_per_inference)
    print("action_frequency_hz:", f"{frequency:.9f}")
    print(
        "replanning_interval_ms:",
        f"{1000.0 * int(steps_per_inference) / frequency:.3f}",
    )
    tcp_delta_scale_vec = _parse_tcp_delta_scales(tcp_delta_scales)
    motion_momentum_previous_weight = float(motion_momentum_previous_weight)
    if not 0.0 <= motion_momentum_previous_weight < 1.0:
        raise click.ClickException(
            "--motion_momentum_previous_weight must be in [0, 1)."
        )
    if plan_only:
        print(
            "plan_only: this script will not submit waypoints, but the robot "
            "controller is still connected; running until stopped."
        )
    if max_policy_iters is not None:
        print("max_policy_iters:", max_policy_iters)
    if tcp_delta_scale_vec is not None:
        print("tcp_delta_scales:", tcp_delta_scale_vec.tolist())
    if action_scale != 1.0:
        print("action_scale:", action_scale)
    if motion_momentum_previous_weight > 0.0:
        print(
            "motion_momentum: previous="
            f"{motion_momentum_previous_weight:.6f} current="
            f"{1.0 - motion_momentum_previous_weight:.6f} "
            "(TCP + gripper; measured-state anchor each policy cycle)"
        )
    if freeze_rotation:
        print("freeze_rotation: on")
    policy_image_audit_enabled = bool(policy_input_audit or show_policy_image)
    coord_transform_audit_enabled = bool(coord_transform_audit or pose_eval_audit)
    with SharedMemoryManager() as shm_manager:
        sm_ctx = nullcontext(None)
        if not no_spacemouse:
            try:
                from umi.real_world.spacemouse_shared_memory import Spacemouse
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "SpaceMouse requested but `spnav` is not installed. "
                    "Install spnav in the container, or run with --no_spacemouse."
                ) from exc
            sm_ctx = Spacemouse(shm_manager=shm_manager)
        with sm_ctx as sm, direct_gripper_cm as direct_gripper, UmiEnv(
                output_dir=output,
                robot_ip=rc['robot_ip'],
                gripper_ip=gc.get('gripper_ip'),
                gripper_port=gc.get('gripper_port', 502),
                gripper_slave_id=gc.get('gripper_slave_id', 65),
                gripper_type=gc.get('gripper_type', 'rg2ft'),
                rg2ft_frequency=gc.get('rg2ft_frequency', 100),
                rg2ft_force=gc.get('rg2ft_force', 20.0),
                rg2ft_home_to_open=(
                    False if plan_only else gc.get('rg2ft_home_to_open', False)
                ),
                rg2ft_move_max_speed=gc.get('rg2ft_move_max_speed', 0.2),
                rg2ft_open_tolerance=gc.get('rg2ft_open_tolerance', 0.005),
                rg2ft_zero_on_start=zero_ft_on_start,
                rg2ft_zero_samples=ft_zero_samples,
                gripper_commands_enabled=(not plan_only),
                gripper_serial_port=gc.get('gripper_serial_port'),
                dynamixel_id=gc.get('dynamixel_id', 1),
                dynamixel_baudrate=gc.get('dynamixel_baudrate', 57600),
                dynamixel_protocol_version=gc.get('dynamixel_protocol_version', 2.0),
                dynamixel_open_position=gc.get('dynamixel_open_position', 1600),
                dynamixel_close_position=gc.get('dynamixel_close_position', 200),
                dynamixel_max_gripper_width=gc.get(
                    'dynamixel_max_gripper_width', gc.get('max_gripper_width', 0.09)
                ),
                dynamixel_profile_velocity=gc.get('dynamixel_profile_velocity', 30),
                dynamixel_profile_acceleration=gc.get('dynamixel_profile_acceleration', 15),
                dynamixel_current_limit=gc.get('dynamixel_current_limit'),
                dynamixel_pwm_limit=gc.get('dynamixel_pwm_limit'),
                dynamixel_move_max_speed=gc.get('dynamixel_move_max_speed', 0.05),
                dynamixel_home_to_open=(
                    False if plan_only else gc.get('dynamixel_home_to_open', False)
                ),
                use_gripper=(not no_gripper),
                robot_type=rc['robot_type'],
                tcp_offset=rc['tcp_offset'],
                frequency=frequency,
                obs_image_resolution=obs_res,
                # A context classifier consumes every camera frame rather
                # than only the diffusion policy's two image observations.
                # Keep two seconds at its intended 60 Hz rate so a slower
                # policy replan cannot discard the unseen frames.
                max_obs_buffer_size=(120 if valve_context_enabled else 60),
                context_camera_history_frames=(120 if valve_context_enabled else 60),
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=(False if plan_only else init_joints),
                # The raw "Multi Cam Vis" window bypasses the match overlay and
                # is easily mistaken for the evaluation view.  The main window
                # below is the single full-resolution camera view.
                enable_multi_cam_vis=False,
                camera_obs_latency=float(cfg.task.get("camera_obs_latency", 0.125)),
                robot_obs_latency=rc['robot_obs_latency'],
                gripper_obs_latency=gc.get('gripper_obs_latency', 0.01),
                robot_action_latency=rc.get('robot_action_latency', 0.1),
                gripper_action_latency=gc.get('gripper_action_latency', 0.1),
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=int(
                    OmegaConf.select(
                        cfg,
                        "task.shape_meta.obs.robot0_gripper_width.horizon",
                        default=OmegaConf.select(
                            cfg, "task.low_dim_obs_horizon", default=2
                        ),
                    )
                ),
                ft_obs_horizon=ft_obs_horizon,
                ft_obs_stride=ft_obs_stride,
                ft_obs_frequency=ft_sample_frequency,
                ft_max_age=ft_max_age if dual_ft_enabled else None,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                policy_image_crop_ratio=policy_image_crop_ratio,
                mask_before_image_transform=eval_image_mask,
                inpaint_aruco_tags=inpaint_aruco_tags,
                aruco_config_path=aruco_config,
                mirror_swap=mirror_swap,
                max_pos_speed=float(rc.get("indy_max_pos_speed_m_s", 0.3)),
                max_rot_speed=float(rc.get("indy_max_rot_speed_rad_s", 1.0)),
                indy_command_timeout_s=float(
                    rc.get("indy_command_timeout_s", 0.3)
                ),
                indy_task_rot_is_euler=rc.get("indy_task_rot_is_euler", True),
                indy_task_rot_euler_seq=rc.get("indy_task_rot_euler_seq", "xyz"),
                indy_task_rot_euler_in_degrees=rc.get(
                    "indy_task_rot_euler_in_degrees", True
                ),
                indy_task_rot_euler_extrinsic=rc.get(
                    "indy_task_rot_euler_extrinsic", True
                ),
                indy_task_frame_xyz_signs=tuple(
                    rc.get("indy_task_frame_xyz_signs", [1, 1, 1])
                ),
                indy_tool_rot_offset_deg=tuple(
                    rc.get("indy_tool_rot_offset_deg", [0, 0, 0])
                ),
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            cv2.namedWindow("default", cv2.WINDOW_AUTOSIZE)
            if show_policy_image:
                cv2.namedWindow("policy_input", cv2.WINDOW_AUTOSIZE)
            has_gripper_control = (
                ((not no_gripper) and (not plan_only))
                or (direct_gripper is not None)
            )
            if no_gripper and direct_gripper is not None:
                no_gripper_obs_width = _sanitize_gripper_width(
                    direct_gripper.initial_width_m,
                    max_gripper_width,
                    tag="direct gripper initial width",
                )
                print(
                    "no_gripper synthetic obs initialized from real Dynamixel: "
                    f"{float(no_gripper_obs_width):.9f} m"
                )
            print("Waiting for camera")
            time.sleep(1.0)

            startup_bias_result = None
            if dual_ft_enabled:
                click.confirm(
                    "Remove all contact/load from both RG2 fingers, keep the "
                    "gripper and robot still, then continue with startup F/T bias calibration",
                    abort=True,
                )
                print(
                    "Calibrating native left/right F/T startup bias from "
                    f"{startup_bias_cfg.sample_count} unique samples..."
                )
                try:
                    startup_bias_result = acquire_startup_bias(
                        env.gripper, startup_bias_cfg
                    )
                except (KeyError, ValueError, TimeoutError) as exc:
                    raise click.ClickException(
                        f"live F/T startup bias calibration failed: {exc}"
                    ) from exc
                startup_bias_12d = np.asarray(
                    startup_bias_result["bias_12d"], dtype=np.float64
                )
                software_tare_offset_12d = np.asarray(
                    env.rg2ft_ft_offset, dtype=np.float64
                ).copy()
                startup_residual_bias_12d = startup_residual_after_software_tare(
                    startup_bias_12d, software_tare_offset_12d
                )
                # Training applies software tare first and removes only the
                # remaining stationary episode bias.  The sampler above reads
                # native raw wrenches, so convert it to that tared frame before
                # giving it to UmiEnv.
                env.set_ft_startup_bias(startup_residual_bias_12d)
                calibration_record = {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "coordinate_frame": "native left[6] + native right[6]",
                    "calibration_order": (
                        "raw native -> subtract software_tare_offset_12d -> "
                        "subtract startup_residual_after_software_tare_12d"
                    ),
                    "used_for": "this process only; never an offline episode bias",
                    "software_tare_offset_12d": software_tare_offset_12d.tolist(),
                    "raw_startup_bias_12d": startup_bias_12d.tolist(),
                    "startup_residual_after_software_tare_12d": (
                        startup_residual_bias_12d.tolist()
                    ),
                    "config": vars(startup_bias_cfg),
                    **{
                        key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in startup_bias_result.items()
                    },
                }
                calibration_path = pathlib.Path(output).joinpath(
                    "ft_startup_bias_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
                )
                calibration_path.write_text(
                    json.dumps(calibration_record, indent=2), encoding="utf-8"
                )
                print(
                    "F/T raw startup baseline accepted:",
                    np.array2string(startup_bias_12d, precision=5),
                )
                print(
                    "F/T residual after software tare applied to policy/feedback:",
                    np.array2string(startup_residual_bias_12d, precision=5),
                )
                print("F/T calibration provenance:", calibration_path)

            # load match_dataset
            episode_first_frame_map = dict()
            episode_first_policy_frame_map = dict()
            match_replay_buffer = None
            match_zarr_path = None
            match_policy_image_info = None
            if match_dataset is not None:
                match_zarr_path, match_dir = _resolve_match_dataset_paths(match_dataset)
                print(f"[match_dataset] zarr: {match_zarr_path}")
                match_replay_buffer = _load_match_replay_buffer(match_zarr_path)
                try:
                    episode_first_policy_frame_map, match_policy_image_info = (
                        _load_zarr_episode_first_policy_frames(match_zarr_path)
                    )
                    episode_first_frame_map.update(episode_first_policy_frame_map)
                except Exception as exc:
                    print(
                        "[match_dataset] zarr policy-image load failed "
                        f"(policy_input overlap unavailable): {exc}"
                    )
                if len(episode_first_frame_map) == 0:
                    match_video_dir = match_dir.joinpath('videos')
                    for vid_dir in match_video_dir.glob("*/"):
                        episode_idx = int(vid_dir.stem)
                        match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                        if match_video_path.exists():
                            img = None
                            with av.open(str(match_video_path)) as container:
                                stream = container.streams.video[0]
                                for frame in container.decode(stream):
                                    img = frame.to_ndarray(format='rgb24')
                                    break

                            episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = _get_eval_workspace_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            if dual_ft_enabled:
                encoder_shape = tuple(policy.obs_encoder.output_shape())
                if encoder_shape != (1, 786):
                    raise RuntimeError(
                        "restored dual-F/T observation encoder must emit "
                        f"condition [1,786], got {encoder_shape}"
                    )
                print("restored observation-encoder condition shape: [1,786]")
            if disable_eval_image_aug:
                disabled_keys = _disable_policy_image_transforms(policy)
                if disabled_keys:
                    print(
                        "eval image augmentation disabled for keys:",
                        disabled_keys,
                        "(RandomCrop/ColorJitter -> Identity)",
                    )
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)
            if pose_eval_audit:
                _print_ckpt_pose_eval_contract(cfg)
            policy_image_audit_printed = False
            coord_transform_audit_printed = False

            requested_device = str(device).strip().lower()
            device = torch.device('cpu')
            if requested_device == 'auto':
                if torch.cuda.is_available():
                    try:
                        device = torch.device('cuda')
                        policy.eval().to(device)
                    except Exception as exc:
                        print(f"CUDA init failed ({exc}). Falling back to CPU inference.")
                        device = torch.device('cpu')
                        policy.eval().to(device)
                else:
                    print("CUDA not available. Falling back to CPU inference.")
                    policy.eval().to(device)
            elif requested_device in ('cpu', ''):
                print("Using CPU inference by request.")
                policy.eval().to(device)
            elif requested_device.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise click.ClickException(
                        f"--device {requested_device!s} requested but CUDA is unavailable"
                    )
                try:
                    device = torch.device(requested_device)
                    policy.eval().to(device)
                except Exception as exc:
                    raise click.ClickException(
                        f"failed to initialize --device {requested_device}: {exc}"
                    ) from exc
            else:
                raise click.ClickException(
                    "--device must be auto, cpu, cuda, or cuda:N; got "
                    f"{requested_device!r}"
                )

            valve_context_runtime = None
            valve_classifier_override_used = False
            if valve_context_enabled:
                requested_classifier_device = str(
                    valve_classifier_device
                ).strip().lower()
                if requested_classifier_device == "auto":
                    classifier_device = str(device)
                elif requested_classifier_device in ("", "cpu"):
                    classifier_device = "cpu"
                elif requested_classifier_device.startswith("cuda"):
                    if not torch.cuda.is_available():
                        raise click.ClickException(
                            "--valve_classifier_device requests CUDA, but CUDA is unavailable"
                        )
                    classifier_device = requested_classifier_device
                else:
                    raise click.ClickException(
                        "--valve_classifier_device must be auto, cpu, cuda, or cuda:N; "
                        f"got {valve_classifier_device!r}"
                    )
                configured_classifier_path = valve_classifier_checkpoint
                if configured_classifier_path in (None, ""):
                    configured_classifier_path = OmegaConf.select(
                        cfg, "task.valve_context.checkpoint_path", default=None
                    )
                if configured_classifier_path in (None, ""):
                    configured_classifier_path = _DEFAULT_VALVE_CLASSIFIER_CHECKPOINT
                classifier_path = pathlib.Path(str(configured_classifier_path)).expanduser()
                if not classifier_path.is_absolute():
                    classifier_path = _PROJECT_ROOT / classifier_path
                classifier_path = classifier_path.resolve()
                if not classifier_path.is_file():
                    raise click.ClickException(
                        "frozen valve classifier checkpoint is missing: "
                        f"{classifier_path}"
                    )
                actual_classifier_sha256 = _sha256_file(classifier_path)
                if allow_valve_classifier_override:
                    if valve_classifier_checkpoint in (None, ""):
                        raise click.ClickException(
                            "--allow_valve_classifier_override requires an explicit "
                            "--valve_classifier_checkpoint"
                        )
                    if valve_context_schema != VALVE_CONTEXT_V2_SCHEMA:
                        raise click.ClickException(
                            "RGB/F-T classifier override is supported only for the "
                            "four-state v2 policy context"
                        )
                    runtime_type = RGBForceValveContextRuntime
                    valve_classifier_override_used = True
                    print(
                        "WARNING: explicit RGB/F-T valve classifier override enabled. "
                        "The action policy was trained with observer sha256="
                        f"{expected_valve_classifier_sha256}, but live evaluation will "
                        f"use sha256={actual_classifier_sha256}. The 5-D output contract "
                        "is identical; probability calibration/distribution may differ."
                    )
                else:
                    if (
                        actual_classifier_sha256.lower()
                        != expected_valve_classifier_sha256
                    ):
                        raise click.ClickException(
                            "valve classifier SHA-256 mismatch; context checkpoint "
                            f"requires {expected_valve_classifier_sha256}, got "
                            f"{actual_classifier_sha256}. Use the explicit compatible "
                            "override only after verifying four-state output semantics."
                        )
                    runtime_type = (
                        ValveStateContextRuntimeV2
                        if valve_context_schema == VALVE_CONTEXT_V2_SCHEMA
                        else ValveStateContextRuntime
                    )
                try:
                    valve_context_runtime = runtime_type(
                        classifier_path, device=classifier_device
                    )
                except Exception as exc:
                    raise click.ClickException(
                        f"failed to load frozen valve context observer: {exc}"
                    ) from exc
                if valve_context_runtime.context_schema != valve_context_schema:
                    raise click.ClickException(
                        "valve observer schema does not match the action checkpoint: "
                        f"observer={valve_context_runtime.context_schema} "
                        f"policy={valve_context_schema}"
                    )
                print(
                    "valve context observer: "
                    f"{classifier_path} sha256={actual_classifier_sha256} "
                    f"device={classifier_device} "
                    f"version={valve_context_runtime.observer_version}"
                )
                if save_context_inputs:
                    print(
                        "[eval_log] context input capture enabled: every exact "
                        "frozen-classifier temporal input and output will be saved per episode."
                    )
            elif save_context_inputs:
                print(
                    "[eval_log] context input capture skipped: this checkpoint has no "
                    "valve_context input."
                )

            fusion_attention_enabled = False
            fusion_attention_tokens = []
            if save_fusion_attention:
                capture_attention = getattr(
                    policy.obs_encoder, "set_fusion_attention_capture", None
                )
                token_names = getattr(policy.obs_encoder, "fusion_token_names", None)
                if not dual_ft_enabled or not callable(capture_attention) or not callable(token_names):
                    print(
                        "[eval_log] fusion-attention capture unavailable for this "
                        "checkpoint; F/T/output diagnostics remain enabled."
                    )
                else:
                    capture_attention(True)
                    fusion_attention_tokens = list(token_names())
                    fusion_attention_enabled = True
                    print(
                        "[eval_log] fusion-attention capture enabled: "
                        + ", ".join(fusion_attention_tokens)
                    )

            print("Warming up policy inference")
            obs = _get_warmup_observation_with_retry(env)
            if no_gripper:
                obs = _with_synthetic_gripper_width(
                    obs,
                    no_gripper_obs_width,
                    fallback=max_gripper_width,
                )
            episode_start_pose = [
                np.concatenate([
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_rot_axis_angle'],
                ], axis=-1)[-1]
            ]
            with torch.inference_mode():
                policy.reset()
                valve_context_record = None
                obs_with_context = obs
                if valve_context_runtime is not None:
                    obs_with_context, valve_context_record = (
                        _add_valve_context_to_policy_observation(
                            obs, runtime=valve_context_runtime
                        )
                    )
                    print(
                        "[valve_context warmup] "
                        f"phase={valve_context_record.phase_name} "
                        f"reason={valve_context_record.error_reason_name} "
                        f"warmed_up={int(valve_context_record.warmed_up)}"
                    )
                obs_for_model = prepare_rg2ft_policy_obs(
                    obs_with_context, cfg.task.shape_meta
                )
                episode_start_pose_for_model = episode_start_pose
                if not dual_ft_enabled:
                    obs_for_model = _apply_slam_frame_fix_to_obs(
                        obs_for_model, n_robots
                    )
                    episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(
                        episode_start_pose
                    )
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs_for_model, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose_for_model)
                obs_dict_np = _policy_obs_float32(obs_dict_np)
                _check_policy_inputs_finite(obs_dict_np, "[warmup]")
                audit_match_episode = match_episode
                if audit_match_episode is None and len(episode_first_policy_frame_map) > 0:
                    audit_match_episode = min(episode_first_policy_frame_map)
                audit_train_rgb = (
                    episode_first_policy_frame_map.get(int(audit_match_episode))
                    if audit_match_episode is not None
                    else None
                )
                if policy_image_audit_enabled and not policy_image_audit_printed:
                    _print_policy_image_audit(
                        obs_for_model,
                        obs_dict_np,
                        cfg.task.shape_meta,
                        "[warmup]",
                        train_rgb=audit_train_rgb,
                        train_info=match_policy_image_info,
                    )
                    policy_image_audit_printed = True
                if print_model_input:
                    _print_model_input_debug(
                        obs_dict_np,
                        obs,
                        episode_start_pose,
                        obs_pose_rep,
                        "[warmup]",
                    )
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                raw_pred = result["action_pred"][0].detach().to("cpu").numpy()
                expected_action_shape = (
                    int(checkpoint_contract["action_horizon"]),
                    int(checkpoint_contract["action_dim"]) * n_robots,
                )
                if raw_pred.shape != expected_action_shape:
                    raise RuntimeError(
                        "policy must return its full trained trajectory; got "
                        f"{raw_pred.shape}, expected {expected_action_shape}"
                    )
                _check_finite_array("[warmup] raw action_pred", raw_pred)
                if force_feedback is not None:
                    _check_finite_array(
                        "[warmup] predicted grasp force reference",
                        raw_pred[:, 10],
                    )
                if not dual_ft_enabled:
                    raw_pred = _apply_slam_frame_fix(raw_pred, n_robots)
                action_dataset = _decode_real_umi_action_checked(
                    raw_pred, obs_for_model, action_pose_repr, "[warmup dataset]"
                )
                action = action_dataset
                if not dual_ft_enabled:
                    action = _transform_tcp7_action(
                        action_dataset, _ROBOT_FROM_DATASET_T, n_robots
                    )
                _check_finite_array("[warmup] tcp7 action", action)
                action = _apply_policy_tcp7_rot_roundtrip(
                    action,
                    enabled=policy_rot_rt,
                    euler_seq=policy_rot_seq,
                    euler_extrinsic=policy_rot_ext,
                    n_robots=n_robots,
                )
                assert action.shape[-1] == 7 * n_robots
                if coord_transform_audit_enabled and not coord_transform_audit_printed:
                    _print_coord_transform_audit(
                        "[warmup]",
                        obs,
                        obs_for_model,
                        action_dataset=action_dataset,
                        action_robot=action,
                        match_debug_data=_make_match_pose_debug_data(
                            match_replay_buffer,
                            audit_match_episode,
                        ),
                        match_source_idx=0,
                    )
                    coord_transform_audit_printed = True
                if pose_eval_audit:
                    _print_pose_z_audit(
                        obs,
                        action,
                        action_pose_repr,
                        -1,
                        n_robots,
                        "[pose_eval_audit warmup]",
                        dataset_z_stats=dataset_z_stats,
                        raw_action_pred=raw_pred,
                    )
                if print_policy_output:
                    _print_policy_action_debug(
                        "[policy warmup]", raw_pred, action, submitted=None
                    )
                del result

            print("Ready!")
            print(
                "Indy rotation: teleop deltas use Euler seq "
                f"{teleop_euler_seq!r} (extrinsic={teleop_euler_extrinsic}); "
                "policy tcp7 rotvec round-trip "
                f"{'on' if policy_rot_rt else 'off'} via seq {policy_rot_seq!r} "
                f"(extrinsic={policy_rot_ext}). "
                "Match human vs Indy charts by setting indy_policy_tcp7_rot_euler_* "
                "and indy_teleop_rot_euler_* in robot_config."
            )
            print("Keyboard controls (human mode):")
            print('- Esc: quit, c: start policy, n/b: next/prev match episode, g: print match pose (teleop to align; add --match_g_move_robot to move)')
            print('- v: replay/follow selected match episode trajectory slowly for data-quality check')
            print('- t: save current TCP as start pose | p: move robot to saved start pose (4s)')
            print('- a/d: x+,x- | s/w: y+,y- | e/q: z+,z-')
            print('- j/l: roll-/+ | i/k: pitch+/+ | u/o: yaw-/+')
            if has_gripper_control:
                print('- z/x: gripper close/open')
            print(
                "Safety: Ctrl+C ends the script and stops the env controller "
                "(use the robot E-stop if motion does not stop immediately)."
            )
            saved_start_tcp6 = None
            selected_match_episode_for_eval = None
            auto_start_policy_pending = bool(auto_start_policy)
            if auto_start_policy_pending:
                print("auto_start_policy: will start policy after the first live obs/frame.")
            if _SAVED_START_POSE_PATH.exists():
                try:
                    with open(_SAVED_START_POSE_PATH) as f:
                        saved_start_tcp6 = np.asarray(
                            yaml.safe_load(f)["tcp6"], dtype=np.float64
                        )
                    print(
                        "saved start pose loaded "
                        f"({_SAVED_START_POSE_PATH.name}; press p to go there): "
                        f"{np.round(saved_start_tcp6, 4).tolist()}"
                    )
                except Exception as exc:
                    print(f"failed to load saved start pose: {exc}")
            terminal_key_poller = _TerminalKeyPoller()
            if terminal_key_poller.start():
                atexit.register(terminal_key_poller.close)
                print(
                    "Terminal keyboard fallback enabled: jog keys work from "
                    "this Docker terminal too."
                )
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                # Baseline from get_obs (same ActualTCPPose pipeline as policy), not a
                # single ring-buffer sample, so the first waypoint matches what we see.
                obs_human = env.get_obs(include_valve_context_stream=False)
                target_pose = np.asarray(
                    [
                        np.concatenate(
                            [
                                obs_human["robot0_eef_pos"][-1],
                                obs_human["robot0_eef_rot_axis_angle"][-1],
                            ]
                        )
                    ]
                )

                if not no_gripper:
                    gripper_target_pos = np.asarray(
                        [float(obs_human["robot0_gripper_width"][-1, 0])]
                    )
                else:
                    gripper_target_pos = np.asarray(
                        [float(no_gripper_obs_width)], dtype=np.float32
                    )

                episode_origin_tcp6 = target_pose[0].copy()
                t_start = time.monotonic()
                iter_idx = 0
                teleop_motion_latch_armed = True
                keyboard_motion_keys = frozenset(
                    {
                        ord("a"),
                        ord("d"),
                        ord("s"),
                        ord("w"),
                        ord("e"),
                        ord("q"),
                        ord("j"),
                        ord("l"),
                        ord("i"),
                        ord("k"),
                        ord("u"),
                        ord("o"),
                    }
                )
                if has_gripper_control:
                    keyboard_motion_keys = keyboard_motion_keys | frozenset(
                        (ord("z"), ord("x"))
                    )
                try:
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_sample = t_cycle_end - command_latency
                        t_command_target = t_cycle_end + dt
    
                        # pump obs
                        obs = env.get_obs(include_valve_context_stream=False)
                        if iter_idx == 0:
                            target_pose[0] = np.concatenate(
                                [
                                    obs["robot0_eef_pos"][-1],
                                    obs["robot0_eef_rot_axis_angle"][-1],
                                ]
                            )
                            if not no_gripper:
                                gripper_target_pos[0] = float(
                                    obs["robot0_gripper_width"][-1, 0]
                                )
    
                        # visualize (full-res camera feed; obs rgb is masked 224x224 for policy)
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = _get_live_display_bgr(env, camera_idx=match_camera)
                        if match_replay_buffer is not None:
                            match_min_episode = 0
                            match_max_episode = match_replay_buffer.n_episodes - 1
                        elif len(episode_first_frame_map) > 0:
                            match_min_episode = min(episode_first_frame_map)
                            match_max_episode = max(episode_first_frame_map)
                        else:
                            match_min_episode = episode_id
                            match_max_episode = episode_id

                        match_episode_id = episode_id
                        if match_episode is not None:
                            match_episode_id = match_episode
                        match_episode_id = int(
                            np.clip(match_episode_id, match_min_episode, match_max_episode)
                        )
                        match_episode = match_episode_id

                        match_has_first_frame = match_episode_id in episode_first_frame_map
                        if match_episode_id in episode_first_frame_map:
                            vis_img = _blend_match_rgb_on_live_bgr(
                                vis_img,
                                episode_first_frame_map[match_episode_id],
                            )

                        header = (
                            f"Eval ep: {episode_id} | Match ep: "
                            f"{match_episode_id}/{match_max_episode}"
                        )
                        if not match_has_first_frame:
                            header += " (no first-frame image)"
                        if vis_pose:
                            vis_img = _overlay_pose_vis(
                                vis_img,
                                header=header,
                                cur_tcp6=_tcp6_from_obs(obs),
                                target_tcp6=target_pose[0],
                                episode_origin_tcp6=episode_origin_tcp6,
                            )
                        else:
                            vis_img = _overlay_episode_text(vis_img, header)
                        cv2.imshow("default", vis_img)
                        if show_policy_image:
                            match_policy_rgb = episode_first_policy_frame_map.get(
                                match_episode_id
                            )
                            _show_policy_input_window(
                                obs,
                                f"policy input | match ep {match_episode_id}",
                                match_rgb=match_policy_rgb,
                            )
                        key = _poll_control_key(terminal_key_poller)
                        start_policy = False
                        if auto_start_policy_pending:
                            start_policy = True
                            auto_start_policy_pending = False
                        if key == 27:  # Esc
                            env.end_episode()
                            exit(0)
                        elif key == ord("c"):
                            start_policy = True
                        elif key == ord("n"):
                            match_episode = min(match_episode_id + 1, match_max_episode)
                            print(f"[match] selected episode {match_episode}")
                        elif key == ord("b"):
                            match_episode = max(match_episode_id - 1, match_min_episode)
                            print(f"[match] selected episode {match_episode}")
                        elif key == ord("g") and match_replay_buffer is not None:
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            pos = ep["robot0_eef_pos"][0]
                            rot = ep["robot0_eef_rot_axis_angle"][0]
                            pose = np.concatenate([pos, rot])
                            robot_pose = _match_episode_to_robot_tcp7(
                                ep,
                                fallback_gripper_width=no_gripper_obs_width,
                                stride=1,
                                max_samples=1,
                            )[0, :6]
                            live_tcp6 = _tcp6_from_obs(obs)
                            _print_match_pose_compare(
                                match_episode_id,
                                pose,
                                live_tcp6,
                                will_move=(match_g_move_robot and not plan_only),
                            )
                            if match_g_move_robot and plan_only:
                                print("[plan_only] skipped match-start robot/gripper move.")
                            elif match_g_move_robot:
                                duration = 3.0
                                grip = float(
                                    _match_episode_to_robot_tcp7(
                                        ep,
                                        fallback_gripper_width=no_gripper_obs_width,
                                        stride=1,
                                        max_samples=1,
                                    )[0, 6]
                                )
                                t_goal = time.time() + duration
                                if hasattr(env.robot, "servoL"):
                                    env.robot.servoL(robot_pose, duration=duration)
                                else:
                                    env.robot.schedule_waypoint(
                                        robot_pose, target_time=t_goal
                                    )
                                if not no_gripper and not plan_only:
                                    env.gripper.schedule_waypoint(
                                        grip, target_time=t_goal
                                    )
                                elif direct_gripper is not None:
                                    clipped, _ = direct_gripper.command_width(
                                        grip, force=True
                                    )
                                    no_gripper_obs_width = _sanitize_gripper_width(
                                        clipped,
                                        max_gripper_width,
                                            tag="match direct gripper feedback",
                                        )
                                target_pose[0] = robot_pose
                                gripper_target_pos[0] = grip
                                time.sleep(duration)
                        elif (
                            key == ord("v")
                            and match_replay_buffer is not None
                            and dual_ft_enabled
                        ):
                            print(
                                "[dual-F/T] absolute replay of a SLAM training "
                                "trajectory is disabled; use first-frame overlap "
                                "and teleop initialization only."
                            )
                        elif key == ord("v") and match_replay_buffer is not None:
                            if plan_only:
                                print("[plan_only] skipped match trajectory replay.")
                                continue
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            replay_actions = _match_episode_to_robot_tcp7(
                                ep,
                                fallback_gripper_width=no_gripper_obs_width,
                                stride=match_replay_stride,
                                max_samples=match_replay_max_samples,
                            )
                            live_tcp6 = _tcp6_from_obs(obs)
                            first_tcp6 = replay_actions[0, :6]
                            dpos = first_tcp6[:3] - live_tcp6[:3]
                            drot = (
                                st.Rotation.from_rotvec(first_tcp6[3:6])
                                * st.Rotation.from_rotvec(live_tcp6[3:6]).inv()
                            ).magnitude()
                            replay_dt = dt * max(1, int(match_replay_stride)) * max(
                                0.1, float(match_replay_duration_scale)
                            )
                            replay_start = time.time() + 0.25
                            replay_timestamps = (
                                np.arange(len(replay_actions), dtype=np.float64)
                                * replay_dt
                                + replay_start
                            )
                            print(
                                f"[match replay] episode={match_episode_id} "
                                f"samples={len(replay_actions)} "
                                f"stride={match_replay_stride} "
                                f"max_samples={match_replay_max_samples} "
                                f"duration_scale={match_replay_duration_scale}"
                            )
                            print(
                                "  first calibrated gap xyz(m):",
                                np.array2string(dpos, precision=5),
                                f"|d|={float(np.linalg.norm(dpos)):.5f}",
                                f" gap_rot={drot:.5f} rad",
                            )
                            if np.linalg.norm(dpos) > 0.10 or drot > 0.35:
                                print(
                                    "  replay aborted: first pose is too far from "
                                    "current robot pose. Teleop-align to the first "
                                    "overlay image, press g to confirm the calibrated "
                                    "gap is small, then press v again."
                                )
                                continue
                            print(
                                "  submitting selected zarr trajectory in robot frame; "
                                "Esc/Ctrl+C or robot E-stop if motion is wrong."
                            )
                            if direct_gripper is not None and len(replay_actions) > 0:
                                clipped, _ = direct_gripper.command_width(
                                    float(replay_actions[0, 6]), force=True
                                )
                                no_gripper_obs_width = _sanitize_gripper_width(
                                    clipped,
                                    max_gripper_width,
                                    tag="match replay direct gripper start",
                                )
                                replay_actions[:, 6] = np.clip(
                                    replay_actions[:, 6],
                                    direct_gripper.width_min_m,
                                    direct_gripper.width_max_m,
                                )
                            target_pose[0] = replay_actions[-1, :6]
                            gripper_target_pos[0] = float(replay_actions[-1, 6])
                            replay_log_dir = pathlib.Path(output).joinpath(
                                "eval_logs",
                                f"match_replay_ep{match_episode_id}_"
                                f"{time.strftime('%Y%m%d_%H%M%S')}",
                            )
                            replay_log_dir.mkdir(parents=True, exist_ok=True)
                            replay_video_path = replay_log_dir.joinpath("comparison.mp4")
                            print(f"[match replay] saving video to {replay_video_path}")

                            match_debug_data = _load_match_episode_debug_data(
                                match_zarr_path,
                                match_episode_id,
                            )
                            source_indices = np.arange(
                                0,
                                len(ep["robot0_eef_pos"]),
                                max(1, int(match_replay_stride)),
                                dtype=np.int64,
                            )
                            if match_replay_max_samples is not None and int(match_replay_max_samples) > 0:
                                source_indices = source_indices[:int(match_replay_max_samples)]
                            source_indices = source_indices[:len(replay_actions)]

                            # Full-episode replay can be hundreds/thousands of
                            # waypoints. Stream small chunks so the controller
                            # shared-memory queue does not fill up.
                            replay_start = time.time() + 0.25
                            replay_timestamps = (
                                np.arange(len(replay_actions), dtype=np.float64)
                                * replay_dt
                                + replay_start
                            )
                            next_submit_idx = 0
                            submit_chunk_size = 8
                            submit_horizon_s = max(0.35, min(1.0, replay_dt * 16.0))
                            video_writer = None
                            capture_dt = 1.0 / max(1.0, min(20.0, float(frequency)))
                            next_capture_t = time.time()
                            try:
                                while (
                                    time.time() <= float(replay_timestamps[-1]) + 0.1
                                    or next_submit_idx < len(replay_actions)
                                ):
                                    now = time.time()
                                    submit_until = now + submit_horizon_s
                                    while (
                                        next_submit_idx < len(replay_actions)
                                        and replay_timestamps[next_submit_idx] <= submit_until
                                    ):
                                        end_idx = next_submit_idx
                                        while (
                                            end_idx < len(replay_actions)
                                            and end_idx - next_submit_idx < submit_chunk_size
                                            and replay_timestamps[end_idx] <= submit_until
                                        ):
                                            end_idx += 1
                                        if end_idx == next_submit_idx:
                                            break
                                        try:
                                            env.exec_actions(
                                                actions=replay_actions[next_submit_idx:end_idx],
                                                timestamps=replay_timestamps[next_submit_idx:end_idx],
                                                compensate_latency=False,
                                            )
                                            next_submit_idx = end_idx
                                        except Exception as exc:
                                            if type(exc).__name__ == "Full":
                                                print(
                                                    "[match replay] controller queue full; "
                                                    "pausing waypoint submission briefly."
                                                )
                                                break
                                            raise
                                    if now < next_capture_t:
                                        time.sleep(min(0.01, next_capture_t - now))
                                        continue
                                    next_capture_t += capture_dt
                                    obs_video = env.get_obs(include_valve_context_stream=False)
                                    live_tcp6 = np.concatenate([
                                        obs_video["robot0_eef_pos"][-1],
                                        obs_video["robot0_eef_rot_axis_angle"][-1],
                                    ])
                                    replay_i = int(
                                        np.clip(
                                            np.searchsorted(replay_timestamps, now, side="right") - 1,
                                            0,
                                            len(replay_actions) - 1,
                                        )
                                    )
                                    source_idx = (
                                        int(source_indices[replay_i])
                                        if replay_i < len(source_indices)
                                        else replay_i
                                    )
                                    original_rgb = None
                                    original_tcp6 = None
                                    robot_tcp6 = replay_actions[replay_i, :6]
                                    if match_debug_data is not None:
                                        source_idx = int(
                                            np.clip(
                                                source_idx,
                                                0,
                                                len(match_debug_data["raw_pose6"]) - 1,
                                            )
                                        )
                                        original_tcp6 = match_debug_data["raw_pose6"][source_idx]
                                        robot_tcp6 = match_debug_data["robot_pose6"][source_idx]
                                        if match_debug_data.get("rgb") is not None:
                                            original_rgb = match_debug_data["rgb"][source_idx]
                                    current_bgr = _policy_input_bgr_from_obs(obs_video)
                                    if current_bgr is None:
                                        current_bgr = _get_live_display_bgr(
                                            env, camera_idx=match_camera
                                        )
                                    frame = _render_eval_video_frame(
                                        original_rgb,
                                        current_bgr,
                                        original_tcp6,
                                        robot_tcp6,
                                        live_tcp6,
                                        source_idx=source_idx,
                                        match_episode_id=match_episode_id,
                                    )
                                    if video_writer is None:
                                        fh, fw = frame.shape[:2]
                                        video_writer = cv2.VideoWriter(
                                            str(replay_video_path),
                                            cv2.VideoWriter_fourcc(*"mp4v"),
                                            max(1.0, min(20.0, float(frequency))),
                                            (fw, fh),
                                        )
                                        if not video_writer.isOpened():
                                            print(
                                                "[match replay] WARNING: failed to open "
                                                f"video writer: {replay_video_path}"
                                            )
                                            break
                                    video_writer.write(frame)
                            finally:
                                if video_writer is not None:
                                    video_writer.release()
                                    if replay_video_path.exists():
                                        print(
                                            "[match replay] wrote video: "
                                            f"{replay_video_path} "
                                            f"({replay_video_path.stat().st_size / 1024 / 1024:.2f} MB)"
                                        )
                        elif key == ord("t"):
                            live_tcp6 = _tcp6_from_obs(obs)
                            _SAVED_START_POSE_PATH.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            with open(_SAVED_START_POSE_PATH, "w") as f:
                                yaml.safe_dump(
                                    {"tcp6": [float(v) for v in live_tcp6]}, f
                                )
                            saved_start_tcp6 = np.asarray(
                                live_tcp6, dtype=np.float64
                            )
                            print(
                                f"saved start pose -> {_SAVED_START_POSE_PATH}: "
                                f"{np.round(live_tcp6, 4).tolist()}"
                            )
                        elif key == ord("p"):
                            if saved_start_tcp6 is None:
                                print(
                                    "no saved start pose. Press s at the desired "
                                    f"pose first (writes {_SAVED_START_POSE_PATH})."
                                )
                            elif plan_only:
                                print("[plan_only] skipped saved-start robot move.")
                            else:
                                pose = saved_start_tcp6.copy()
                                live_pose = _tcp6_from_obs(obs)
                                start_pos_gap = float(
                                    np.linalg.norm(pose[:3] - live_pose[:3])
                                )
                                start_rot_gap = float(
                                    (
                                        st.Rotation.from_rotvec(pose[3:6])
                                        * st.Rotation.from_rotvec(
                                            live_pose[3:6]
                                        ).inv()
                                    ).magnitude()
                                )
                                if (
                                    start_pos_gap
                                    > 10.0 * motion_safety_cfg.max_position_delta_m
                                    or start_rot_gap
                                    > 5.0 * motion_safety_cfg.max_rotation_delta_rad
                                ):
                                    print(
                                        "saved-start move refused: current gap "
                                        f"position={start_pos_gap:.4f} m, "
                                        f"rotation={start_rot_gap:.4f} rad. "
                                        "Teleop closer or save a new start pose."
                                    )
                                    continue
                                start_move_safety_cfg = PolicyMotionSafetyConfig(
                                    **{
                                        **vars(motion_safety_cfg),
                                        "max_position_delta_m": 10.0
                                        * motion_safety_cfg.max_position_delta_m,
                                        "max_rotation_delta_rad": 5.0
                                        * motion_safety_cfg.max_rotation_delta_rad,
                                    }
                                )
                                try:
                                    validate_policy_waypoints(
                                        np.concatenate(
                                            [pose, [gripper_target_pos[0]]]
                                        )[None],
                                        live_pose,
                                        gripper_target_pos[0],
                                        start_move_safety_cfg,
                                    )
                                except PolicySafetyError as exc:
                                    print(f"saved-start move refused: {exc}")
                                    continue
                                if not click.confirm(
                                    "Move the robot to the saved start pose?",
                                    default=False,
                                ):
                                    continue
                                duration = 4.0
                                print(
                                    f"moving to saved start pose over {duration}s: "
                                    f"{np.round(pose, 4).tolist()}"
                                )
                                if hasattr(env.robot, "servoL"):
                                    env.robot.servoL(pose, duration=duration)
                                else:
                                    env.robot.schedule_waypoint(
                                        pose, target_time=time.time() + duration
                                    )
                                target_pose[0] = pose
                                time.sleep(duration)
                        elif key == 8:
                            if click.confirm("Are you sure to drop an episode?"):
                                env.drop_episode()

                        if key not in keyboard_motion_keys:
                            teleop_motion_latch_armed = True

                        if start_policy:
                            selected_match_episode_for_eval = match_episode_id
                            if policy_image_audit_enabled:
                                policy_image_audit_printed = False
                            if coord_transform_audit_enabled:
                                coord_transform_audit_printed = False
                            break

                        precise_wait(t_sample)
                        if (not no_spacemouse) and (sm is not None):
                            sm_state = sm.get_motion_state_transformed()
                            dpos = sm_state[:3] * (0.5 / frequency)
                            drot_xyz = sm_state[3:] * (1.5 / frequency)
                            # Ignore sensor noise so idle SpaceMouse does not arm the robot.
                            if np.linalg.norm(dpos) < 2e-4:
                                dpos = np.zeros(3)
                            if np.linalg.norm(drot_xyz) < 2e-4:
                                drot_xyz = np.zeros(3)
                            grip_delta = 0.0
                            if has_gripper_control and sm.is_button_pressed(0):
                                grip_delta = -gripper_speed / frequency
                            if has_gripper_control and sm.is_button_pressed(1):
                                grip_delta = gripper_speed / frequency
                        else:
                            dpos = np.zeros(3)
                            drot_xyz = np.zeros(3)
                            grip_delta = 0.0
                            pos_step = 0.10 / frequency
                            rot_step = 1.00 / frequency
                            if (
                                key in keyboard_motion_keys
                                and teleop_motion_latch_armed
                            ):
                                teleop_motion_latch_armed = False
                                if key == ord("a"):
                                    dpos[0] += pos_step
                                elif key == ord("d"):
                                    dpos[0] -= pos_step
                                elif key == ord("s"):
                                    dpos[1] += pos_step
                                elif key == ord("w"):
                                    dpos[1] -= pos_step
                                elif key == ord("e"):
                                    dpos[2] += pos_step
                                elif key == ord("q"):
                                    dpos[2] -= pos_step
                                elif key == ord("j"):
                                    drot_xyz[0] -= rot_step
                                elif key == ord("l"):
                                    drot_xyz[0] += rot_step
                                elif key == ord("i"):
                                    drot_xyz[1] += rot_step
                                elif key == ord("k"):
                                    drot_xyz[1] -= rot_step
                                elif key == ord("u"):
                                    drot_xyz[2] -= rot_step
                                elif key == ord("o"):
                                    drot_xyz[2] += rot_step
                                elif has_gripper_control and key == ord("z"):
                                    grip_delta = -gripper_speed / frequency
                                elif has_gripper_control and key == ord("x"):
                                    grip_delta = gripper_speed / frequency

                        target_pose[0, :3] += dpos
                        target_pose[0, 3:] = _human_teleop_compose_rotvec(
                            target_pose[0, 3:],
                            drot_xyz,
                            teleop_euler_seq,
                            teleop_euler_extrinsic,
                        )
    
                        if has_gripper_control:
                            gripper_target_pos[0] = np.clip(
                                gripper_target_pos[0] + grip_delta, 0, max_gripper_width)
                        else:
                            gripper_target_pos[0] = 0.0
    
                        action = np.zeros((7,))
                        action[:6] = target_pose[0]
                        action[6] = gripper_target_pos[0]
    
    
                        # Only send command when there is an explicit human input.
                        has_motion_cmd = (np.linalg.norm(dpos) > 1e-9) or (np.linalg.norm(drot_xyz) > 1e-9)
                        has_grip_cmd = abs(grip_delta) > 1e-9
                        if (has_motion_cmd or has_grip_cmd) and plan_only:
                            print("[plan_only] skipped manual robot/gripper command.")
                        elif has_motion_cmd or has_grip_cmd:
                            if has_grip_cmd and direct_gripper is not None:
                                clipped, _ = direct_gripper.command_width(
                                    float(action[6])
                                )
                                no_gripper_obs_width = _sanitize_gripper_width(
                                    clipped,
                                    max_gripper_width,
                                    tag="human direct gripper feedback",
                                )
                                action[6] = clipped
                            cur_tcp = np.concatenate(
                                [
                                    obs["robot0_eef_pos"][-1],
                                    obs["robot0_eef_rot_axis_angle"][-1],
                                ]
                            )
                            tgt_tcp = np.asarray(action[:6], dtype=np.float64)
                            print(
                                "[teleop] current_tcp xyz(m) rotvec(rad):",
                                np.array2string(cur_tcp, precision=5),
                            )
                            print(
                                "[teleop] target_tcp xyz(m) rotvec(rad):",
                                np.array2string(tgt_tcp, precision=5),
                            )
                            print(
                                "[teleop] grip_cmd width(m):",
                                float(action[6]),
                                "delta_this_frame:",
                                float(grip_delta),
                            )
                            env.exec_actions(
                                actions=[action], 
                                timestamps=[t_command_target-time.monotonic()+time.time()],
                                compensate_latency=False)
                        precise_wait(t_cycle_end)
                        iter_idx += 1
                
                except KeyboardInterrupt:
                    print("Interrupted (Ctrl+C). Flushing episode and exiting.")
                    try:
                        env.hold_robot()
                        env.end_episode()
                    except Exception:
                        pass
                    raise

                # ========== policy control loop ==============
                eval_csv_file = None
                eval_csv_writer = None
                eval_csv_header_written = False
                eval_video_writer = None
                eval_log_dir = None
                ft_input_csv_file = None
                ft_input_csv_writer = None
                policy_output_csv_file = None
                policy_output_csv_writer = None
                scheduled_action_csv_file = None
                scheduled_action_csv_writer = None
                valve_context_csv_file = None
                valve_context_csv_writer = None
                context_input_capture = None
                policy_input_capture = None
                context_worker = None
                fusion_attention_csv_file = None
                fusion_attention_csv_writer = None
                ft_timeline_rows = []
                fusion_attention_sum = None
                fusion_attention_count = 0
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # ``get_obs`` uses camera-latency-compensated timestamps,
                    # so its first policy image may be slightly earlier than
                    # eval_t_start.  Keep a bounded live-camera preroll in the
                    # classifier only; episode recording still begins exactly
                    # at eval_t_start below.
                    context_preroll_s = 0.50
                    context_history_start_s = eval_t_start - context_preroll_s
                    if valve_context_runtime is not None:
                        # The classifier's recurrent/image history is episode
                        # local, just as it was when the context sidecar was
                        # generated for training.
                        valve_context_runtime.reset(
                            episode_start_timestamp_s=context_history_start_s
                        )

                    # per-run CSV + comparison-video logging (policy input / model
                    # output / actually-transmitted action), one folder per episode
                    eval_episode_id = env.replay_buffer.n_episodes
                    eval_log_dir = pathlib.Path(output).joinpath(
                        'eval_logs', f'ep{eval_episode_id}_{time.strftime("%Y%m%d_%H%M%S")}')
                    eval_log_dir.mkdir(parents=True, exist_ok=True)
                    eval_csv_file = open(eval_log_dir.joinpath('log.csv'), 'w', newline='')
                    eval_csv_writer = csv.writer(eval_csv_file)
                    policy_output_csv_file = open(
                        eval_log_dir.joinpath('policy_outputs.csv'), 'w', newline=''
                    )
                    policy_output_csv_writer = csv.writer(policy_output_csv_file)
                    policy_output_csv_writer.writerow(
                        ['iter_idx', 'wall_time', 'horizon_idx']
                        + [f'raw_{name}' for name in _POSE10D_LABELS]
                        + [
                            'decoded_tcp_x_m', 'decoded_tcp_y_m', 'decoded_tcp_z_m',
                            'decoded_tcp_rx_rad', 'decoded_tcp_ry_rad', 'decoded_tcp_rz_rad',
                            'decoded_gripper_width_m',
                        ]
                    )
                    scheduled_action_csv_file = open(
                        eval_log_dir.joinpath('scheduled_actions.csv'), 'w', newline=''
                    )
                    scheduled_action_csv_writer = csv.writer(scheduled_action_csv_file)
                    scheduled_action_csv_writer.writerow(
                        [
                            'iter_idx', 'wall_time', 'source_horizon_idx',
                            'target_timestamp', 'will_send_to_robot',
                            'sent_tcp_x_m', 'sent_tcp_y_m', 'sent_tcp_z_m',
                            'sent_tcp_rx_rad', 'sent_tcp_ry_rad', 'sent_tcp_rz_rad',
                            'sent_gripper_width_m', 'predicted_grasp_force_n',
                        ]
                    )
                    if valve_context_runtime is not None:
                        valve_context_csv_file = open(
                            eval_log_dir.joinpath('valve_context.csv'), 'w', newline=''
                        )
                        valve_context_csv_writer = csv.writer(valve_context_csv_file)
                        valve_context_csv_writer.writerow(
                            [
                                'iter_idx', 'wall_time', 'classifier_timestamp_s',
                                'phase', 'error_reason', 'warmed_up',
                                *_context_log_value_columns(
                                    valve_context_runtime.context_schema
                                ),
                            ]
                        )
                        if save_context_inputs:
                            context_input_capture = _ValveContextInputCapture(
                                eval_log_dir.joinpath('context_inputs'),
                                # Persist the classifier's bounded episode-local
                                # preroll too: those samples can be selected by
                                # the first 16-step temporal windows.
                                episode_start_timestamp_s=context_history_start_s,
                                context_schema=valve_context_runtime.context_schema,
                            )
                        if isinstance(
                            valve_context_runtime,
                            RGBForceValveContextRuntime,
                        ):
                            print(
                                "[valve_context] RGB/F-T observer runs once per "
                                "policy cycle from exact retained sensor history."
                            )
                        else:
                            context_worker = _ValveContextWorker(
                                valve_context_runtime,
                                env.get_valve_context_stream,
                                input_capture=context_input_capture,
                                poll_hz=60.0,
                                max_cached_records=256,
                                history_frames=120,
                                initial_history_frames=(
                                    4
                                    if valve_context_runtime.context_schema
                                    == VALVE_CONTEXT_V2_SCHEMA
                                    else 16
                                ),
                                anchor_recovery_history_frames=32,
                            )
                            context_worker.start(
                                episode_start_timestamp_s=context_history_start_s
                            )
                    if save_policy_inputs:
                        policy_input_capture = _PolicyInputCapture(
                            eval_log_dir.joinpath('policy_inputs'),
                            shape_meta=cfg.task.shape_meta,
                            episode_start_timestamp_s=eval_t_start,
                        )
                        print(
                            "[eval_log] policy input capture enabled: exact "
                            "pre-normalizer RGB/TCP/F-T/context NumPy samples "
                            "will be saved per policy inference."
                        )
                    if dual_ft_enabled:
                        ft_input_csv_file = open(
                            eval_log_dir.joinpath('input_ft_history.csv'), 'w', newline=''
                        )
                        ft_input_csv_writer = csv.writer(ft_input_csv_file)
                        ft_input_csv_writer.writerow(
                            [
                                'iter_idx', 'wall_time', 'finger', 'sample_idx_oldest_to_latest',
                                'is_latest_causal_sample', 'source_timestamp',
                            ]
                            + [f'physical_{name}_{unit}' for name, unit in zip(_FT_CHANNEL_LABELS, _FT_CHANNEL_UNITS)]
                            + [f'normalized_{name}' for name in _FT_CHANNEL_LABELS]
                        )
                    if fusion_attention_enabled:
                        fusion_attention_csv_file = open(
                            eval_log_dir.joinpath('fusion_attention.csv'), 'w', newline=''
                        )
                        fusion_attention_csv_writer = csv.writer(fusion_attention_csv_file)
                        fusion_attention_csv_writer.writerow(
                            ['iter_idx', 'wall_time', 'head', 'query_token', 'key_token', 'weight']
                        )
                    diagnostic_manifest = {
                        'format_version': 2,
                        'video': 'comparison.mp4',
                        'ft_input_history': 'input_ft_history.csv' if dual_ft_enabled else None,
                        'ft_input_plot': 'input_ft_timeline.png' if dual_ft_enabled else None,
                        'policy_outputs': 'policy_outputs.csv',
                        'scheduled_actions': 'scheduled_actions.csv',
                        'valve_context': (
                            'valve_context.csv' if valve_context_runtime is not None else None
                        ),
                        'fusion_attention': (
                            'fusion_attention.csv' if fusion_attention_enabled else None
                        ),
                        'fusion_attention_note': (
                            'query-to-key self-attention, descriptive only; not causal attribution'
                            if fusion_attention_enabled else None
                        ),
                        'checkpoint': str(input),
                        'robot_config': str(robot_config),
                        'plan_only': bool(plan_only),
                        'action_scale': float(action_scale),
                        'motion_momentum_previous_weight': float(
                            motion_momentum_previous_weight
                        ),
                        'valve_context_enabled': bool(valve_context_enabled),
                        'context_inputs': (
                            {
                                'directory': 'context_inputs',
                                'frames': 'context_inputs/context_frames.csv',
                                'wrenches': 'context_inputs/context_wrenches.csv',
                                'images': 'context_inputs/images/frame_*.png',
                                'classifier_windows': (
                                    'context_inputs/classifier_windows/window_*.npz'
                                ),
                                'classifier_windows_index': (
                                    'context_inputs/classifier_windows/index.csv'
                                ),
                                'note': (
                                    'Every frozen-classifier temporal input is saved using '
                                    'the variant documented by capture_manifest.json. The '
                                    'RGB/F-T override stores two RGB references and one '
                                    'native causal F/T history with no TCP/gripper input; '
                                    'legacy observers retain their lowdim/mask fields. No '
                                    'fabricated IMU input is emitted.'
                                ),
                            }
                            if context_input_capture is not None else None
                        ),
                        'context_worker': (
                            {
                                'poll_hz': 60.0,
                                'camera_history_frames': 120,
                                'initial_history_frames': (
                                    4
                                    if valve_context_runtime.context_schema
                                    == VALVE_CONTEXT_V2_SCHEMA else 16
                                ),
                                'anchor_recovery_history_frames': 32,
                                'preroll_s': context_preroll_s,
                                'max_cached_records': 256,
                                'policy_anchor_rule': (
                                    'exact RGB timestamp and SHA-256 fingerprint must '
                                    'match the worker record'
                                ),
                            }
                            if context_worker is not None else None
                        ),
                        'policy_inputs': (
                            {
                                'directory': 'policy_inputs',
                                'samples': 'policy_inputs/samples/sample_*.npz',
                                'images': 'policy_inputs/images/sample_*_camera0_rgb_t*.png',
                                'index': 'policy_inputs/index.csv',
                                'ft_history': 'policy_inputs/ft_history.csv',
                                'input_key_rule': 'every and only shape_meta.obs key',
                            }
                            if policy_input_capture is not None else None
                        ),
                        'valve_classifier_checkpoint': (
                            str(valve_context_runtime.checkpoint_path)
                            if valve_context_runtime is not None else None
                        ),
                        'valve_classifier_sha256': (
                            valve_context_runtime.checkpoint_sha256
                            if valve_context_runtime is not None else None
                        ),
                        'valve_classifier_device': (
                            valve_context_runtime.device
                            if valve_context_runtime is not None else None
                        ),
                        'valve_classifier_override': (
                            valve_classifier_override_used
                            if valve_context_runtime is not None else False
                        ),
                        'valve_classifier_expected_sha256': (
                            expected_valve_classifier_sha256
                            if valve_context_runtime is not None else None
                        ),
                    }
                    eval_log_dir.joinpath('diagnostic_manifest.json').write_text(
                        json.dumps(diagnostic_manifest, indent=2), encoding='utf-8'
                    )
                    print(f"[eval_log] logging to {eval_log_dir}")
                    match_debug_data = _load_match_episode_debug_data(
                        match_zarr_path,
                        selected_match_episode_for_eval,
                    )
                    if match_debug_data is not None:
                        print(
                            "[eval_log] comparison.mp4 source: original match "
                            f"episode {match_debug_data['episode']}"
                        )
                    else:
                        print(
                            "[eval_log] comparison.mp4 source: original match "
                            "episode unavailable; current image + coordinates only"
                        )

                    # get current pose
                    obs = env.get_obs(include_valve_context_stream=False)
                    if no_gripper:
                        obs = _with_synthetic_gripper_width(
                            obs,
                            no_gripper_obs_width,
                            fallback=max_gripper_width,
                        )
                    episode_start_pose = [
                        np.concatenate([
                            obs['robot0_eef_pos'],
                            obs['robot0_eef_rot_axis_angle'],
                        ], axis=-1)[-1]
                    ]
                    episode_start_pose_for_model = episode_start_pose
                    if not dual_ft_enabled:
                        episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(
                            episode_start_pose
                        )

                    # Persist a frame before the first inference.  This makes
                    # comparison.mp4 available even when waypoint safety rejects
                    # the first model output before the regular end-of-loop video
                    # logging is reached.
                    initial_eval_frame = _make_eval_comparison_frame(
                        obs,
                        match_debug_data,
                        source_idx=0,
                        env=env,
                        camera_idx=match_camera,
                    )
                    fh, fw = initial_eval_frame.shape[:2]
                    eval_video_writer = cv2.VideoWriter(
                        str(eval_log_dir.joinpath('comparison.mp4')),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        max(1.0, 1.0 / dt),
                        (fw, fh),
                    )
                    if eval_video_writer.isOpened():
                        eval_video_writer.write(initial_eval_frame)
                    else:
                        print("[eval_log] could not open comparison.mp4 for writing")
                        eval_video_writer.release()
                        eval_video_writer = None

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    policy_iter_count = 0
                    perv_target_pose = None
                    runtime_metrics = {
                        "attempted_cycles": 0,
                        "completed_cycles": 0,
                        "valid_observations": 0,
                        "safety_rejections": 0,
                        "context_recovery_skips": 0,
                        "ft_left_age": [],
                        "ft_right_age": [],
                        "safety_ft_age": [],
                        "obs_assembly": [],
                        "inference": [],
                        "loop": [],
                        "deadline_misses": 0,
                        "robot_command_calls": 0,
                    }
                    consecutive_context_recovery_skips = 0
                    while True:
                        runtime_metrics["attempted_cycles"] += 1
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        t_loop_start = time.perf_counter()
                        obs = env.get_obs(include_valve_context_stream=False)
                        if no_gripper:
                            obs = _with_synthetic_gripper_width(
                                obs,
                                no_gripper_obs_width,
                                fallback=max_gripper_width,
                            )
                        obs_timestamps = obs['timestamp']
                        runtime_metrics["obs_assembly"].append(
                            time.perf_counter() - t_loop_start
                        )
                        runtime_metrics["valid_observations"] += 1
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        if dual_ft_enabled:
                            runtime_metrics["ft_left_age"].append(
                                float(obs["robot0_ft_left_age"])
                            )
                            runtime_metrics["ft_right_age"].append(
                                float(obs["robot0_ft_right_age"])
                            )
                            print(
                                "Dual-F/T causal timing: "
                                f"anchor={float(obs_timestamps[-1]):.9f} "
                                f"left_last={float(obs['robot0_ft_left_timestamps'][-1]):.9f} "
                                f"left_age_ms={float(obs['robot0_ft_left_age']) * 1000.0:.3f} "
                                f"right_last={float(obs['robot0_ft_right_timestamps'][-1]):.9f} "
                                f"right_age_ms={float(obs['robot0_ft_right_age']) * 1000.0:.3f}"
                            )

                        # run inference
                        with torch.inference_mode():
                            s = time.time()
                            valve_context_record = None
                            obs_with_context = obs
                            context_recovery_reason = None
                            if valve_context_runtime is not None:
                                try:
                                    if isinstance(
                                        valve_context_runtime,
                                        RGBForceValveContextRuntime,
                                    ):
                                        context_stream = (
                                            env.get_valve_context_stream(
                                                history_frames=32
                                            )
                                        )
                                        (
                                            obs_with_context,
                                            valve_context_record,
                                        ) = (
                                            _add_rgb_force_context_to_policy_observation(
                                                obs,
                                                runtime=valve_context_runtime,
                                                stream=context_stream,
                                                input_capture=context_input_capture,
                                            )
                                        )
                                    else:
                                        if context_worker is None:
                                            raise RuntimeError(
                                                "stateful context checkpoint "
                                                "started without its camera-rate worker"
                                            )
                                        policy_rgb = _policy_input_rgb_from_obs(obs)
                                        if policy_rgb is None:
                                            raise RuntimeError(
                                                "policy observation has no final "
                                                "camera0_rgb frame"
                                            )
                                        valve_context_record = (
                                            context_worker.record_for_policy_anchor(
                                                timestamp_s=float(
                                                    obs_timestamps[-1]
                                                ),
                                                rgb=policy_rgb,
                                                timeout_s=1.00,
                                            )
                                        )
                                        obs_with_context = dict(obs)
                                        obs_with_context['valve_context'] = (
                                            valve_context_record.values.reshape(
                                                1, -1
                                            )
                                        )
                                    if not valve_context_record.warmed_up:
                                        context_recovery_reason = "context_valid=0"
                                except TimeoutError as exc:
                                    context_recovery_reason = (
                                        "exact policy anchor unavailable: "
                                        f"{exc}"
                                    )
                                except Exception as exc:
                                    raise PolicySafetyError(
                                        "valve-context input/prediction failed: "
                                        f"{exc}"
                                    ) from exc
                                if context_recovery_reason is not None:
                                    consecutive_context_recovery_skips = (
                                        _register_context_recovery_skip(
                                            env,
                                            runtime_metrics,
                                            plan_only=plan_only,
                                            consecutive_skips=(
                                                consecutive_context_recovery_skips
                                            ),
                                            reason=context_recovery_reason,
                                        )
                                    )
                                    runtime_metrics["loop"].append(
                                        time.perf_counter() - t_loop_start
                                    )
                                    if time.time() - eval_t_start > max_duration:
                                        print(
                                            "Max Duration reached during "
                                            "context recovery."
                                        )
                                        env.end_episode()
                                        break
                                    precise_wait(
                                        time.time()
                                        + max(
                                            frame_latency,
                                            (
                                                2.0 * context_worker.poll_period_s
                                                if context_worker is not None
                                                else 2.0 / 60.0
                                            ),
                                        )
                                    )
                                    continue
                                consecutive_context_recovery_skips = 0
                            obs_for_model = prepare_rg2ft_policy_obs(
                                obs_with_context, cfg.task.shape_meta
                            )
                            if not dual_ft_enabled:
                                obs_for_model = _apply_slam_frame_fix_to_obs(
                                    obs_for_model, n_robots
                                )
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs_for_model, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=None,
                                episode_start_pose=episode_start_pose_for_model)
                            obs_dict_np = _policy_obs_float32(obs_dict_np)
                            _check_policy_inputs_finite(
                                obs_dict_np, f"[policy iter={iter_idx}]"
                            )
                            if policy_input_capture is not None:
                                try:
                                    policy_input_capture.append(
                                        policy_iter_idx=iter_idx,
                                        policy_anchor_timestamp_s=float(
                                            obs_timestamps[-1]
                                        ),
                                        obs_dict_np=obs_dict_np,
                                        source_obs=obs,
                                    )
                                except Exception as exc:
                                    raise PolicySafetyError(
                                        "policy input capture failed before "
                                        f"inference: {exc}"
                                    ) from exc
                            if policy_image_audit_enabled and not policy_image_audit_printed:
                                train_rgb = None
                                if selected_match_episode_for_eval is not None:
                                    train_rgb = episode_first_policy_frame_map.get(
                                        int(selected_match_episode_for_eval)
                                    )
                                _print_policy_image_audit(
                                    obs_for_model,
                                    obs_dict_np,
                                    cfg.task.shape_meta,
                                    f"[policy iter={iter_idx}]",
                                    train_rgb=train_rgb,
                                    train_info=match_policy_image_info,
                                )
                                policy_image_audit_printed = True
                            if print_model_input:
                                _print_model_input_debug(
                                    obs_dict_np,
                                    obs,
                                    episode_start_pose,
                                    obs_pose_rep,
                                    f"[policy iter={iter_idx}]",
                                )
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            if 'context' in result:
                                from diffusion_policy.context.runtime import log_context
                                log_context(result, obs_timestamps[-1], iter_idx, eval_log_dir,
                                    int(getattr(policy, 'context_config', {}).get('log_every', 10)))
                            raw_action = result["action_pred"][0].detach().to("cpu").numpy()
                            raw_model_action = raw_action.copy()
                            expected_action_shape = (
                                int(checkpoint_contract["action_horizon"]),
                                int(checkpoint_contract["action_dim"]) * n_robots,
                            )
                            if raw_action.shape != expected_action_shape:
                                raise RuntimeError(
                                    "policy must return its full trained trajectory; got "
                                    f"{raw_action.shape}, expected {expected_action_shape}"
                                )
                            _check_finite_array(
                                f"[policy iter={iter_idx}] raw action_pred",
                                raw_action,
                            )
                            predicted_force_reference = None
                            if force_feedback is not None:
                                predicted_force_reference = raw_action[:, 10].copy()
                                _check_finite_array(
                                    f"[policy iter={iter_idx}] predicted grasp force reference",
                                    predicted_force_reference,
                                )
                            if not dual_ft_enabled:
                                raw_action = _apply_slam_frame_fix(
                                    raw_action, n_robots
                                )
                            action_dataset = _decode_real_umi_action_checked(
                                raw_action,
                                obs_for_model,
                                action_pose_repr,
                                f"[policy iter={iter_idx} dataset]",
                            )
                            action = action_dataset
                            if not dual_ft_enabled:
                                action = _transform_tcp7_action(
                                    action_dataset, _ROBOT_FROM_DATASET_T, n_robots
                                )
                            _check_finite_array(
                                f"[policy iter={iter_idx}] robot-frame tcp7 action",
                                action,
                            )
                            action = _apply_policy_tcp7_rot_roundtrip(
                                action,
                                enabled=policy_rot_rt,
                                euler_seq=policy_rot_seq,
                                euler_extrinsic=policy_rot_ext,
                                n_robots=n_robots,
                            )
                            # These diagnostics are intentionally written before
                            # motion safety validation. A rejected first waypoint
                            # is exactly the case where its policy output and F/T
                            # input are needed for debugging.
                            fusion_attention_for_video = None
                            try:
                                diagnostic_wall_time = time.time()
                                if ft_input_csv_writer is not None:
                                    normalized_ft = _normalized_ft_policy_inputs(
                                        policy, obs_dict
                                    )
                                    ft_histories = _write_ft_input_history(
                                        ft_input_csv_writer,
                                        iter_idx=iter_idx,
                                        wall_time=diagnostic_wall_time,
                                        obs=obs,
                                        obs_dict_np=obs_dict_np,
                                        normalized_ft=normalized_ft,
                                    )
                                    if ft_histories is not None:
                                        ft_timeline_rows.append(
                                            {
                                                'iter_idx': int(iter_idx),
                                                'wall_time': diagnostic_wall_time,
                                                'left': ft_histories[0][-1].copy(),
                                                'right': ft_histories[1][-1].copy(),
                                            }
                                        )
                                    ft_input_csv_file.flush()
                                if policy_output_csv_writer is not None:
                                    for horizon_idx, (raw_row_full, decoded_row) in enumerate(
                                        zip(raw_model_action, action)
                                    ):
                                        policy_output_csv_writer.writerow(
                                            [iter_idx, diagnostic_wall_time, horizon_idx]
                                            + np.asarray(raw_row_full, dtype=np.float64).ravel().tolist()
                                            + np.asarray(decoded_row, dtype=np.float64).ravel().tolist()
                                        )
                                    policy_output_csv_file.flush()
                                if (
                                    valve_context_csv_writer is not None
                                    and valve_context_record is not None
                                ):
                                    valve_context_csv_writer.writerow(
                                        [
                                            iter_idx, diagnostic_wall_time,
                                            valve_context_record.timestamp_s,
                                            valve_context_record.phase_name,
                                            valve_context_record.error_reason_name,
                                            int(valve_context_record.warmed_up),
                                        ]
                                        + np.asarray(
                                            valve_context_record.values,
                                            dtype=np.float64,
                                        ).tolist()
                                    )
                                    valve_context_csv_file.flush()
                                if fusion_attention_csv_writer is not None:
                                    attention = getattr(
                                        policy.obs_encoder, 'last_fusion_attention', None
                                    )
                                    attention = _tensor_to_numpy(attention)
                                    if attention.ndim != 4 or attention.shape[0] < 1:
                                        raise ValueError(
                                            'captured fusion attention must be [B,heads,query,key], '
                                            f'got {attention.shape}'
                                        )
                                    attention = np.asarray(attention[0], dtype=np.float64)
                                    if attention.shape[1:] != (
                                        len(fusion_attention_tokens),
                                        len(fusion_attention_tokens),
                                    ):
                                        raise ValueError(
                                            'captured fusion-attention token shape does not match '
                                            f'{fusion_attention_tokens}: {attention.shape}'
                                        )
                                    for head_idx in range(attention.shape[0]):
                                        for query_idx, query_name in enumerate(fusion_attention_tokens):
                                            for key_idx, key_name in enumerate(fusion_attention_tokens):
                                                fusion_attention_csv_writer.writerow(
                                                    [
                                                        iter_idx, diagnostic_wall_time, head_idx,
                                                        query_name, key_name,
                                                        float(attention[head_idx, query_idx, key_idx]),
                                                    ]
                                                )
                                    batch_attention_sum = attention.sum(axis=0)
                                    fusion_attention_for_video = attention.mean(axis=0)
                                    fusion_attention_sum = (
                                        batch_attention_sum
                                        if fusion_attention_sum is None
                                        else fusion_attention_sum + batch_attention_sum
                                    )
                                    fusion_attention_count += int(attention.shape[0])
                                    fusion_attention_csv_file.flush()
                            except Exception as exc:
                                print(f"[eval_log] diagnostic capture failed at iter={iter_idx}: {exc}")
                            if (
                                coord_transform_audit_enabled
                                and not coord_transform_audit_printed
                            ):
                                _print_coord_transform_audit(
                                    f"[policy iter={iter_idx}]",
                                    obs,
                                    obs_for_model,
                                    action_dataset=action_dataset,
                                    action_robot=action,
                                    match_debug_data=_make_match_pose_debug_data(
                                        match_replay_buffer,
                                        selected_match_episode_for_eval,
                                    ),
                                    match_source_idx=0,
                                )
                                coord_transform_audit_printed = True
                            inference_latency = time.time() - s
                            runtime_metrics["inference"].append(inference_latency)
                            print("Inference latency:", inference_latency)
                            if pose_eval_audit:
                                _print_pose_z_audit(
                                    obs,
                                    action,
                                    action_pose_repr,
                                    iter_idx,
                                    n_robots,
                                    f"[pose_eval_audit iter={iter_idx}]",
                                    dataset_z_stats=dataset_z_stats,
                                    raw_action_pred=raw_action,
                                )
                            if print_policy_output:
                                _print_policy_action_debug(
                                    f"[policy iter={iter_idx}]",
                                    raw_action,
                                    action,
                                    submitted=None,
                                )
                        
                        # convert policy action to env actions. Use the near-term policy
                        # actions; scheduling late-horizon rows after inference latency
                        # causes jumpy biased motion.
                        n_exec = min(int(steps_per_inference), len(action))
                        this_target_poses = action[:n_exec].copy()
                        source_horizon_indices = np.arange(n_exec, dtype=np.int64)
                        this_force_reference = None
                        if predicted_force_reference is not None:
                            this_force_reference = predicted_force_reference[:n_exec].copy()
                        assert this_target_poses.shape[1] == 7 * n_robots

                        # deal with timing
                        # Schedule from the next available control tick. Basing these
                        # stamps on obs_timestamps[-1] can skip the first several
                        # near-term actions when obs + inference latency is high.
                        action_exec_latency = max(0.002, float(command_latency))
                        curr_time = time.time()
                        next_step_idx = int(np.ceil(
                            (curr_time + action_exec_latency - eval_t_start) / dt
                        ))
                        first_action_timestamp = eval_t_start + next_step_idx * dt
                        action_timestamps = (
                            np.arange(n_exec, dtype=np.float64) * dt
                            + first_action_timestamp
                        )
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        runtime_metrics["deadline_misses"] += int(np.sum(~is_new))
                        if np.sum(is_new) == 0:
                            # Keep the pose/reference row paired and schedule it
                            # strictly in the future even after a long inference.
                            this_target_poses = this_target_poses[[-1]]
                            if this_force_reference is not None:
                                this_force_reference = this_force_reference[[-1]]
                            source_horizon_indices = source_horizon_indices[[-1]]
                            action_timestamp = curr_time + action_exec_latency
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            if this_force_reference is not None:
                                this_force_reference = this_force_reference[is_new]
                            source_horizon_indices = source_horizon_indices[is_new]
                            action_timestamps = action_timestamps[is_new]

                        if (
                            tcp_delta_scale_vec is not None
                            or action_scale != 1.0
                            or freeze_rotation
                        ):
                            this_target_poses = _limit_policy_waypoints(
                                this_target_poses,
                                obs,
                                n_robots=n_robots,
                                tcp_delta_scales=tcp_delta_scale_vec,
                                action_scale=action_scale,
                                freeze_rotation=freeze_rotation,
                                freeze_rotation_ref_pose=episode_start_pose,
                            )

                        force_feedback_result = None
                        if force_feedback is not None:
                            feedback_ft = env.get_latest_ft_state()
                            raw_left = feedback_ft["left_raw"]
                            raw_right = feedback_ft["right_raw"]
                            force_feedback_result = force_feedback.correct_from_native_wrenches(
                                policy_width_m=this_target_poses[:, 6],
                                predicted_force_n=this_force_reference,
                                left_wrench=raw_left,
                                right_wrench=raw_right,
                                startup_bias_12d=startup_bias_12d,
                            )
                            this_target_poses[:, 6] = force_feedback_result[
                                "corrected_width_m"
                            ]
                            print(
                                "F/T width feedback: measured="
                                f"{force_feedback_result['measured_force_n']:.3f} N "
                                "target="
                                f"{float(this_force_reference[0]):.3f} N "
                                "width_correction="
                                f"{float(force_feedback_result['width_correction_m'][0]) * 1000.0:.3f} mm"
                            )

                        current_tcp6 = np.concatenate(
                            [
                                np.asarray(
                                    obs["robot0_eef_pos"][-1], dtype=np.float64
                                ),
                                np.asarray(
                                    obs["robot0_eef_rot_axis_angle"][-1],
                                    dtype=np.float64,
                                ),
                            ]
                        )
                        current_width_m = float(
                            np.asarray(
                                obs["robot0_gripper_width"][-1]
                            ).reshape(-1)[0]
                        )
                        current_pose_width7 = np.concatenate(
                            [current_tcp6, [current_width_m]]
                        )
                        if motion_momentum_previous_weight > 0.0:
                            this_target_poses = _apply_policy_motion_momentum(
                                this_target_poses,
                                current_pose_width7=current_pose_width7,
                                previous_weight=motion_momentum_previous_weight,
                            )

                        # Record this iteration before either F/T or waypoint
                        # safety validation.  A stopped run therefore contains
                        # the exact candidate output and attention that caused it.
                        eval_frame = None
                        try:
                            video_source_idx = None
                            if match_debug_data is not None:
                                elapsed_s = max(0.0, time.monotonic() - t_start)
                                video_source_idx = int(round(
                                    elapsed_s * float(match_debug_data["fps"])
                                ))
                            video_predicted_force = (
                                None if this_force_reference is None
                                else float(this_force_reference[0])
                            )
                            video_measured_force = (
                                None if force_feedback_result is None
                                else float(force_feedback_result["measured_force_n"])
                            )
                            video_width_correction = (
                                None if force_feedback_result is None
                                else float(force_feedback_result["width_correction_m"][0])
                            )
                            eval_frame = _make_eval_comparison_frame(
                                obs,
                                match_debug_data,
                                source_idx=video_source_idx,
                                env=env,
                                camera_idx=match_camera,
                                raw_action_h0=raw_model_action[0],
                                decoded_action_h0=action[0],
                                scheduled_action_h0=(
                                    this_target_poses[0]
                                    if len(this_target_poses) > 0 else None
                                ),
                                predicted_force_n=video_predicted_force,
                                measured_force_n=video_measured_force,
                                width_correction_m=video_width_correction,
                                valve_context_record=valve_context_record,
                                fusion_attention=fusion_attention_for_video,
                                fusion_attention_tokens=fusion_attention_tokens,
                            )
                            if eval_video_writer is None:
                                fh, fw = eval_frame.shape[:2]
                                eval_video_writer = cv2.VideoWriter(
                                    str(eval_log_dir.joinpath('comparison.mp4')),
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    max(1.0, 1.0 / dt), (fw, fh),
                                )
                            if eval_video_writer.isOpened():
                                eval_video_writer.write(eval_frame)
                            else:
                                print("[eval_log] could not open comparison.mp4 for writing")
                                eval_video_writer.release()
                                eval_video_writer = None
                        except Exception as exc:
                            print(f"[eval_log] failed to add output/attention video frame: {exc}")

                        validate_policy_waypoints(
                            this_target_poses,
                            current_tcp6,
                            current_width_m,
                            motion_safety_cfg,
                        )

                        if print_policy_output:
                            _print_policy_action_debug(
                                f"[policy iter={iter_idx} -> exec]",
                                raw_action,
                                action,
                                submitted=this_target_poses,
                            )
                        if print_motion_debug:
                            _print_motion_debug(
                                f"[motion iter={iter_idx}]",
                                obs,
                                this_target_poses,
                                timestamps=action_timestamps,
                                n_robots=n_robots,
                            )

                        if (
                            (not plan_only)
                            and direct_gripper is not None
                            and len(this_target_poses) > 0
                        ):
                            clipped_width, _ = direct_gripper.command_width(
                                float(this_target_poses[0, 6])
                            )
                            no_gripper_obs_width = _sanitize_gripper_width(
                                clipped_width,
                                max_gripper_width,
                                tag="policy direct gripper feedback",
                            )
                            this_target_poses[:, 6] = np.clip(
                                this_target_poses[:, 6],
                                direct_gripper.width_min_m,
                                direct_gripper.width_max_m,
                            )

                        # Read and validate a new F/T snapshot at the actual
                        # command boundary. No rendering, logging, or printing
                        # is allowed between this guard and exec_actions.
                        if plan_only:
                            if force_feedback_result is not None:
                                (
                                    _safety_ft,
                                    safety_grasp_force_n,
                                    safety_ft_age_s,
                                ) = read_and_validate_latest_ft(
                                    env,
                                    startup_bias_12d,
                                    ft_safety_cfg,
                                )
                            print(
                                "[plan_only] skipped exec_actions; "
                                "compare delta xyz above with teleop axes."
                            )
                        else:
                            if force_feedback_result is not None:
                                (
                                    safety_grasp_force_n,
                                    safety_ft_age_s,
                                ) = _exec_actions_with_fresh_ft_guard(
                                    env,
                                    this_target_poses,
                                    action_timestamps,
                                    startup_bias_12d,
                                    ft_safety_cfg,
                                )
                            else:
                                env.exec_actions(
                                    actions=this_target_poses,
                                    timestamps=action_timestamps,
                                    compensate_latency=False,
                                )
                            runtime_metrics["robot_command_calls"] += 1
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

                        if force_feedback_result is not None:
                            runtime_metrics["safety_ft_age"].append(
                                safety_ft_age_s
                            )
                            print(
                                "Pre-command F/T safety: "
                                f"age={safety_ft_age_s * 1000.0:.3f} ms "
                                f"grasp_force={safety_grasp_force_n:.3f} N"
                            )

                        if scheduled_action_csv_writer is not None:
                            scheduled_wall_time = time.time()
                            for scheduled_idx, target_row in enumerate(this_target_poses):
                                force_reference = float('nan')
                                if this_force_reference is not None:
                                    force_reference = float(this_force_reference[scheduled_idx])
                                scheduled_action_csv_writer.writerow(
                                    [
                                        iter_idx,
                                        scheduled_wall_time,
                                        int(source_horizon_indices[scheduled_idx]),
                                        float(action_timestamps[scheduled_idx]),
                                        int(not plan_only),
                                    ]
                                    + np.asarray(target_row, dtype=np.float64).ravel().tolist()
                                    + [force_reference]
                                )
                            scheduled_action_csv_file.flush()

                        # --- per-step eval logging (CSV + comparison video) ---
                        obs_pos = np.asarray(obs['robot0_eef_pos'][-1], dtype=np.float64).ravel()
                        obs_rot = np.asarray(obs['robot0_eef_rot_axis_angle'][-1], dtype=np.float64).ravel()
                        obs_grip = np.asarray(obs.get('robot0_gripper_width', [[0.0]])[-1], dtype=np.float64).ravel()
                        raw_row = np.asarray(raw_action[0], dtype=np.float64).ravel()
                        converted_row = np.asarray(action[0], dtype=np.float64).ravel()
                        sent_row = (np.asarray(this_target_poses[0], dtype=np.float64).ravel()
                            if len(this_target_poses) > 0
                            else converted_row)
                        accum_xyz_cm = (obs_pos - np.asarray(episode_start_pose[0][:3],
                            dtype=np.float64)) * 100.0
                        measured_force_n = (
                            float(force_feedback_result["measured_force_n"])
                            if force_feedback_result is not None else float("nan")
                        )
                        predicted_force_n = (
                            float(force_feedback_result["predicted_force_n"][0])
                            if force_feedback_result is not None else float("nan")
                        )
                        width_correction_m = (
                            float(force_feedback_result["width_correction_m"][0])
                            if force_feedback_result is not None else float("nan")
                        )

                        if not eval_csv_header_written:
                            eval_csv_writer.writerow(
                                ['iter_idx', 'wall_time']
                                + [f'obs_pos_{i}' for i in range(len(obs_pos))]
                                + [f'obs_rot_{i}' for i in range(len(obs_rot))]
                                + [f'obs_grip_{i}' for i in range(len(obs_grip))]
                                + [f'raw_action_{i}' for i in range(len(raw_row))]
                                + [f'converted_{i}' for i in range(len(converted_row))]
                                + [f'sent_{i}' for i in range(len(sent_row))]
                                + [
                                    'n_submitted',
                                    'measured_grasp_force_n',
                                    'predicted_grasp_force_n',
                                    'width_correction_m',
                                ]
                                + [f'accum_cm_{i}' for i in range(3)]
                            )
                            eval_csv_header_written = True
                        eval_csv_writer.writerow(
                            [iter_idx, time.time()]
                            + obs_pos.tolist() + obs_rot.tolist() + obs_grip.tolist()
                            + raw_row.tolist() + converted_row.tolist() + sent_row.tolist()
                            + [
                                len(this_target_poses),
                                measured_force_n,
                                predicted_force_n,
                                width_correction_m,
                            ]
                            + accum_xyz_cm.tolist()
                        )
                        eval_csv_file.flush()

                        # visualize (full-res camera feed; obs rgb is masked 224x224 for policy)
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = _get_live_display_bgr(env, camera_idx=match_camera)
                        match_policy_rgb = None
                        if selected_match_episode_for_eval is not None:
                            match_policy_rgb = episode_first_policy_frame_map.get(
                                int(selected_match_episode_for_eval)
                            )
                        if match_policy_rgb is not None:
                            vis_img = _blend_match_rgb_on_live_bgr(
                                vis_img,
                                match_policy_rgb,
                            )
                        header = "Episode: {}, Time: {:.1f}".format(
                            episode_id, time.monotonic() - t_start
                        )
                        if selected_match_episode_for_eval is not None:
                            header += (
                                " | Match ep: "
                                f"{int(selected_match_episode_for_eval)} | overlap 50/50"
                            )
                        if vis_pose:
                            next_tcp = (
                                this_target_poses[0][:6]
                                if len(this_target_poses) > 0
                                else None
                            )
                            vis_img = _overlay_pose_vis(
                                vis_img,
                                header=header,
                                cur_tcp6=_tcp6_from_obs(obs),
                                target_tcp6=next_tcp,
                                episode_origin_tcp6=episode_start_pose[0],
                            )
                        else:
                            vis_img = _overlay_episode_text(vis_img, header)
                        cv2.imshow("default", vis_img)
                        if show_policy_image:
                            _show_policy_input_window(
                                obs,
                                f"policy input | t={time.monotonic() - t_start:.1f}s",
                                match_rgb=match_policy_rgb,
                            )
                            if eval_frame is not None:
                                cv2.imshow("policy output + attention", eval_frame)
                        key = _poll_control_key(terminal_key_poller)
                        stop_episode = False
                        if key == ord("s"):
                            print("Stopped.")
                            stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        policy_iter_count += 1
                        runtime_metrics["completed_cycles"] += 1
                        runtime_metrics["loop"].append(
                            time.perf_counter() - t_loop_start
                        )
                        if max_policy_iters is not None and policy_iter_count >= max_policy_iters:
                            print(f"max_policy_iters={max_policy_iters} reached.")
                            stop_episode = True
                        if stop_episode:
                            if (
                                key != ord("s")
                                and (not plan_only)
                                and len(action_timestamps) > 0
                            ):
                                final_wait = float(action_timestamps[-1]) + dt
                                precise_wait(final_wait, time_func=time.time)
                            env.hold_robot()
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except PolicySafetyError as exc:
                    runtime_metrics["safety_rejections"] += 1
                    env.hold_robot()
                    env.end_episode()
                    raise click.ClickException(
                        f"policy safety stop (pending waypoints cancelled): {exc}"
                    ) from exc
                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.hold_robot()
                    env.end_episode()
                finally:
                    if eval_csv_file is not None:
                        eval_csv_file.close()
                    if ft_input_csv_file is not None:
                        ft_input_csv_file.close()
                    if policy_output_csv_file is not None:
                        policy_output_csv_file.close()
                    if scheduled_action_csv_file is not None:
                        scheduled_action_csv_file.close()
                    if valve_context_csv_file is not None:
                        valve_context_csv_file.close()
                    context_worker_summary = None
                    if context_worker is not None:
                        try:
                            context_worker.stop()
                        except Exception as exc:
                            print(f"[eval_log] failed to stop context worker: {exc}")
                        context_worker_summary = context_worker.summary()
                        if eval_log_dir is not None:
                            eval_log_dir.joinpath('context_worker_summary.json').write_text(
                                json.dumps(context_worker_summary, indent=2),
                                encoding='utf-8',
                            )
                    if context_input_capture is not None:
                        context_input_capture.close()
                    if policy_input_capture is not None:
                        policy_input_capture.close()
                    if fusion_attention_csv_file is not None:
                        fusion_attention_csv_file.close()
                    if eval_video_writer is not None:
                        eval_video_writer.release()
                    if eval_log_dir is not None:
                        saved_diagnostics = [
                            'log.csv', 'policy_outputs.csv', 'scheduled_actions.csv'
                        ]
                        if valve_context_csv_file is not None:
                            saved_diagnostics.append('valve_context.csv')
                        if context_input_capture is not None:
                            saved_diagnostics.append(
                                'context_inputs/ (all exact classifier temporal inputs + output)'
                            )
                        if policy_input_capture is not None:
                            saved_diagnostics.append(
                                'policy_inputs/ (exact RGB/TCP/F-T/context passed to policy)'
                            )
                        if context_worker_summary is not None:
                            saved_diagnostics.append('context_worker_summary.json')
                        if eval_video_writer is not None:
                            saved_diagnostics.append('comparison.mp4')
                        if ft_input_csv_file is not None:
                            saved_diagnostics.append('input_ft_history.csv')
                            try:
                                if _render_ft_input_timeline(
                                    eval_log_dir.joinpath('input_ft_timeline.png'),
                                    ft_timeline_rows,
                                ):
                                    saved_diagnostics.append('input_ft_timeline.png')
                            except Exception as exc:
                                print(f"[eval_log] failed to render F/T timeline: {exc}")
                        if fusion_attention_count > 0 and fusion_attention_sum is not None:
                            mean_attention = fusion_attention_sum / float(
                                fusion_attention_count
                            )
                            received_attention = mean_attention.mean(axis=0)
                            emitted_attention = mean_attention.mean(axis=1)
                            attention_summary = {
                                'kind': 'Dual-F/T fusion self-attention',
                                'interpretation': (
                                    'Rows are query tokens and columns are key tokens. '
                                    'Attention is descriptive only and must not be treated as '
                                    'causal action attribution.'
                                ),
                                'token_order': fusion_attention_tokens,
                                'head_matrices_averaged': int(fusion_attention_count),
                                'mean_query_to_key': mean_attention.tolist(),
                                'mean_attention_received_by_key': {
                                    name: float(value)
                                    for name, value in zip(
                                        fusion_attention_tokens, received_attention
                                    )
                                },
                                'mean_attention_emitted_by_query': {
                                    name: float(value)
                                    for name, value in zip(
                                        fusion_attention_tokens, emitted_attention
                                    )
                                },
                            }
                            try:
                                eval_log_dir.joinpath('fusion_attention_summary.json').write_text(
                                    json.dumps(attention_summary, indent=2), encoding='utf-8'
                                )
                                saved_diagnostics.extend(
                                    ['fusion_attention.csv', 'fusion_attention_summary.json']
                                )
                                if _render_fusion_attention_heatmap(
                                    eval_log_dir.joinpath('fusion_attention_mean.png'),
                                    fusion_attention_tokens,
                                    mean_attention,
                                ):
                                    saved_diagnostics.append('fusion_attention_mean.png')
                            except Exception as exc:
                                print(f"[eval_log] failed to render fusion attention: {exc}")
                        # chown to host user (uid/gid 1000), container runs as root
                        try:
                            for p in sorted(
                                eval_log_dir.rglob('*'),
                                key=lambda path: len(path.parts),
                                reverse=True,
                            ):
                                os.chown(p, 1000, 1000)
                            os.chown(eval_log_dir, 1000, 1000)
                        except Exception:
                            pass
                        print(
                            "[eval_log] saved " + ", ".join(saved_diagnostics)
                            + f" to {eval_log_dir}"
                        )
                    if 'runtime_metrics' in locals():
                        cycle_counts = _runtime_cycle_counts(runtime_metrics)
                        print("[dual-F/T runtime summary]")
                        print(
                            "  cycles=", cycle_counts["cycles"],
                            "completed_cycles=", cycle_counts["completed_cycles"],
                            "valid_observation_cycles=", cycle_counts["valid_observation_cycles"],
                            "dropped_cycles=", cycle_counts["dropped_cycles"],
                            "safety_rejected_cycles=", cycle_counts["safety_rejected_cycles"],
                            "context_recovery_skipped_cycles=",
                            cycle_counts["context_recovery_skipped_cycles"],
                        )
                        print(
                            "  left_ft_age:",
                            _format_timing_stats_ms(runtime_metrics["ft_left_age"]),
                        )
                        print(
                            "  right_ft_age:",
                            _format_timing_stats_ms(runtime_metrics["ft_right_age"]),
                        )
                        print(
                            "  pre_command_ft_age:",
                            _format_timing_stats_ms(runtime_metrics["safety_ft_age"]),
                        )
                        print(
                            "  observation_assembly:",
                            _format_timing_stats_ms(runtime_metrics["obs_assembly"]),
                        )
                        print(
                            "  policy_total_inference:",
                            _format_timing_stats_ms(runtime_metrics["inference"]),
                        )
                        print(
                            "  total_loop:",
                            _format_timing_stats_ms(runtime_metrics["loop"]),
                        )
                        print(
                            "  deadline_misses=", runtime_metrics["deadline_misses"],
                            "robot_command_calls=", runtime_metrics["robot_command_calls"],
                        )

                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
