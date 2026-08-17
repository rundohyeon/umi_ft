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

Axis check without big motion (read terminal only, robot stays put):
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
import os
import pathlib
import select
import sys
import termios
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
from umi.real_world.rg2ft_obs import prepare_rg2ft_policy_obs

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
    transform_map = getattr(obs_encoder, "key_transform_map", None)
    if transform_map is None:
        return []
    disabled = []
    for key in list(transform_map.keys()):
        transform_map[key] = torch.nn.Identity()
        disabled.append(key)
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
    """Remap raw model output (pose10d cols 0:9 per robot block) from the
    SLAM training frame to the robot frame via _SLAM_FRAME_FIX_P.
    Position transforms as v' = P @ v. Rotation is encoded as rot6d (the
    first two ROWS of the 3x3 rotation matrix, see pose_util.rot6d_to_mat/
    mat_to_rot6d) and must transform by conjugation R' = P @ R @ P.T so the
    encoded orientation stays consistent with the remapped position frame."""
    P = _SLAM_FRAME_FIX_P
    for r in range(n_robots):
        b = r * 10
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
    try:
        action = get_real_umi_action(raw_action, obs, action_pose_repr)
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


_POSE10D_LABELS = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5", "grip"]
_PANEL_W = 320
_PANEL_H = 420   # taller than wide: text panels have ~19-22 lines, 320 clipped them


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


def _render_eval_video_frame(original_rgb, current_bgr, original_tcp6, robot_tcp6,
        live_tcp6, *, source_idx: int | None, match_episode_id: int | None):
    """3-panel eval video: original zarr frame | current image | original coordinate."""
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
    return np.concatenate([left, sep, middle, sep, right], axis=1)


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
    "--coord_transform_audit",
    is_flag=True,
    default=False,
    help=(
        "Print dataset<->robot TCP transform roundtrip checks for obs/actions "
        "and the selected zarr match episode."
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
    help="Run policy inference and print debug, but do not move the robot.",
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
    frequency, command_latency, no_spacemouse, no_gripper,
    direct_dynamixel_gripper, dynamixel_gripper_config, gripper_calib_zarr,
    no_mirror, sim_fov, camera_intrinsics, eval_image_mask, policy_image_crop_ratio,
    inpaint_aruco_tags, aruco_config, disable_eval_image_aug,
    mirror_swap, print_policy_output,
    pose_eval_audit, dataset_zarr, dataset_z_stride, print_model_input,
    show_policy_image, policy_input_audit, coord_transform_audit,
    print_motion_debug, vis_pose, max_policy_iters, plan_only,
    tcp_delta_scales, action_scale, freeze_rotation, match_g_move_robot,
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
    cfg = payload['cfg']
    # The checkpoint payload restores all trained weights immediately after
    # workspace construction. Avoid an unnecessary online pretrained-weight
    # download when this self-contained folder is moved to the robot PC.
    if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is not None:
        cfg.policy.obs_encoder.pretrained = False
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
    else:
        ft_obs_horizon = 0
        ft_obs_stride = 1

    if steps_per_inference is None:
        steps_per_inference = int(
            OmegaConf.select(cfg, "execution.n_action_steps", default=6)
        )
    if dual_ft_enabled:
        allowed_steps = list(
            OmegaConf.select(
                cfg,
                "execution.allowed_n_action_steps",
                default=[1, 2, 4, 8],
            )
        )
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
    if plan_only:
        max_policy_iters = 1
        print("plan_only: robot will not move; printing one inference cycle only.")
    if max_policy_iters is not None:
        print("max_policy_iters:", max_policy_iters)
    if tcp_delta_scale_vec is not None:
        print("tcp_delta_scales:", tcp_delta_scale_vec.tolist())
    if action_scale != 1.0:
        print("action_scale:", action_scale)
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
                dynamixel_home_to_open=gc.get('dynamixel_home_to_open', False),
                use_gripper=(not no_gripper),
                robot_type=rc['robot_type'],
                tcp_offset=rc['tcp_offset'],
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                camera_obs_latency=float(cfg.task.get("camera_obs_latency", 0.125)),
                robot_obs_latency=rc['robot_obs_latency'],
                gripper_obs_latency=gc.get('gripper_obs_latency', 0.01),
                robot_action_latency=rc.get('robot_action_latency', 0.1),
                gripper_action_latency=gc.get('gripper_action_latency', 0.1),
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                ft_obs_horizon=ft_obs_horizon,
                ft_obs_stride=ft_obs_stride,
                ft_obs_frequency=float(
                    OmegaConf.select(
                        cfg,
                        "task.ft_frequency",
                        default=gc.get('rg2ft_frequency', 100.0),
                    )
                ),
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                policy_image_crop_ratio=policy_image_crop_ratio,
                mask_before_image_transform=eval_image_mask,
                inpaint_aruco_tags=inpaint_aruco_tags,
                aruco_config_path=aruco_config,
                mirror_swap=mirror_swap,
                max_pos_speed=2.0,
                max_rot_speed=6.0,
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

            device = torch.device('cpu')
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

            print("Warming up policy inference")
            obs = env.get_obs()
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
            with torch.no_grad():
                policy.reset()
                obs_for_model = prepare_rg2ft_policy_obs(
                    obs, cfg.task.shape_meta
                )
                obs_for_model = _apply_slam_frame_fix_to_obs(
                    obs_for_model, n_robots
                )
                episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(episode_start_pose)
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs_for_model, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose_for_model)
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
                assert raw_pred.shape[-1] == 10 * n_robots
                _check_finite_array("[warmup] raw action_pred before frame fix", raw_pred)
                raw_pred = _apply_slam_frame_fix(raw_pred, n_robots)
                _check_finite_array("[warmup] raw action_pred after frame fix", raw_pred)
                action_dataset = _decode_real_umi_action_checked(
                    raw_pred, obs_for_model, action_pose_repr, "[warmup dataset]"
                )
                action = _transform_tcp7_action(
                    action_dataset, _ROBOT_FROM_DATASET_T, n_robots
                )
                _check_finite_array("[warmup] robot-frame tcp7 action", action)
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
                obs_human = env.get_obs()
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
                        obs = env.get_obs()
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
                            match_img = episode_first_frame_map[match_episode_id]
                            ih, iw, _ = match_img.shape
                            oh, ow, _ = vis_img.shape
                            tf = get_image_transform(
                                input_res=(iw, ih),
                                output_res=(ow, oh),
                                bgr_to_rgb=False,
                            )
                            match_bgr = cv2.cvtColor(tf(match_img), cv2.COLOR_RGB2BGR)
                            vis_img = (
                                (vis_img.astype(np.float32) + match_bgr.astype(np.float32))
                                / 2.0
                            ).astype(np.uint8)

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
                                    obs_video = env.get_obs()
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
                            else:
                                pose = saved_start_tcp6.copy()
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
                        if has_motion_cmd or has_grip_cmd:
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
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # per-run CSV + comparison-video logging (policy input / model
                    # output / actually-transmitted action), one folder per episode
                    eval_episode_id = env.replay_buffer.n_episodes
                    eval_log_dir = pathlib.Path(output).joinpath(
                        'eval_logs', f'ep{eval_episode_id}_{time.strftime("%Y%m%d_%H%M%S")}')
                    eval_log_dir.mkdir(parents=True, exist_ok=True)
                    eval_csv_file = open(eval_log_dir.joinpath('log.csv'), 'w', newline='')
                    eval_csv_writer = csv.writer(eval_csv_file)
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
                    obs = env.get_obs()
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
                    episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(episode_start_pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    policy_iter_count = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        if no_gripper:
                            obs = _with_synthetic_gripper_width(
                                obs,
                                no_gripper_obs_width,
                                fallback=max_gripper_width,
                            )
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        if dual_ft_enabled:
                            print(
                                "Dual-F/T causal timing: "
                                f"anchor={float(obs_timestamps[-1]):.9f} "
                                f"left_last={float(obs['robot0_ft_left_timestamps'][-1]):.9f} "
                                f"left_age_ms={float(obs['robot0_ft_left_age']) * 1000.0:.3f} "
                                f"right_last={float(obs['robot0_ft_right_timestamps'][-1]):.9f} "
                                f"right_age_ms={float(obs['robot0_ft_right_age']) * 1000.0:.3f}"
                            )

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_for_model = prepare_rg2ft_policy_obs(
                                obs, cfg.task.shape_meta
                            )
                            obs_for_model = _apply_slam_frame_fix_to_obs(
                                obs_for_model, n_robots
                            )
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs_for_model, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=None,
                                episode_start_pose=episode_start_pose_for_model)
                            _check_policy_inputs_finite(
                                obs_dict_np, f"[policy iter={iter_idx}]"
                            )
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
                            raw_action = result["action_pred"][0].detach().to("cpu").numpy()
                            _check_finite_array(
                                f"[policy iter={iter_idx}] raw action_pred before frame fix",
                                raw_action,
                            )
                            raw_action = _apply_slam_frame_fix(raw_action, n_robots)
                            _check_finite_array(
                                f"[policy iter={iter_idx}] raw action_pred after frame fix",
                                raw_action,
                            )
                            action_dataset = _decode_real_umi_action_checked(
                                raw_action,
                                obs_for_model,
                                action_pose_repr,
                                f"[policy iter={iter_idx} dataset]",
                            )
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
                            print("Inference latency:", time.time() - s)
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
                        this_target_poses = action[:n_exec]
                        assert this_target_poses.shape[1] == 7 * n_robots

                        # deal with timing
                        # Schedule from the next available control tick. Basing these
                        # stamps on obs_timestamps[-1] can skip the first several
                        # near-term actions when obs + inference latency is high.
                        action_exec_latency = 0.01
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
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
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

                        if direct_gripper is not None and len(this_target_poses) > 0:
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

                        # execute actions
                        if plan_only:
                            print(
                                "[plan_only] skipped exec_actions; "
                                "compare delta xyz above with teleop axes."
                            )
                        else:
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=False
                            )
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

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

                        if not eval_csv_header_written:
                            eval_csv_writer.writerow(
                                ['iter_idx', 'wall_time']
                                + [f'obs_pos_{i}' for i in range(len(obs_pos))]
                                + [f'obs_rot_{i}' for i in range(len(obs_rot))]
                                + [f'obs_grip_{i}' for i in range(len(obs_grip))]
                                + [f'raw_action_{i}' for i in range(len(raw_row))]
                                + [f'converted_{i}' for i in range(len(converted_row))]
                                + [f'sent_{i}' for i in range(len(sent_row))]
                                + ['n_submitted']
                                + [f'accum_cm_{i}' for i in range(3)]
                            )
                            eval_csv_header_written = True
                        eval_csv_writer.writerow(
                            [iter_idx, time.time()]
                            + obs_pos.tolist() + obs_rot.tolist() + obs_grip.tolist()
                            + raw_row.tolist() + converted_row.tolist() + sent_row.tolist()
                            + [len(this_target_poses)]
                            + accum_xyz_cm.tolist()
                        )
                        eval_csv_file.flush()

                        original_rgb = None
                        original_tcp6 = None
                        robot_tcp6 = None
                        source_idx = None
                        if match_debug_data is not None:
                            elapsed_s = max(0.0, time.monotonic() - t_start)
                            source_idx = int(round(elapsed_s * float(match_debug_data["fps"])))
                            n_src = len(match_debug_data["raw_pose6"])
                            source_idx = int(np.clip(source_idx, 0, max(0, n_src - 1)))
                            original_tcp6 = match_debug_data["raw_pose6"][source_idx]
                            robot_tcp6 = match_debug_data["robot_pose6"][source_idx]
                            if match_debug_data.get("rgb") is not None:
                                original_rgb = match_debug_data["rgb"][source_idx]
                        current_bgr = _policy_input_bgr_from_obs(obs)
                        if current_bgr is None:
                            current_bgr = _get_live_display_bgr(
                                env, camera_idx=match_camera
                            )
                        eval_frame = _render_eval_video_frame(
                            original_rgb,
                            current_bgr,
                            original_tcp6,
                            robot_tcp6,
                            _tcp6_from_obs(obs),
                            source_idx=source_idx,
                            match_episode_id=(
                                None if match_debug_data is None
                                else match_debug_data["episode"]
                            ),
                        )
                        if eval_video_writer is None:
                            fh, fw = eval_frame.shape[:2]
                            eval_video_writer = cv2.VideoWriter(
                                str(eval_log_dir.joinpath('comparison.mp4')),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                max(1.0, 1.0 / dt), (fw, fh))
                        eval_video_writer.write(eval_frame)

                        # visualize (full-res camera feed; obs rgb is masked 224x224 for policy)
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = _get_live_display_bgr(env, camera_idx=match_camera)
                        header = "Episode: {}, Time: {:.1f}".format(
                            episode_id, time.monotonic() - t_start
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
                            match_policy_rgb = None
                            if selected_match_episode_for_eval is not None:
                                match_policy_rgb = episode_first_policy_frame_map.get(
                                    int(selected_match_episode_for_eval)
                                )
                            _show_policy_input_window(
                                obs,
                                f"policy input | t={time.monotonic() - t_start:.1f}s",
                                match_rgb=match_policy_rgb,
                            )
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
                        if max_policy_iters is not None and policy_iter_count >= max_policy_iters:
                            print(f"max_policy_iters={max_policy_iters} reached.")
                            stop_episode = True
                        if stop_episode:
                            if (not plan_only) and len(action_timestamps) > 0:
                                final_wait = float(action_timestamps[-1]) + dt
                                precise_wait(final_wait, time_func=time.time)
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                finally:
                    if eval_csv_file is not None:
                        eval_csv_file.close()
                    if eval_video_writer is not None:
                        eval_video_writer.release()
                    if eval_log_dir is not None:
                        # chown to host user (uid/gid 1000), container runs as root
                        try:
                            for p in eval_log_dir.glob('*'):
                                os.chown(p, 1000, 1000)
                            os.chown(eval_log_dir, 1000, 1000)
                        except Exception:
                            pass
                        print(f"[eval_log] saved log.csv + comparison.mp4 to {eval_log_dir}")

                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
