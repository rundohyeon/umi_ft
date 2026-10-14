#!/usr/bin/env python3
"""Guarded live Indy/RG2-FT runner for the RGB4 paired-F/T slip policy.

The default is one planning-only cycle: hardware is opened and the exact live
observation is evaluated, but no robot or gripper waypoint is submitted.  The
deployment wrapper maps ``RG2_ENABLE_MOTION=1`` to motion and uses the same
camera-window ``c`` start key as the established Indy evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
import scipy.spatial.transform as st
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "data/contextFT-4"
DEFAULT_ACTION_CHECKPOINT = DEFAULT_MODEL_DIR / "epoch=0008-val_loss=0.02815.ckpt"
DEFAULT_CONTEXT_CHECKPOINT = DEFAULT_MODEL_DIR / "best.pt"
DEFAULT_ROBOT_CONFIG = ROOT / "example/eval_robots_config_indy_rg2.yaml"
DEFAULT_ARUCO_CONFIG = ROOT / "slam_pipeline_latest/calibration/aruco_config.yaml"
PHASE_NAMES = ("approach", "error1", "error2", "turning")
ACTION_HZ = 60000.0 / 1001.0 / 3.0


class LiveObservationUnavailable(RuntimeError):
    """A transient sensor/history condition; never reinterpret it as a class."""


class ContextProbabilityMomentum:
    """EMA over final context probabilities at the classifier's 10 Hz cadence."""

    def __init__(self, previous_weight: float = 0.0):
        previous_weight = float(previous_weight)
        if not np.isfinite(previous_weight) or not 0.0 <= previous_weight < 1.0:
            raise ValueError("context momentum previous weight must be in [0,1)")
        self.previous_weight = previous_weight
        self.previous = None

    def reset(self) -> None:
        self.previous = None

    def apply(self, context: dict) -> dict:
        current = np.asarray(context["probabilities"], dtype=np.float64)
        if (
            current.shape != (4,)
            or not np.isfinite(current).all()
            or (current < 0.0).any()
            or not np.isclose(current.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("context probabilities must be normalized finite [4]")
        model_probabilities = current.astype(np.float32)
        if self.previous is None or self.previous_weight == 0.0:
            used = current.copy()
        else:
            used = (
                self.previous_weight * self.previous
                + (1.0 - self.previous_weight) * current
            )
            used /= used.sum()
        self.previous = used.copy()
        class_id = int(used.argmax())
        result = dict(context)
        result.update(
            model_probabilities=model_probabilities.tolist(),
            probabilities=used.astype(np.float32).tolist(),
            class_id=class_id,
            class_name=PHASE_NAMES[class_id],
        )
        return result


class TerminalKeyPoller:
    """Nonblocking single-key input so teleop also works from a Docker terminal."""

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
            print(f"Terminal keyboard fallback disabled: {exc}", flush=True)
            self.close()
            return False

    def close(self) -> None:
        if self.enabled and self.fd is not None and self.old_attrs is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
            except Exception:
                pass
        self.enabled = False

    def poll(self) -> int:
        if not self.enabled or self.fd is None:
            return -1
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return -1
            value = os.read(self.fd, 1)
        except Exception:
            return -1
        if not value:
            return -1
        if value == b"\x1b":
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    os.read(self.fd, 8)
                    return -1
            except Exception:
                return -1
            return 27
        return value[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n"
    )


def resolve_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def assess_result_age(
    total_age_s: float,
    *,
    camera_latency_s: float,
    max_total_age_s: float,
    max_processing_age_s: float,
) -> dict:
    """Separate expected camera latency from queueing and model runtime."""

    total_age_s = float(total_age_s)
    processing_age_s = total_age_s - float(camera_latency_s)
    stale_reasons = []
    if not np.isfinite(total_age_s) or total_age_s < 0.0:
        stale_reasons.append("invalid_total_age")
    elif total_age_s > max_total_age_s:
        stale_reasons.append("total_age")
    if not np.isfinite(processing_age_s) or processing_age_s < 0.0:
        stale_reasons.append("invalid_processing_age")
    elif processing_age_s > max_processing_age_s:
        stale_reasons.append("processing_age")
    return {
        "total_age_s": total_age_s,
        "processing_age_s": processing_age_s,
        "max_total_age_s": float(max_total_age_s),
        "max_processing_age_s": float(max_processing_age_s),
        "is_fresh": not stale_reasons,
        "stale_reasons": stale_reasons,
    }


def select_context_update_anchors(
    rgb_times,
    last_timestamp,
    *,
    period_s: float,
    tolerance_s: float = 0.024,
    next_target_timestamp=None,
) -> np.ndarray:
    """Select every still-unprocessed RGB anchor on the classifier cadence."""

    times = np.asarray(rgb_times, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("context cadence requires ordered finite RGB timestamps")
    period_s = float(period_s)
    tolerance_s = float(tolerance_s)
    if (
        not np.isfinite(period_s)
        or period_s <= 0
        or not np.isfinite(tolerance_s)
        or tolerance_s <= 0
        or tolerance_s >= period_s / 2
    ):
        raise ValueError("invalid context cadence period/tolerance")
    if last_timestamp is None:
        return times[[-1]].copy()
    cursor = float(last_timestamp)
    if not np.isfinite(cursor):
        raise ValueError("last context timestamp must be finite")
    target = (
        cursor + period_s
        if next_target_timestamp is None
        else float(next_target_timestamp)
    )
    if not np.isfinite(target) or target <= cursor:
        raise ValueError("next context cadence target must follow the last timestamp")
    selected = []
    latest = float(times[-1])
    while True:
        # Do not bind a cadence slot to an early frame merely because it is
        # barely within tolerance. Waiting until the target time is present
        # keeps consecutive selected timestamps inside the runtime's 25 ms
        # cadence allowance instead of alternating early/late and resetting
        # the probability history.
        if latest < target:
            break
        available = np.flatnonzero(times > cursor)
        if len(available) == 0:
            break
        index = int(available[np.argmin(np.abs(times[available] - target))])
        error = abs(float(times[index]) - target)
        if error > tolerance_s:
            raise LiveObservationUnavailable("context_rgb_cadence_mismatch")
        cursor = float(times[index])
        selected.append(cursor)
        target += period_s
    if not selected:
        raise LiveObservationUnavailable("waiting_for_context_cadence")
    return np.asarray(selected, dtype=np.float64)


def select_context_updates_for_runtime(rgb_times, context_runtime) -> tuple[np.ndarray, bool]:
    """Select cadence anchors, resynchronizing after a camera-frame miss."""

    try:
        return select_context_update_anchors(
            rgb_times,
            context_runtime.last_timestamp,
            period_s=context_runtime.model.history_period_s,
            next_target_timestamp=context_runtime.next_update_target_timestamp,
        ), False
    except LiveObservationUnavailable as exc:
        if str(exc) != "context_rgb_cadence_mismatch":
            raise
        # A target whose tolerance window has already passed can never become
        # valid by waiting for newer frames. Keep the task boundary and sensor
        # buffers, discard only probability history, and restart at the newest
        # causal RGB anchor so policy inference can continue in this cycle.
        context_runtime.reset_history()
        return select_context_update_anchors(
            rgb_times,
            None,
            period_s=context_runtime.model.history_period_s,
        ), True


def rgb_uint8(value) -> np.ndarray:
    """Convert one policy RGB image to contiguous HWC uint8 without swapping channels."""

    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"policy RGB must be HWC/CHW with 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("policy RGB contains no finite values")
        if float(finite.max()) <= 1.0 + 1e-6:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def load_match_policy_frame(dataset: Path, episode: int) -> tuple[np.ndarray, dict]:
    """Load only the selected episode's first policy frame from a zarr dataset."""

    try:
        import imagecodecs.numcodecs as imagecodecs_numcodecs

        imagecodecs_numcodecs.register_codecs()
    except Exception:
        pass
    import zarr

    store = None
    dataset = resolve_path(dataset)
    if dataset.is_file() and dataset.name.lower().endswith(".zip"):
        store = zarr.ZipStore(str(dataset), mode="r")
        root = zarr.open_group(store=store, mode="r")
    elif dataset.is_dir():
        root = zarr.open_group(str(dataset), mode="r")
    else:
        raise FileNotFoundError(f"match dataset not found: {dataset}")
    try:
        if "data" not in root or "camera0_rgb" not in root["data"]:
            raise KeyError("match dataset has no data/camera0_rgb")
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        if ends.ndim != 1 or len(ends) == 0:
            raise ValueError("match dataset contains no episodes")
        episode = int(episode)
        if not 0 <= episode < len(ends):
            raise IndexError(
                f"match episode {episode} is outside [0,{len(ends) - 1}]"
            )
        start = 0 if episode == 0 else int(ends[episode - 1])
        images = root["data"]["camera0_rgb"]
        frame = rgb_uint8(np.asarray(images[start]))
        info = {
            "dataset": str(dataset),
            "episode": episode,
            "n_episodes": int(len(ends)),
            "frame_index": start,
            "array_shape": tuple(int(x) for x in images.shape),
            "array_dtype": str(images.dtype),
        }
        return frame, info
    finally:
        if store is not None:
            store.close()


def make_policy_image_panel(
    live_rgb,
    match_rgb=None,
    *,
    label: str,
) -> np.ndarray:
    """Build live/train/50:50 policy-view panels in OpenCV BGR format."""

    import cv2

    live = rgb_uint8(live_rgb)
    live = cv2.resize(live, (336, 336), interpolation=cv2.INTER_NEAREST)
    panels = [("live policy input", live)]
    if match_rgb is not None:
        match = cv2.resize(
            rgb_uint8(match_rgb), (336, 336), interpolation=cv2.INTER_NEAREST
        )
        panels = [
            ("train zarr first frame", match),
            ("live policy input", live),
            ("overlap 50/50", cv2.addWeighted(live, 0.5, match, 0.5, 0.0)),
        ]
    rendered = []
    for title, rgb in panels:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        rendered.append(bgr)
    panel = np.concatenate(rendered, axis=1)
    cv2.putText(
        panel,
        label,
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def show_policy_image(
    live_rgb,
    match_rgb=None,
    *,
    label: str,
    terminal_key_poller: TerminalKeyPoller | None = None,
) -> int:
    import cv2

    cv2.imshow(
        "policy_input",
        make_policy_image_panel(live_rgb, match_rgb, label=label),
    )
    cv_key = cv2.waitKey(1)
    if cv_key >= 0 and (cv_key & 0xFF) != 255:
        return cv_key & 0xFF
    if terminal_key_poller is not None:
        return terminal_key_poller.poll()
    return -1


TELEOP_MOTION_KEYS = frozenset(ord(value) for value in "adsweqjlikuo")
TELEOP_GRIPPER_KEYS = frozenset((ord("z"), ord("x")))


def teleop_delta_for_key(
    key: int,
    *,
    position_step_m: float,
    rotation_step_rad: float,
    gripper_step_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map the established Indy keyboard controls to one bounded teleop step."""

    dpos = np.zeros(3, dtype=np.float64)
    drot_xyz = np.zeros(3, dtype=np.float64)
    grip_delta = 0.0
    if key == ord("a"):
        dpos[0] += position_step_m
    elif key == ord("d"):
        dpos[0] -= position_step_m
    elif key == ord("s"):
        dpos[1] += position_step_m
    elif key == ord("w"):
        dpos[1] -= position_step_m
    elif key == ord("e"):
        dpos[2] += position_step_m
    elif key == ord("q"):
        dpos[2] -= position_step_m
    elif key == ord("j"):
        drot_xyz[0] -= rotation_step_rad
    elif key == ord("l"):
        drot_xyz[0] += rotation_step_rad
    elif key == ord("i"):
        drot_xyz[1] += rotation_step_rad
    elif key == ord("k"):
        drot_xyz[1] -= rotation_step_rad
    elif key == ord("u"):
        drot_xyz[2] -= rotation_step_rad
    elif key == ord("o"):
        drot_xyz[2] += rotation_step_rad
    elif key == ord("z"):
        grip_delta -= gripper_step_m
    elif key == ord("x"):
        grip_delta += gripper_step_m
    return dpos, drot_xyz, grip_delta


def compose_teleop_rotvec(
    previous_rotvec,
    delta_euler_xyz,
    *,
    euler_sequence: str,
    extrinsic: bool,
) -> np.ndarray:
    scipy_sequence = (
        str(euler_sequence).lower() if extrinsic else str(euler_sequence).upper()
    )
    delta_rotation = st.Rotation.from_euler(
        scipy_sequence, np.asarray(delta_euler_xyz), degrees=False
    )
    return (
        delta_rotation
        * st.Rotation.from_rotvec(np.asarray(previous_rotvec, dtype=np.float64))
    ).as_rotvec()


def print_policy_input_audit(stream: dict, obs: dict, prepared: dict, match_info=None) -> None:
    """Print the exact selected live tensors and verify RGB conversion bit-for-bit."""

    indices = np.asarray(prepared["timing"]["rgb_indices"], dtype=np.int64)
    selected = np.asarray(stream["camera0_rgb"])[indices]
    expected = np.ascontiguousarray(selected.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    diff = np.asarray(obs["camera0_rgb"], dtype=np.float32) - expected
    print("[policy_input_audit] exact action-policy observation", flush=True)
    print(
        f"  RGB stream={selected.shape}/{selected.dtype} -> "
        f"model={obs['camera0_rgb'].shape}/{obs['camera0_rgb'].dtype}; "
        f"max_abs_diff={float(np.max(np.abs(diff))):.9g}",
        flush=True,
    )
    print(
        "  RGB timestamps:",
        np.array2string(
            np.asarray(prepared["timing"]["rgb_timestamps_s"]), precision=6
        ),
        flush=True,
    )
    print(
        "  F/T age ms:",
        np.array2string(
            1000.0 * np.asarray(prepared["timing"]["ft_age_s"]), precision=3
        ),
        flush=True,
    )
    for key in ("robot0_ft_left", "robot0_ft_right", "valve_context"):
        value = np.asarray(obs[key])
        print(
            f"  {key}: shape={value.shape} dtype={value.dtype} "
            f"range={float(value.min()):.6g}..{float(value.max()):.6g}",
            flush=True,
        )
    if match_info is not None:
        print(f"  match visualization only: {match_info}", flush=True)


def parse_camera_reorder(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera reorder must be comma-separated integers") from exc
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("camera reorder must contain unique nonnegative indices")
    return result


def assemble_action_observation(
    stream: dict,
    rgb_indices: np.ndarray,
    force_indices: np.ndarray,
    context_probabilities,
) -> dict[str, np.ndarray]:
    """Build the unbatched checkpoint input from already selected causal rows."""

    rgb_indices = np.asarray(rgb_indices, dtype=np.int64)
    force_indices = np.asarray(force_indices, dtype=np.int64)
    frames = np.asarray(stream["camera0_rgb"])
    left = np.asarray(stream["robot0_ft_left"], dtype=np.float32)
    right = np.asarray(stream["robot0_ft_right"], dtype=np.float32)
    probabilities = np.asarray(context_probabilities, dtype=np.float32)
    if rgb_indices.shape != (4,) or force_indices.shape != (4, 41):
        raise ValueError("slip action selection must be RGB [4] and paired F/T [4,41]")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (224, 224, 3):
        raise ValueError("live action RGB must be uint8 [T,224,224,3]")
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 6:
        raise ValueError("live left/right F/T streams must be matching [T,6]")
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
        raise ValueError("context probabilities must be finite [4]")
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError("context probabilities must be nonnegative and sum to one")
    obs = {
        "camera0_rgb": np.ascontiguousarray(
            frames[rgb_indices].transpose(0, 3, 1, 2), dtype=np.float32
        )
        / 255.0,
        "robot0_ft_left": np.ascontiguousarray(left[force_indices], dtype=np.float32),
        "robot0_ft_right": np.ascontiguousarray(right[force_indices], dtype=np.float32),
        "valve_context": np.concatenate([probabilities, np.ones(1, np.float32)])[None],
    }
    expected = {
        "camera0_rgb": (4, 3, 224, 224),
        "robot0_ft_left": (4, 41, 6),
        "robot0_ft_right": (4, 41, 6),
        "valve_context": (1, 5),
    }
    for key, shape in expected.items():
        if obs[key].shape != shape or obs[key].dtype != np.float32:
            raise ValueError(f"invalid live observation {key}: {obs[key].shape}/{obs[key].dtype}")
        if not np.isfinite(obs[key]).all():
            raise ValueError(f"nonfinite live observation: {key}")
    return obs


def prepare_live_observation(
    stream: dict,
    context_runtime,
    *,
    now: float,
    context_momentum: ContextProbabilityMomentum | None = None,
) -> tuple[dict, dict, dict]:
    """Use one shared RGB/F/T selection for context and action inference."""

    from umi.real_world.rgb_force_context import ObservationUnavailable
    from umi.real_world.slip_temporal_context import select_slip_window

    rgb_times = np.asarray(stream["rgb_timestamp_s"], dtype=np.float64)
    ft_times = np.asarray(stream["ft_timestamp_s"], dtype=np.float64)
    frames = np.asarray(stream["camera0_rgb"])
    wrench = np.concatenate(
        [stream["robot0_ft_left"], stream["robot0_ft_right"]], axis=-1
    ).astype(np.float32, copy=False)
    if len(rgb_times) == 0:
        raise LiveObservationUnavailable("empty_rgb_stream")
    try:
        context_anchors, cadence_resynced = select_context_updates_for_runtime(
            rgb_times, context_runtime
        )
        if cadence_resynced and context_momentum is not None:
            context_momentum.reset()
        anchor = float(context_anchors[-1])
        rgb_indices, force_indices = select_slip_window(
            rgb_times,
            ft_times,
            anchor,
            context_runtime.task_start,
            context_runtime.max_force_age_s,
            rgb_horizon=4,
            rgb_span_s=0.5,
        )
        context = None
        for context_anchor in context_anchors:
            context = context_runtime.predict(
                rgb_times,
                frames,
                ft_times,
                wrench,
                anchor_timestamp=float(context_anchor),
                now=now,
            )
            if context_momentum is not None:
                if int(context.get("history_samples", 0)) == 0:
                    context_momentum.reset()
                context = context_momentum.apply(context)
            if context_runtime.next_update_target_timestamp is None:
                context_runtime.next_update_target_timestamp = (
                    float(context_anchor)
                    + context_runtime.model.history_period_s
                )
            else:
                context_runtime.next_update_target_timestamp += (
                    context_runtime.model.history_period_s
                )
        assert context is not None
    except ObservationUnavailable as exc:
        if context_momentum is not None:
            context_momentum.reset()
        raise LiveObservationUnavailable(str(exc)) from exc
    obs = assemble_action_observation(
        stream, rgb_indices, force_indices, context["probabilities"]
    )
    timing = {
        "anchor_timestamp_s": anchor,
        "rgb_indices": rgb_indices,
        "rgb_timestamps_s": rgb_times[rgb_indices],
        "ft_end_timestamps_s": ft_times[force_indices[:, -1]],
        "ft_age_s": rgb_times[rgb_indices] - ft_times[force_indices[:, -1]],
        "oldest_ft_timestamp_s": float(ft_times[force_indices[0, 0]]),
        "context_update_count": int(len(context_anchors)),
        "context_update_timestamps_s": context_anchors,
        "context_cadence_resynced": bool(cadence_resynced),
    }
    source = {
        "robot0_eef_pos": np.asarray(stream["robot0_eef_pos"])[rgb_indices[-1]],
        "robot0_eef_rot_axis_angle": np.asarray(
            stream["robot0_eef_rot_axis_angle"]
        )[rgb_indices[-1]],
        "robot0_gripper_width": np.asarray(stream["robot0_gripper_width"])[
            rgb_indices[-1]
        ],
    }
    return obs, context, {"timing": timing, "source": source}


def future_action_rows(raw_action: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Skip trained horizon row 0 because it represents the current state."""

    action = np.asarray(raw_action, dtype=np.float64)
    if action.shape != (16, 11) or not np.isfinite(action).all():
        raise ValueError(f"policy output must be finite [16,11], got {action.shape}")
    if not 1 <= int(steps) <= 15:
        raise ValueError("steps per inference must be in [1,15]")
    rows = action[1 : 1 + int(steps)]
    return rows[:, :10], rows[:, 10].copy()


def decode_relative_targets(action10: np.ndarray, current_tcp6: np.ndarray) -> np.ndarray:
    """Convert pose9+width from anchor-relative coordinates to robot TCP7."""

    from umi.real_world.real_inference_util import get_real_umi_action

    current = np.asarray(current_tcp6, dtype=np.float64).reshape(6)
    env_obs = {
        "robot0_eef_pos": current[None, :3],
        "robot0_eef_rot_axis_angle": current[None, 3:6],
    }
    result = get_real_umi_action(
        np.asarray(action10, dtype=np.float64), env_obs, action_pose_repr="relative"
    )
    result = np.asarray(result, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7 or not np.isfinite(result).all():
        raise ValueError(f"decoded action must be finite [N,7], got {result.shape}")
    return result


def scale_pose_targets(
    targets: np.ndarray,
    current_tcp6: np.ndarray,
    scale: float,
    *,
    freeze_rotation: bool = False,
) -> np.ndarray:
    """Optionally shrink predicted TCP motion around the latest measured TCP."""

    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("action scale must be in (0,1]")
    out = np.asarray(targets, dtype=np.float64).copy()
    current = np.asarray(current_tcp6, dtype=np.float64).reshape(6)
    current_rotation = st.Rotation.from_rotvec(current[3:6])
    for row in out:
        row[:3] = current[:3] + scale * (row[:3] - current[:3])
        if freeze_rotation:
            row[3:6] = current[3:6]
        elif scale != 1.0:
            target_rotation = st.Rotation.from_rotvec(row[3:6])
            delta = (target_rotation * current_rotation.inv()).as_rotvec() * scale
            row[3:6] = (st.Rotation.from_rotvec(delta) * current_rotation).as_rotvec()
    return out


def apply_policy_motion_momentum(
    targets: np.ndarray,
    *,
    current_pose_width7: np.ndarray,
    previous_weight: float,
) -> np.ndarray:
    """Blend sequential TCP increments and widths from the measured state.

    A previous weight of 1/3 gives the established one-part previous to
    two-parts current-policy ratio. Each policy cycle starts from measured
    TCP/gripper state, so an unexecuted waypoint from an older horizon is never
    treated as physical state.
    """

    weight = float(previous_weight)
    if not 0.0 <= weight < 1.0:
        raise ValueError("motion momentum previous weight must be in [0,1)")
    out = np.asarray(targets, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 7 or not np.isfinite(out).all():
        raise ValueError(f"motion momentum requires finite [N,7], got {out.shape}")
    current = np.asarray(current_pose_width7, dtype=np.float64).reshape(7)
    if not np.isfinite(current).all():
        raise ValueError("motion momentum requires finite measured state")
    if weight == 0.0:
        return out

    reference = current.copy()
    previous_delta = np.zeros(7, dtype=np.float64)
    policy_weight = 1.0 - weight
    measured_width = float(current[6])
    for index, policy_target in enumerate(out):
        policy_delta_position = policy_target[:3] - reference[:3]
        policy_delta_rotation = (
            st.Rotation.from_rotvec(policy_target[3:6])
            * st.Rotation.from_rotvec(reference[3:6]).inv()
        ).as_rotvec()
        previous_width = float(reference[6])
        blended_width = weight * measured_width + policy_weight * policy_target[6]
        blended_delta = np.concatenate(
            [
                weight * previous_delta[:3]
                + policy_weight * policy_delta_position,
                weight * previous_delta[3:6]
                + policy_weight * policy_delta_rotation,
                [blended_width - previous_width],
            ]
        )
        reference[:3] += blended_delta[:3]
        reference[3:6] = (
            st.Rotation.from_rotvec(blended_delta[3:6])
            * st.Rotation.from_rotvec(reference[3:6])
        ).as_rotvec()
        reference[6] = blended_width
        out[index] = reference
        previous_delta = blended_delta
    return out


def select_device(requested: str):
    import torch

    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def load_bound_models(action_checkpoint: Path, context_checkpoint: Path, device, weights: str):
    import torch
    from omegaconf import OmegaConf

    from eval_slip_action import load_policy
    from umi.real_world.slip_temporal_context import SlipTemporalContextRuntime

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg, policy, action_metadata = load_policy(action_checkpoint, device, weights)
    expected = OmegaConf.to_container(
        cfg.task.model_contract.context_training_metadata, resolve=True
    )
    expected_sha = str(expected.get("checkpoint_sha256", "")).lower()
    actual_sha = sha256_file(context_checkpoint)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError(
            "Context checkpoint is not the classifier bound to the action checkpoint: "
            f"expected {expected_sha or '<missing>'}, got {actual_sha}"
        )
    if tuple(expected.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError("action checkpoint has an incompatible context class order")
    runtime = SlipTemporalContextRuntime(context_checkpoint, device=str(device))
    if (
        tuple(runtime.metadata["phase_names"]) != PHASE_NAMES
        or int(runtime.metadata["rgb_horizon"]) != 4
        or not np.isclose(float(runtime.metadata["rgb_span_s"]), 0.5, atol=0.01)
    ):
        raise ValueError("Live action requires the saved RGB4/0.5s slip context model")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    return cfg, policy, runtime, {
        **action_metadata,
        "context_checkpoint_sha256": actual_sha,
        "context_checkpoint_metadata": runtime.metadata,
    }


def warm_up_models(policy, context_runtime, device, runs: int = 1) -> dict:
    """Allocate CUDA kernels before establishing the live task time boundary."""

    import torch

    runs = int(runs)
    if runs < 0:
        raise ValueError("warmup runs must be nonnegative")
    timings = []
    for index in range(runs):
        context_runtime.reset(task_start_timestamp=0.0)
        context_obs = {
            "camera0_rgb": torch.zeros(4, 3, 224, 224, dtype=torch.float32),
            "robot0_ft_left": torch.zeros(4, 41, 6, dtype=torch.float32),
            "robot0_ft_right": torch.zeros(4, 41, 6, dtype=torch.float32),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        context_result = context_runtime.predict_observation(context_obs, timestamp=1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        context_s = time.perf_counter() - started
        context_value = np.asarray(
            [*context_result["probabilities"], 1.0], dtype=np.float32
        ).reshape(1, 1, 5)
        action_obs = {
            "camera0_rgb": torch.zeros(
                1, 4, 3, 224, 224, dtype=torch.float32, device=device
            ),
            "robot0_ft_left": torch.zeros(
                1, 4, 41, 6, dtype=torch.float32, device=device
            ),
            "robot0_ft_right": torch.zeros(
                1, 4, 41, 6, dtype=torch.float32, device=device
            ),
            "valve_context": torch.from_numpy(context_value).to(device),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = policy.predict_action(action_obs)
        if tuple(result["action_pred"].shape) != (1, 16, 11):
            raise ValueError("action warmup returned an invalid trajectory shape")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        action_s = time.perf_counter() - started
        timings.append(
            {"run": index, "context_s": context_s, "action_s": action_s}
        )
    # Warmup must not consume the reproducible live diffusion random stream or
    # leak synthetic context history across the task boundary.
    context_runtime.reset(task_start_timestamp=-np.inf)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    return {
        "runs": runs,
        "timings": timings,
        "action_inference_steps": int(policy.num_inference_steps),
    }


def wait_for_camera_start_key(
    env,
    match_dataset,
    match_episode,
    *,
    auto_start=False,
    motion_enabled: bool,
    max_gripper_width_m: float,
    command_latency_s: float,
    position_step_m: float,
    rotation_step_rad: float,
    gripper_step_m: float,
    teleop_euler_sequence: str,
    teleop_euler_extrinsic: bool,
    motion_safety_config,
    ft_safety_config,
    raw_startup_bias,
):
    """Run keyboard teleop over the overlap view until ``c`` starts policy."""

    import cv2
    from umi.real_world.dual_ft_policy_safety import (
        PolicySafetyError,
        read_and_validate_latest_ft,
        validate_teleop_target,
    )

    episode = int(match_episode)
    match_rgb = None
    match_info = None

    def load_episode(selected):
        if match_dataset is None:
            return None, None
        frame, info = load_match_policy_frame(match_dataset, selected)
        print(
            f"[match] episode {selected}/{info['n_episodes'] - 1}, "
            f"first frame index={info['frame_index']}",
            flush=True,
        )
        return frame, info

    match_rgb, match_info = load_episode(episode)
    if auto_start:
        return episode, match_rgb, match_info

    cv2.namedWindow("policy_input", cv2.WINDOW_AUTOSIZE)
    print(
        "Teleop controls: a/d=x+/x-, s/w=y+/y-, e/q=z+/z-, "
        "j/l=roll-/+, i/k=pitch+/-, u/o=yaw-/+, z/x=gripper close/open.",
        flush=True,
    )
    print(
        "Other controls: c=start policy, n/b=next/previous match episode, Esc=quit.",
        flush=True,
    )
    print(
        "Teleop command mode: "
        + ("robot/gripper motion enabled" if motion_enabled else "plan-only preview"),
        flush=True,
    )
    terminal_keys = TerminalKeyPoller()
    if terminal_keys.start():
        print("Terminal keyboard fallback enabled.", flush=True)

    robot_state = env.get_robot_state()
    target_pose = np.asarray(
        robot_state["ActualTCPPose"], dtype=np.float64
    ).reshape(6)
    gripper_state = env.gripper.get_state()
    target_width = float(gripper_state["gripper_position"])
    all_teleop_keys = TELEOP_MOTION_KEYS | TELEOP_GRIPPER_KEYS
    motion_latch_armed = True
    try:
        while True:
            stream = env.get_valve_context_stream(history_frames=1)
            live_rgb = np.asarray(stream["camera0_rgb"])[-1]
            label = (
                f"teleop | xyz={np.array2string(target_pose[:3], precision=3)} "
                f"grip={target_width:.3f} m | match {episode} | c=start"
            )
            key = show_policy_image(
                live_rgb,
                match_rgb,
                label=label,
                terminal_key_poller=terminal_keys,
            )
            if key == ord("c"):
                print("Policy started by key 'c'.", flush=True)
                return episode, match_rgb, match_info
            if key == 27:
                raise KeyboardInterrupt
            if key in (ord("n"), ord("b")) and match_info is not None:
                delta = 1 if key == ord("n") else -1
                selected = int(
                    np.clip(episode + delta, 0, int(match_info["n_episodes"]) - 1)
                )
                if selected != episode:
                    episode = selected
                    match_rgb, match_info = load_episode(episode)

            if key not in all_teleop_keys:
                motion_latch_armed = True
            if key in all_teleop_keys and motion_latch_armed:
                motion_latch_armed = False
                dpos, drot_xyz, grip_delta = teleop_delta_for_key(
                    key,
                    position_step_m=position_step_m,
                    rotation_step_rad=rotation_step_rad,
                    gripper_step_m=gripper_step_m,
                )
                # Build each keypress from measured state. A previously queued
                # teleop target may not have been reached yet and must not make
                # the next 5 mm keypress look like one large policy waypoint.
                current_pose = np.asarray(
                    env.get_robot_state()["ActualTCPPose"], dtype=np.float64
                ).reshape(6)
                current_width = float(env.gripper.get_state()["gripper_position"])
                proposed_pose = current_pose.copy()
                proposed_pose[:3] += dpos
                proposed_pose[3:6] = compose_teleop_rotvec(
                    proposed_pose[3:6],
                    drot_xyz,
                    euler_sequence=teleop_euler_sequence,
                    extrinsic=teleop_euler_extrinsic,
                )
                proposed_width = float(
                    np.clip(
                        current_width + grip_delta,
                        0.0,
                        float(max_gripper_width_m),
                    )
                )
                target_tcp7 = np.concatenate(
                    [proposed_pose, np.asarray([proposed_width], dtype=np.float64)]
                )
                try:
                    validate_teleop_target(
                        target_tcp7, current_pose, motion_safety_config
                    )
                    if motion_enabled:
                        read_and_validate_latest_ft(
                            env, raw_startup_bias, ft_safety_config
                        )
                except PolicySafetyError as exc:
                    print(
                        f"[teleop] command refused by safety guard; "
                        f"Teleop remains active: {exc}",
                        flush=True,
                    )
                else:
                    if motion_enabled:
                        env.exec_actions(
                            target_tcp7[None],
                            np.asarray(
                                [time.time() + command_latency_s], dtype=np.float64
                            ),
                            compensate_latency=False,
                        )
                    else:
                        print("[plan_only] skipped teleop command", flush=True)
                    target_pose = proposed_pose
                    target_width = proposed_width
                    print(
                        "[teleop] key="
                        f"{chr(key)!r} target_tcp6="
                        f"{np.array2string(target_pose, precision=5)} "
                        f"width={target_width:.5f}m",
                        flush=True,
                    )
            time.sleep(0.01)
    finally:
        terminal_keys.close()


def validate_robot_config(config: dict) -> tuple[dict, dict]:
    robots = config.get("robots", [])
    grippers = config.get("grippers", [])
    if len(robots) != 1 or len(grippers) != 1:
        raise ValueError("slip live evaluation requires exactly one robot and one RG2-FT")
    robot, gripper = robots[0], grippers[0]
    if "indy" not in str(robot.get("robot_type", "")).lower():
        raise ValueError("this runner supports the guarded Indy path only")
    if str(gripper.get("gripper_type", "rg2ft")).lower() != "rg2ft":
        raise ValueError("this runner requires gripper_type: rg2ft")
    transform = np.asarray(
        robot.get("indy_robot_from_dataset_transform", np.eye(4)), dtype=np.float64
    )
    if transform.shape != (4, 4) or not np.allclose(transform, np.eye(4), atol=1e-9):
        raise ValueError("slip deployment requires identity dataset-to-robot transform")
    if bool(robot.get("indy_policy_tcp7_rot_euler_roundtrip", False)):
        raise ValueError("slip deployment requires indy_policy_tcp7_rot_euler_roundtrip=false")
    if int(gripper.get("rg2ft_frequency", 100)) != 100:
        raise ValueError("slip checkpoints require native 100 Hz RG2-FT")
    return robot, gripper


def make_output_dir(requested: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if requested is None:
        output = ROOT / "outputs/slip_action_real" / timestamp
    else:
        base = resolve_path(requested)
        base.mkdir(parents=True, exist_ok=True)
        output = base / timestamp
    output = resolve_path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_live(args) -> None:
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    from umi.real_world.dual_ft_policy_safety import (
        FTSafetyConfig,
        PolicySafetyError,
        PolicyMotionSafetyConfig,
        read_and_validate_latest_ft,
        validate_workspace_waypoints,
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
    from umi.real_world.umi_env import UmiEnv

    with args.robot_config.open() as stream:
        config = yaml.safe_load(stream)
    robot, gripper = validate_robot_config(config)
    startup_cfg = FTStartupBiasConfig.from_mapping(config.get("ft_startup_bias", {}))
    motion_cfg = PolicyMotionSafetyConfig.from_mapping(config.get("policy_safety", {}))
    ft_cfg = FTSafetyConfig.from_mapping(config.get("ft_safety", {}))
    if args.latest_ft_max_age_sec is not None:
        ft_cfg = replace(
            ft_cfg, max_latest_sample_age_s=args.latest_ft_max_age_sec
        )
    max_width = float(gripper.get("max_gripper_width", 0.1))
    feedback_cfg = GraspForceWidthFeedbackConfig(width_max_m=max_width)
    force_feedback = GraspForceWidthFeedbackController(feedback_cfg)
    output = make_output_dir(args.output)
    print(f"Output: {output}", flush=True)
    print(f"Requested inference device: {args.device}", flush=True)
    print(
        "Mode: "
        + (
            "MOTION ENABLED"
            if args.execute
            else "PLAN ONLY (no robot or gripper waypoint submission)"
        ),
        flush=True,
    )
    print(
        f"F/T: startup software tare={'on' if args.zero_ft_on_start else 'off'} "
        f"({args.ft_zero_samples} samples), latest-command age limit="
        f"{ft_cfg.max_latest_sample_age_s * 1000:.1f} ms. "
        "Keep both fingers unloaded and still.",
        flush=True,
    )
    if args.motion_momentum_previous_weight > 0:
        print(
            "Motion momentum: previous="
            f"{args.motion_momentum_previous_weight:.6f}, policy="
            f"{1.0 - args.motion_momentum_previous_weight:.6f} "
            "(measured-state anchor each cycle)",
            flush=True,
        )
    if args.context_momentum_previous_weight > 0:
        print(
            "Context momentum: previous="
            f"{args.context_momentum_previous_weight:.6f}, current="
            f"{1.0 - args.context_momentum_previous_weight:.6f} "
            "(reset whenever causal context history resets)",
            flush=True,
        )

    run_record = {
        "schema": "slip_action_live_indy_rg2_v1",
        "mode": "execute" if args.execute else "plan_only",
        "action_checkpoint": str(args.action_checkpoint),
        "action_checkpoint_sha256": sha256_file(args.action_checkpoint),
        "context_checkpoint": str(args.context_checkpoint),
        "context_checkpoint_sha256": sha256_file(args.context_checkpoint),
        "robot_config": str(args.robot_config),
        "requested_device": args.device,
        "phase_names": list(PHASE_NAMES),
        "steps_per_inference": args.steps_per_inference,
        "policy_action_hz": ACTION_HZ,
        "requested_replan_hz": args.rate,
        "camera_latency_s": args.camera_latency,
        "max_result_age_s": args.max_result_age,
        "max_processing_age_s": args.max_processing_age,
        "action_scale": args.action_scale,
        "motion_momentum_previous_weight": args.motion_momentum_previous_weight,
        "context_momentum_previous_weight": args.context_momentum_previous_weight,
        "freeze_rotation": args.freeze_rotation,
        "teleop_position_step_m": args.teleop_position_step,
        "teleop_rotation_step_rad": args.teleop_rotation_step,
        "teleop_gripper_step_m": args.teleop_gripper_step,
        "zero_ft_on_start": args.zero_ft_on_start,
        "ft_zero_samples": args.ft_zero_samples,
        "legacy_requested_causal_ft_max_age_s": args.ft_max_age_sec,
        "ft_latest_sample_max_age_s": ft_cfg.max_latest_sample_age_s,
        "match_dataset": None if args.match_dataset is None else str(args.match_dataset),
        "match_episode_requested": args.match_episode,
        "motion_safety": asdict(motion_cfg),
        "motion_safety_mode": "workspace_ft_freshness_only",
        "ft_safety": asdict(ft_cfg),
        "force_feedback": asdict(feedback_cfg),
        "completed_cycles": 0,
        "exit_status": "starting",
    }
    write_json(output / "run.json", run_record)

    executed_any = False
    run_started = time.monotonic()
    live_started = None
    last_submitted_timestamp = None
    last_anchor = None
    last_wait_message = -np.inf
    stale_results = 0
    valid_cycles = 0
    audit_printed = False
    match_rgb = None
    match_info = None
    exit_status = "completed"
    with SharedMemoryManager() as shm_manager:
        with UmiEnv(
            output_dir=output,
            robot_ip=robot["robot_ip"],
            gripper_ip=gripper["gripper_ip"],
            gripper_port=gripper.get("gripper_port", 502),
            gripper_slave_id=gripper.get("gripper_slave_id", 65),
            gripper_type="rg2ft",
            rg2ft_frequency=100,
            rg2ft_force=gripper.get("rg2ft_force", 20.0),
            rg2ft_home_to_open=False,
            rg2ft_move_max_speed=gripper.get("rg2ft_move_max_speed", 0.2),
            rg2ft_open_tolerance=gripper.get("rg2ft_open_tolerance", 0.005),
            rg2ft_zero_on_start=args.zero_ft_on_start,
            rg2ft_zero_samples=args.ft_zero_samples,
            gripper_commands_enabled=args.execute,
            use_gripper=True,
            frequency=args.rate,
            robot_type=robot["robot_type"],
            tcp_offset=robot["tcp_offset"],
            obs_image_resolution=(224, 224),
            max_obs_buffer_size=120,
            context_camera_history_frames=120,
            obs_float32=False,
            camera_reorder=args.camera_reorder,
            init_joints=False,
            enable_multi_cam_vis=False,
            camera_obs_latency=args.camera_latency,
            robot_obs_latency=robot.get("robot_obs_latency", 0.0001),
            gripper_obs_latency=gripper.get("gripper_obs_latency", 0.01),
            robot_action_latency=robot.get("robot_action_latency", 0.1),
            gripper_action_latency=gripper.get("gripper_action_latency", 0.1),
            camera_obs_horizon=1,
            robot_obs_horizon=1,
            gripper_obs_horizon=1,
            ft_obs_horizon=0,
            no_mirror=False,
            fisheye_converter=None,
            policy_image_crop_ratio=1.0,
            mask_before_image_transform=True,
            inpaint_aruco_tags=True,
            aruco_config_path=args.aruco_config,
            mirror_swap=False,
            max_pos_speed=float(robot.get("indy_max_pos_speed_m_s", 0.3)),
            max_rot_speed=float(robot.get("indy_max_rot_speed_rad_s", 1.0)),
            indy_command_timeout_s=float(robot.get("indy_command_timeout_s", 0.3)),
            indy_task_rot_is_euler=robot.get("indy_task_rot_is_euler", True),
            indy_task_rot_euler_seq=robot.get("indy_task_rot_euler_seq", "xyz"),
            indy_task_rot_euler_in_degrees=robot.get(
                "indy_task_rot_euler_in_degrees", True
            ),
            indy_task_rot_euler_extrinsic=robot.get(
                "indy_task_rot_euler_extrinsic", True
            ),
            indy_task_frame_xyz_signs=tuple(
                robot.get("indy_task_frame_xyz_signs", [1, 1, 1])
            ),
            indy_tool_rot_offset_deg=tuple(
                robot.get("indy_tool_rot_offset_deg", [0, 0, 0])
            ),
            shm_manager=shm_manager,
        ) as env:
            try:
                print(
                    "Keep both fingers unloaded and still for startup F/T calibration.",
                    flush=True,
                )
                startup = acquire_startup_bias(env.gripper, startup_cfg)
                raw_startup_bias = np.asarray(startup["bias_12d"], dtype=np.float64)
                residual_bias = startup_residual_after_software_tare(
                    raw_startup_bias, env.rg2ft_ft_offset
                )
                env.set_ft_startup_bias(residual_bias)
                run_record["startup_bias"] = startup
                # Select/initialize CUDA only after all camera, robot, and RG2-FT
                # worker processes have started. Forking after CUDA initialization
                # is not supported by the deployment stack.
                device = select_device(args.device)
                run_record["device"] = str(device)
                print(f"Loading models on {device}...", flush=True)
                cfg, policy, context_runtime, model_metadata = load_bound_models(
                    args.action_checkpoint, args.context_checkpoint, device, args.weights
                )
                run_record["model"] = model_metadata
                run_record["action_contract"] = str(cfg.task.model_contract.version)
                run_record["causal_ft_max_age_s"] = context_runtime.max_force_age_s
                print(
                    "Causal training-window F/T age limit: "
                    f"{context_runtime.max_force_age_s * 1000:.1f} ms; "
                    "command-time latest F/T limit: "
                    f"{ft_cfg.max_latest_sample_age_s * 1000:.1f} ms.",
                    flush=True,
                )
                if not np.isclose(
                    args.ft_max_age_sec, context_runtime.max_force_age_s
                ):
                    print(
                        "Legacy --ft_max_age_sec request: "
                        f"{args.ft_max_age_sec * 1000:.1f} ms. The RGB/F/T "
                        "checkpoint contract remains fixed at "
                        f"{context_runtime.max_force_age_s * 1000:.1f} ms.",
                        flush=True,
                    )
                print(
                    f"Warming up context and action models ({args.warmup_runs} run(s), "
                    f"DDIM={policy.num_inference_steps})...",
                    flush=True,
                )
                warmup = warm_up_models(
                    policy, context_runtime, device, runs=args.warmup_runs
                )
                run_record["warmup"] = warmup
                if warmup["timings"]:
                    last_warmup = warmup["timings"][-1]
                    print(
                        "Warmup latency: "
                        f"context={last_warmup['context_s'] * 1000:.1f}ms, "
                        f"action={last_warmup['action_s'] * 1000:.1f}ms",
                        flush=True,
                    )
                selected_episode, match_rgb, match_info = wait_for_camera_start_key(
                    env,
                    args.match_dataset,
                    args.match_episode,
                    auto_start=not args.wait_for_start_key,
                    motion_enabled=args.execute,
                    max_gripper_width_m=max_width,
                    command_latency_s=args.command_latency,
                    position_step_m=args.teleop_position_step,
                    rotation_step_rad=args.teleop_rotation_step,
                    gripper_step_m=args.teleop_gripper_step,
                    teleop_euler_sequence=robot.get(
                        "indy_teleop_rot_euler_seq", "xyz"
                    ),
                    teleop_euler_extrinsic=bool(
                        robot.get("indy_teleop_rot_euler_extrinsic", False)
                    ),
                    motion_safety_config=motion_cfg,
                    ft_safety_config=ft_cfg,
                    raw_startup_bias=raw_startup_bias,
                )
                run_record["match_episode"] = selected_episode
                run_record["match_frame"] = match_info
                if args.show_policy_image and not args.wait_for_start_key:
                    import cv2

                    cv2.namedWindow("policy_input", cv2.WINDOW_AUTOSIZE)

                task_start = time.time()
                context_runtime.reset(task_start_timestamp=task_start)
                context_momentum = ContextProbabilityMomentum(
                    args.context_momentum_previous_weight
                )
                live_started = time.monotonic()
                run_record["task_start_timestamp_s"] = task_start
                run_record["exit_status"] = "running"
                write_json(output / "run.json", run_record)
                print(
                    "Collecting a new ~0.9 s causal RGB/F/T window...",
                    flush=True,
                )
                deadline = time.monotonic()
                with (output / "cycles.jsonl").open("w", buffering=1) as log:
                    while valid_cycles < args.max_cycles:
                        if time.monotonic() - live_started > args.max_seconds:
                            raise TimeoutError(
                                f"run exceeded --max-seconds={args.max_seconds}"
                            )
                        stream = env.get_valve_context_stream(history_frames=120)
                        anchor = float(stream["rgb_timestamp_s"][-1])
                        if last_anchor is not None and anchor <= last_anchor:
                            time.sleep(0.002)
                            continue
                        if last_anchor is not None and anchor - last_anchor < 0.075:
                            time.sleep(0.002)
                            continue
                        last_anchor = anchor
                        sensor_age_before_s = time.time() - anchor
                        prepare_started = time.perf_counter()
                        try:
                            obs_np, context, prepared = prepare_live_observation(
                                stream,
                                context_runtime,
                                now=time.time(),
                                context_momentum=context_momentum,
                            )
                        except LiveObservationUnavailable as exc:
                            if executed_any:
                                env.hold_robot()
                            if time.monotonic() - last_wait_message >= 1.0:
                                print(f"Waiting for valid causal history: {exc}", flush=True)
                                last_wait_message = time.monotonic()
                            if args.show_policy_image or args.wait_for_start_key:
                                key = show_policy_image(
                                    np.asarray(stream["camera0_rgb"])[-1],
                                    match_rgb,
                                    label=f"warming up causal history | {exc}",
                                )
                                if key in (27, ord("q"), ord("s")):
                                    raise KeyboardInterrupt
                            deadline = max(deadline + 1.0 / args.rate, time.monotonic())
                            time.sleep(max(0.0, deadline - time.monotonic()))
                            continue
                        context_prepare_s = time.perf_counter() - prepare_started

                        inference_start = time.perf_counter()
                        obs_torch = dict_apply(
                            obs_np,
                            lambda value: torch.from_numpy(value)
                            .unsqueeze(0)
                            .to(device),
                        )
                        with torch.inference_mode():
                            result = policy.predict_action(obs_torch)
                        raw_action = result["action_pred"][0].detach().cpu().numpy()
                        inference_s = time.perf_counter() - inference_start
                        result_age_s = time.time() - prepared["timing"][
                            "anchor_timestamp_s"
                        ]
                        result_age = assess_result_age(
                            result_age_s,
                            camera_latency_s=args.camera_latency,
                            max_total_age_s=args.max_result_age,
                            max_processing_age_s=args.max_processing_age,
                        )
                        if not result_age["is_fresh"]:
                            stale_results += 1
                            if executed_any:
                                env.hold_robot()
                            stale_record = {
                                "status": "stale_result_discarded",
                                "recorded_at": time.time(),
                                "context": context,
                                **prepared["timing"],
                                "sensor_age_before_s": sensor_age_before_s,
                                "context_prepare_s": context_prepare_s,
                                "action_inference_s": inference_s,
                                "result_age_s": result_age_s,
                                "processing_age_s": result_age["processing_age_s"],
                                "max_result_age_s": args.max_result_age,
                                "max_processing_age_s": args.max_processing_age,
                                "stale_reasons": result_age["stale_reasons"],
                                "action_inference_steps": int(
                                    policy.num_inference_steps
                                ),
                            }
                            log.write(
                                json.dumps(
                                    stale_record,
                                    allow_nan=False,
                                    default=json_default,
                                )
                                + "\n"
                            )
                            print(
                                "Discarded stale prediction: "
                                f"total={result_age_s:.3f}/{args.max_result_age:.3f}s, "
                                "after_expected_camera="
                                f"{result_age['processing_age_s']:.3f}/"
                                f"{args.max_processing_age:.3f}s "
                                f"({','.join(result_age['stale_reasons'])}; "
                                f"sensor_before={sensor_age_before_s:.3f}s, "
                                f"context={context_prepare_s:.3f}s, "
                                f"action={inference_s:.3f}s). Waiting for a new frame.",
                                flush=True,
                            )
                            deadline = max(
                                deadline + 1.0 / args.rate, time.monotonic()
                            )
                            time.sleep(max(0.0, deadline - time.monotonic()))
                            continue

                        robot_state = env.get_robot_state()
                        current_tcp6 = np.asarray(
                            robot_state["ActualTCPPose"], dtype=np.float64
                        ).reshape(6)
                        gripper_state = env.gripper.get_state()
                        current_width = float(gripper_state["gripper_position"])
                        action10, force_reference = future_action_rows(
                            raw_action, args.steps_per_inference
                        )
                        targets = decode_relative_targets(action10, current_tcp6)
                        targets = scale_pose_targets(
                            targets,
                            current_tcp6,
                            args.action_scale,
                            freeze_rotation=args.freeze_rotation,
                        )
                        _, measured_force, _ = read_and_validate_latest_ft(
                            env, raw_startup_bias, ft_cfg
                        )
                        feedback = force_feedback.correct(
                            targets[:, 6], force_reference, measured_force
                        )
                        targets[:, 6] = feedback["corrected_width_m"]
                        targets = apply_policy_motion_momentum(
                            targets,
                            current_pose_width7=np.concatenate(
                                [current_tcp6, [current_width]]
                            ),
                            previous_weight=args.motion_momentum_previous_weight,
                        )
                        deltas_mm = 1000.0 * (targets[:, :3] - current_tcp6[:3])
                        try:
                            validate_workspace_waypoints(
                                targets, current_tcp6, motion_cfg
                            )
                        except PolicySafetyError as exc:
                            rejection_result_age_s = time.time() - prepared["timing"][
                                "anchor_timestamp_s"
                            ]
                            rejection_result_age = assess_result_age(
                                rejection_result_age_s,
                                camera_latency_s=args.camera_latency,
                                max_total_age_s=args.max_result_age,
                                max_processing_age_s=args.max_processing_age,
                            )
                            rejected_input = output / (
                                f"rejected_input_{valid_cycles:06d}.npz"
                            )
                            np.savez_compressed(
                                rejected_input,
                                **obs_np,
                                rgb_timestamps_s=prepared["timing"][
                                    "rgb_timestamps_s"
                                ],
                                ft_end_timestamps_s=prepared["timing"][
                                    "ft_end_timestamps_s"
                                ],
                            )
                            rejected_record = {
                                "status": "waypoint_safety_rejected",
                                "cycle": valid_cycles,
                                "candidate_cycle": valid_cycles,
                                "recorded_at": time.time(),
                                "executed": False,
                                "error": str(exc),
                                "context": context,
                                **prepared["timing"],
                                "sensor_age_before_s": sensor_age_before_s,
                                "context_prepare_s": context_prepare_s,
                                "action_inference_s": inference_s,
                                "result_age_s": result_age_s,
                                "processing_age_s": result_age["processing_age_s"],
                                "command_result_age_s": rejection_result_age_s,
                                "command_processing_age_s": rejection_result_age[
                                    "processing_age_s"
                                ],
                                "current_tcp6": current_tcp6,
                                "current_width_m": current_width,
                                "raw_action": raw_action,
                                "source_horizon_rows": np.arange(
                                    1, 1 + len(targets), dtype=np.int64
                                ),
                                "targets_tcp7": targets,
                                "delta_xyz_mm": deltas_mm,
                                "force_feedback": feedback,
                                "anchor_source": prepared["source"],
                                "input_file": rejected_input.name,
                            }
                            rejected_json = output / (
                                f"rejected_candidate_{valid_cycles:06d}.json"
                            )
                            write_json(rejected_json, rejected_record)
                            run_record["last_rejected_candidate"] = rejected_json.name
                            print(
                                f"Saved rejected candidate: {rejected_json}",
                                flush=True,
                            )
                            raise
                        command_result_age_s = time.time() - prepared["timing"][
                            "anchor_timestamp_s"
                        ]
                        command_result_age = assess_result_age(
                            command_result_age_s,
                            camera_latency_s=args.camera_latency,
                            max_total_age_s=args.max_result_age,
                            max_processing_age_s=args.max_processing_age,
                        )
                        if args.execute:
                            if not command_result_age["is_fresh"]:
                                stale_results += 1
                                if executed_any:
                                    env.hold_robot()
                                log.write(
                                    json.dumps(
                                        {
                                            "status": "stale_before_command_discarded",
                                            "recorded_at": time.time(),
                                            "context": context,
                                            **prepared["timing"],
                                            "result_age_s": result_age_s,
                                            "command_result_age_s": command_result_age_s,
                                            "command_processing_age_s": command_result_age[
                                                "processing_age_s"
                                            ],
                                            "max_result_age_s": args.max_result_age,
                                            "max_processing_age_s": args.max_processing_age,
                                            "stale_reasons": command_result_age[
                                                "stale_reasons"
                                            ],
                                        },
                                        allow_nan=False,
                                        default=json_default,
                                    )
                                    + "\n"
                                )
                                print(
                                    "Discarded prediction before command: "
                                    f"total={command_result_age_s:.3f}/"
                                    f"{args.max_result_age:.3f}s, "
                                    "after_expected_camera="
                                    f"{command_result_age['processing_age_s']:.3f}/"
                                    f"{args.max_processing_age:.3f}s "
                                    f"({','.join(command_result_age['stale_reasons'])}).",
                                    flush=True,
                                )
                                continue
                            # Keep the latest F/T check and submission adjacent.
                            read_and_validate_latest_ft(env, raw_startup_bias, ft_cfg)
                            command_start = time.time() + args.command_latency
                            timestamps = (
                                command_start
                                + np.arange(len(targets), dtype=np.float64) / ACTION_HZ
                            )
                            env.exec_actions(targets, timestamps, compensate_latency=False)
                            executed_any = True
                            last_submitted_timestamp = float(timestamps[-1])
                        else:
                            command_start = time.time() + args.command_latency
                            timestamps = (
                                command_start
                                + np.arange(len(targets), dtype=np.float64) / ACTION_HZ
                            )
                            print("[plan_only] waypoint submission skipped", flush=True)

                        model_probabilities = np.asarray(
                            context.get(
                                "model_probabilities", context["probabilities"]
                            ),
                            dtype=np.float64,
                        )
                        model_class = PHASE_NAMES[int(model_probabilities.argmax())]
                        print(
                            f"cycle={valid_cycles} context_used={context['class_name']} "
                            f"p={max(context['probabilities']):.3f} "
                            f"context_model={model_class} "
                            f"p={model_probabilities.max():.3f} "
                            f"context_history={context['history_samples']}/"
                            f"{context_runtime.model.history_steps} "
                            "context_updates="
                            f"{prepared['timing']['context_update_count']} "
                            f"age={command_result_age_s * 1000:.0f}ms "
                            f"delta_xyz_mm={np.array2string(deltas_mm[0], precision=2)}",
                            flush=True,
                        )

                        if args.policy_input_audit and not audit_printed:
                            print_policy_input_audit(
                                stream, obs_np, prepared, match_info=match_info
                            )
                            audit_printed = True
                        if args.print_policy_output:
                            print(
                                "[policy_output] raw action_pred [16,11]:\n"
                                + np.array2string(
                                    raw_action,
                                    precision=6,
                                    suppress_small=False,
                                    threshold=np.inf,
                                ),
                                flush=True,
                            )
                            print(
                                "[policy_output] submitted/preview TCP7:\n"
                                + np.array2string(
                                    targets,
                                    precision=6,
                                    suppress_small=False,
                                    threshold=np.inf,
                                ),
                                flush=True,
                            )
                            print(
                                "[policy_output] force reference: "
                                + np.array2string(force_reference, precision=6),
                                flush=True,
                            )

                        record = {
                            "cycle": valid_cycles,
                            "recorded_at": time.time(),
                            "executed": bool(args.execute),
                            "context": context,
                            **prepared["timing"],
                            "result_age_s": result_age_s,
                            "processing_age_s": result_age["processing_age_s"],
                            "command_result_age_s": command_result_age_s,
                            "command_processing_age_s": command_result_age[
                                "processing_age_s"
                            ],
                            "sensor_age_before_s": sensor_age_before_s,
                            "context_prepare_s": context_prepare_s,
                            "inference_s": inference_s,
                            "current_tcp6": current_tcp6,
                            "current_width_m": current_width,
                            "raw_action": raw_action,
                            "source_horizon_rows": np.arange(
                                1, 1 + len(targets), dtype=np.int64
                            ),
                            "targets_tcp7": targets,
                            "target_timestamps_s": timestamps,
                            "delta_xyz_mm": deltas_mm,
                            "force_feedback": feedback,
                            "anchor_source": prepared["source"],
                        }
                        if args.save_inputs:
                            input_path = output / f"input_{valid_cycles:06d}.npz"
                            np.savez_compressed(
                                input_path,
                                **obs_np,
                                rgb_timestamps_s=prepared["timing"][
                                    "rgb_timestamps_s"
                                ],
                                ft_end_timestamps_s=prepared["timing"][
                                    "ft_end_timestamps_s"
                                ],
                            )
                            record["input_file"] = input_path.name
                        log.write(
                            json.dumps(
                                record,
                                allow_nan=False,
                                default=json_default,
                            )
                            + "\n"
                        )
                        cycle_index = valid_cycles
                        valid_cycles += 1
                        if args.show_policy_image:
                            key = show_policy_image(
                                np.asarray(stream["camera0_rgb"])[
                                    int(prepared["timing"]["rgb_indices"][-1])
                                ],
                                match_rgb,
                                label=(
                                    f"cycle {cycle_index} | {context['class_name']} "
                                    f"p={max(context['probabilities']):.3f}"
                                ),
                            )
                            if key in (27, ord("q"), ord("s")):
                                raise KeyboardInterrupt
                        deadline = max(deadline + 1.0 / args.rate, time.monotonic())
                        time.sleep(max(0.0, deadline - time.monotonic()))
            except KeyboardInterrupt:
                exit_status = "interrupted"
            except Exception as exc:
                exit_status = f"failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                if args.execute:
                    if (
                        exit_status == "completed"
                        and last_submitted_timestamp is not None
                    ):
                        time.sleep(max(0.0, last_submitted_timestamp + 0.02 - time.time()))
                    env.hold_robot()
                run_record["completed_cycles"] = valid_cycles
                run_record["stale_results_discarded"] = stale_results
                run_record["exit_status"] = exit_status
                run_record["elapsed_s"] = time.monotonic() - run_started
                write_json(output / "run.json", run_record)
                if args.show_policy_image or args.wait_for_start_key:
                    try:
                        import cv2

                        cv2.destroyAllWindows()
                    except Exception:
                        pass
    print(f"Finished {valid_cycles} valid cycle(s): {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-checkpoint",
        "--action_checkpoint",
        type=Path,
        default=DEFAULT_ACTION_CHECKPOINT,
    )
    parser.add_argument(
        "--context-checkpoint",
        "--context_checkpoint",
        type=Path,
        default=DEFAULT_CONTEXT_CHECKPOINT,
    )
    parser.add_argument(
        "--robot-config", "--robot_config", type=Path, default=DEFAULT_ROBOT_CONFIG
    )
    parser.add_argument(
        "--aruco-config", "--aruco_config", type=Path, default=DEFAULT_ARUCO_CONFIG
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--match-dataset",
        "--match_dataset",
        type=Path,
        help="Training zarr used only for first-frame alignment visualization",
    )
    parser.add_argument("--match-episode", "--match_episode", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("ema_model", "model"), default="ema_model")
    parser.add_argument(
        "--camera-reorder",
        "--camera_reorder",
        type=parse_camera_reorder,
        default=[0],
    )
    parser.add_argument("--camera-latency", "--camera_latency", type=float, default=0.125)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument(
        "--steps-per-inference", "--steps_per_inference", type=int, default=2
    )
    parser.add_argument(
        "--command-latency", "--command_latency", type=float, default=0.05
    )
    parser.add_argument(
        "--max-result-age",
        "--max_result_age",
        type=float,
        default=0.40,
        help="Maximum total age from calibrated camera capture to command",
    )
    parser.add_argument(
        "--max-processing-age",
        "--max_processing_age",
        type=float,
        default=0.25,
        help=(
            "Maximum age after subtracting the configured expected camera latency; "
            "covers frame queueing, context preparation and action inference"
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        "--warmup_runs",
        type=int,
        default=1,
        help="Synthetic context+action forwards before starting the live time boundary",
    )
    parser.add_argument(
        "--max-cycles", "--max_cycles", "--max_policy_iters", type=int, default=1
    )
    parser.add_argument(
        "--max-seconds", "--max_seconds", "--max_duration", type=float, default=60.0
    )
    parser.add_argument("--action-scale", "--action_scale", type=float, default=1.0)
    parser.add_argument(
        "--motion-momentum-previous-weight",
        "--motion_momentum_previous_weight",
        type=float,
        default=0.0,
        help=(
            "Blend waypoint increments as previous*w + policy*(1-w); "
            "use 0.333333 for the established 1:2 ratio"
        ),
    )
    parser.add_argument(
        "--context-momentum-previous-weight",
        "--context_momentum_previous_weight",
        type=float,
        default=0.0,
        help=(
            "EMA over final context probabilities at the 10 Hz context cadence; "
            "0 disables it and 0.5 gives equal previous/current weight"
        ),
    )
    parser.add_argument(
        "--teleop-position-step",
        "--teleop_position_step",
        type=float,
        default=0.005,
        help="Keyboard teleop translation per keypress in metres",
    )
    parser.add_argument(
        "--teleop-rotation-step",
        "--teleop_rotation_step",
        type=float,
        default=0.05,
        help="Keyboard teleop Euler rotation per keypress in radians",
    )
    parser.add_argument(
        "--teleop-gripper-step",
        "--teleop_gripper_step",
        type=float,
        default=0.01,
        help="Keyboard teleop gripper-width change per keypress in metres",
    )
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument(
        "--freeze-rotation", "--freeze_rotation", dest="freeze_rotation", action="store_true"
    )
    rotation.add_argument(
        "--allow-rotation", "--allow_rotation", dest="freeze_rotation", action="store_false"
    )
    parser.set_defaults(freeze_rotation=False)
    tare = parser.add_mutually_exclusive_group()
    tare.add_argument(
        "--zero-ft-on-start",
        "--zero_ft_on_start",
        dest="zero_ft_on_start",
        action="store_true",
    )
    tare.add_argument(
        "--no-zero-ft-on-start",
        "--no_zero_ft_on_start",
        dest="zero_ft_on_start",
        action="store_false",
    )
    parser.set_defaults(zero_ft_on_start=True)
    parser.add_argument("--ft-zero-samples", "--ft_zero_samples", type=int, default=25)
    parser.add_argument(
        "--ft-max-age-sec",
        "--ft_max_age_sec",
        type=float,
        default=0.020,
        help=(
            "Legacy causal-window compatibility value; the RGB/F/T checkpoint "
            "still enforces its saved 12 ms contract"
        ),
    )
    parser.add_argument(
        "--latest-ft-max-age-sec",
        "--latest_ft_max_age_sec",
        type=float,
        default=None,
        help=(
            "Latest F/T age immediately before Teleop/policy commands; default "
            "comes from robot config (currently 50 ms)"
        ),
    )
    parser.add_argument("--save-inputs", "--save_inputs", action="store_true")
    parser.add_argument(
        "--show-policy-image", "--show_policy_image", action="store_true"
    )
    parser.add_argument(
        "--policy-input-audit", "--policy_input_audit", action="store_true"
    )
    parser.add_argument(
        "--print-policy-output", "--print_policy_output", action="store_true"
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--wait-for-start-key",
        "--wait_for_start_key",
        dest="wait_for_start_key",
        action="store_true",
        help="Show alignment view and wait for camera-window key c",
    )
    start.add_argument(
        "--auto-start-policy",
        "--auto_start_policy",
        dest="wait_for_start_key",
        action="store_false",
    )
    parser.set_defaults(wait_for_start_key=False)
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Enable robot/gripper waypoint submission; no typed confirmation",
    )
    motion.add_argument(
        "--plan-only", "--plan_only", dest="execute", action="store_false"
    )
    parser.set_defaults(execute=False)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load and verify both bound checkpoints without opening hardware",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.action_checkpoint = resolve_path(args.action_checkpoint)
    args.context_checkpoint = resolve_path(args.context_checkpoint)
    args.robot_config = resolve_path(args.robot_config)
    args.aruco_config = resolve_path(args.aruco_config)
    if args.match_dataset is not None:
        args.match_dataset = resolve_path(args.match_dataset)
    required_files = [args.action_checkpoint, args.context_checkpoint]
    if not args.inspect:
        required_files.extend([args.robot_config, args.aruco_config])
    missing = [str(path) for path in required_files if not path.is_file()]
    if (
        not args.inspect
        and args.match_dataset is not None
        and not args.match_dataset.exists()
    ):
        missing.append(str(args.match_dataset))
    if missing:
        parser.error("missing required file(s): " + ", ".join(missing))
    positive = (
        args.rate,
        args.command_latency,
        args.max_result_age,
        args.max_processing_age,
        args.max_seconds,
        args.camera_latency,
        args.ft_max_age_sec,
        args.teleop_position_step,
        args.teleop_rotation_step,
        args.teleop_gripper_step,
    )
    if not all(np.isfinite(value) and value > 0 for value in positive):
        parser.error("rate, latencies, max age, and max seconds must be positive")
    if args.latest_ft_max_age_sec is not None and (
        not np.isfinite(args.latest_ft_max_age_sec)
        or args.latest_ft_max_age_sec <= 0
    ):
        parser.error("--latest-ft-max-age-sec must be positive")
    if not 1 <= args.steps_per_inference <= 15:
        parser.error("--steps-per-inference must be in [1,15]")
    if (
        args.max_cycles < 1
        or args.ft_zero_samples < 1
        or args.warmup_runs < 0
        or args.match_episode < 0
    ):
        parser.error(
            "cycle/sample counts must be positive; warmup and match episode nonnegative"
        )
    if not np.isfinite(args.action_scale) or not 0 < args.action_scale <= 1:
        parser.error("--action-scale must be in (0,1]")
    if (
        not np.isfinite(args.motion_momentum_previous_weight)
        or not 0 <= args.motion_momentum_previous_weight < 1
    ):
        parser.error("--motion-momentum-previous-weight must be in [0,1)")
    if (
        not np.isfinite(args.context_momentum_previous_weight)
        or not 0 <= args.context_momentum_previous_weight < 1
    ):
        parser.error("--context-momentum-previous-weight must be in [0,1)")
    if args.inspect:
        device = select_device(args.device)
        cfg, _, _, metadata = load_bound_models(
            args.action_checkpoint, args.context_checkpoint, device, args.weights
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "action_checkpoint": str(args.action_checkpoint),
                    "action_checkpoint_sha256": sha256_file(args.action_checkpoint),
                    "context_checkpoint": str(args.context_checkpoint),
                    "contract": str(cfg.task.model_contract.version),
                    "phase_names": list(PHASE_NAMES),
                    "device": str(device),
                    "model": metadata,
                },
                indent=2,
                default=json_default,
            )
        )
        return
    run_live(args)


if __name__ == "__main__":
    main()
