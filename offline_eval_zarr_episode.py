#!/usr/bin/env python3
"""Run eval_real_indy-style policy inference on one zarr episode.

This script does not need ROS or robot hardware. It reads one episode from the
0709 zarr, builds the same model input path used by eval_real_indy.py, decodes
the model output into robot-frame TCP7 waypoints, and writes CSV/NPZ files that
can be inspected or replayed in Gazebo by ros2_execute_tcp7_csv.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

import dill
import hydra
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import torch
import yaml
import zarr
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from umi.real_world.real_inference_util import get_real_umi_obs_dict  # noqa: E402

import eval_real_indy as eval_indy  # noqa: E402
from eval_real_indy import (  # noqa: E402
    _apply_policy_tcp7_rot_roundtrip,
    _apply_slam_frame_fix,
    _apply_slam_frame_fix_to_obs,
    _apply_slam_frame_fix_to_start_pose,
    _check_finite_array,
    _decode_real_umi_action_checked,
    _disable_policy_image_transforms,
    _set_robot_dataset_transform,
    _transform_pos_rot_with_T,
    _transform_tcp7_action,
)


DEFAULT_DATASET = ROOT_DIR / "artifacts" / "0709_robot_tcp" / "dataset_robot_tcp.zarr.zip"
DEFAULT_CKPT = ROOT_DIR / "artifacts" / "0709_robot_tcp" / "checkpoints" / "latest.ckpt"
DEFAULT_ROBOT_CONFIG = ROOT_DIR / "example" / "eval_robots_config_indy.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "offline_eval_zarr_episode"


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
    p = pathlib.Path(os.path.expanduser(str(path)))
    candidates = [p] if p.is_absolute() else [pathlib.Path.cwd() / p, ROOT_DIR / p]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    if must_exist:
        raise FileNotFoundError(str(candidates[0]))
    return candidates[0].resolve()


def _episode_bounds(root: zarr.Group, episode: int) -> tuple[int, int, np.ndarray]:
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if not (0 <= int(episode) < len(ends)):
        raise ValueError(f"episode {episode} out of range [0, {len(ends)})")
    start = 0 if int(episode) == 0 else int(ends[int(episode) - 1])
    end = int(ends[int(episode)])
    return start, end, ends


def _register_image_codecs() -> None:
    try:
        import imagecodecs.numcodecs as imagecodecs_numcodecs

        imagecodecs_numcodecs.register_codecs()
    except Exception:
        pass


def _shape_attr(shape_meta: Any, key: str) -> dict:
    return OmegaConf.to_container(shape_meta["obs"][key], resolve=True)


def _obs_indices(
    current_idx: int,
    start_idx: int,
    end_idx: int,
    *,
    horizon: int,
    down_sample_steps: int | float,
    latency_steps: int | float,
) -> np.ndarray:
    idx = np.array(
        [
            current_idx - i * float(down_sample_steps) + float(latency_steps)
            for i in range(int(horizon))
        ],
        dtype=np.float64,
    )[::-1]
    return np.clip(idx, start_idx, end_idx - 1)


def _read_lowdim_horizon(
    arr,
    idx: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    is_rotation: bool = False,
) -> np.ndarray:
    if np.allclose(idx, np.round(idx)):
        return np.asarray(arr[np.round(idx).astype(np.int64)], dtype=np.float64)

    interpolation_start = max(int(np.floor(idx[0])) - 5, start_idx)
    interpolation_end = min(int(np.ceil(idx[-1])) + 2 + 5, end_idx)
    times = np.arange(interpolation_start, interpolation_end, dtype=np.float64)
    values = np.asarray(arr[interpolation_start:interpolation_end], dtype=np.float64)
    if is_rotation:
        slerp = st.Slerp(times, st.Rotation.from_rotvec(values))
        return slerp(idx).as_rotvec()
    interp = si.interp1d(times, values, axis=0, assume_sorted=True)
    return np.asarray(interp(idx), dtype=np.float64)


def _read_rgb_horizon(
    arr,
    current_idx: int,
    start_idx: int,
    *,
    horizon: int,
    down_sample_steps: int,
) -> np.ndarray:
    num_valid = min(int(horizon), (current_idx - start_idx) // int(down_sample_steps) + 1)
    slice_start = current_idx - (num_valid - 1) * int(down_sample_steps)
    out = np.asarray(arr[slice_start : current_idx + 1 : int(down_sample_steps)])
    if out.shape[0] < int(horizon):
        padding = np.repeat(out[:1], int(horizon) - out.shape[0], axis=0)
        out = np.concatenate([padding, out], axis=0)
    return out


def _make_env_obs_robot(
    root: zarr.Group,
    cfg: Any,
    current_idx: int,
    start_idx: int,
    end_idx: int,
    *,
    data_frequency: float,
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    shape_meta = cfg.task.shape_meta

    for key in shape_meta["obs"].keys():
        attr = _shape_attr(shape_meta, key)
        if attr.get("type", "low_dim") == "rgb":
            raw = root["data"][key]
            obs[key] = _read_rgb_horizon(
                raw,
                current_idx,
                start_idx,
                horizon=int(attr["horizon"]),
                down_sample_steps=int(attr["down_sample_steps"]),
            )

    pos_attr = _shape_attr(shape_meta, "robot0_eef_pos")
    pos_idx = _obs_indices(
        current_idx,
        start_idx,
        end_idx,
        horizon=int(pos_attr["horizon"]),
        down_sample_steps=pos_attr["down_sample_steps"],
        latency_steps=pos_attr.get("latency_steps", 0.0),
    )
    rot_attr = _shape_attr(shape_meta, "robot0_eef_rot_axis_angle")
    rot_idx = _obs_indices(
        current_idx,
        start_idx,
        end_idx,
        horizon=int(rot_attr["horizon"]),
        down_sample_steps=rot_attr["down_sample_steps"],
        latency_steps=rot_attr.get("latency_steps", 0.0),
    )
    pos_dataset = _read_lowdim_horizon(
        root["data"]["robot0_eef_pos"], pos_idx, start_idx=start_idx, end_idx=end_idx
    )
    rot_dataset = _read_lowdim_horizon(
        root["data"]["robot0_eef_rot_axis_angle"],
        rot_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        is_rotation=True,
    )
    obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"] = _transform_pos_rot_with_T(
        pos_dataset, rot_dataset, eval_indy._ROBOT_FROM_DATASET_T
    )

    if "robot0_gripper_width" in shape_meta["obs"]:
        grip_attr = _shape_attr(shape_meta, "robot0_gripper_width")
        grip_idx = _obs_indices(
            current_idx,
            start_idx,
            end_idx,
            horizon=int(grip_attr["horizon"]),
            down_sample_steps=grip_attr["down_sample_steps"],
            latency_steps=grip_attr.get("latency_steps", 0.0),
        )
        if "robot0_gripper_width" in root["data"]:
            obs["robot0_gripper_width"] = _read_lowdim_horizon(
                root["data"]["robot0_gripper_width"],
                grip_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            ).astype(np.float32)
        else:
            obs["robot0_gripper_width"] = np.zeros((len(grip_idx), 1), dtype=np.float32)

    obs_horizon = max(v.shape[0] for v in obs.values() if isinstance(v, np.ndarray))
    last_t = current_idx / float(data_frequency)
    dt = 1.0 / (float(data_frequency) / float(pos_attr["down_sample_steps"]))
    obs["timestamp"] = last_t - np.arange(obs_horizon)[::-1] * dt
    return obs


def _tcp7_from_dataset_at(
    root: zarr.Group,
    idx: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    pos = np.asarray(root["data"]["robot0_eef_pos"][idx], dtype=np.float64)
    rot = np.asarray(root["data"]["robot0_eef_rot_axis_angle"][idx], dtype=np.float64)
    if "robot0_gripper_width" in root["data"]:
        grip = np.asarray(root["data"]["robot0_gripper_width"][idx], dtype=np.float64)
        grip = grip.reshape(len(idx), -1)[:, :1]
    else:
        grip = np.zeros((len(idx), 1), dtype=np.float64)
    return np.concatenate([pos, rot, grip], axis=-1)


def _dataset_tcp7_to_robot(tcp7_dataset: np.ndarray) -> np.ndarray:
    arr = np.asarray(tcp7_dataset, dtype=np.float64).copy()
    pos, rot = _transform_pos_rot_with_T(
        arr[:, :3], arr[:, 3:6], eval_indy._ROBOT_FROM_DATASET_T
    )
    arr[:, :3] = pos
    arr[:, 3:6] = rot
    return arr


def _rot_error_rad(a_rotvec: np.ndarray, b_rotvec: np.ndarray) -> np.ndarray:
    ra = st.Rotation.from_rotvec(np.asarray(a_rotvec, dtype=np.float64))
    rb = st.Rotation.from_rotvec(np.asarray(b_rotvec, dtype=np.float64))
    return (ra * rb.inv()).magnitude()


def _load_robot_config(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    robots = data["robots"]
    if len(robots) != 1:
        raise ValueError("offline_eval_zarr_episode expects exactly one robot")
    return robots[0]


def _load_policy(args: argparse.Namespace, rc: dict) -> PolicyBundle:
    ckpt_path = _resolve(args.input)
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is not None:
        cfg.policy.obs_encoder.pretrained = False
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


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten(prefix: str, values: np.ndarray, n: int) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{i}": float(v[i]) for i in range(min(n, len(v)))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = _resolve(args.dataset)
    robot_config_path = _resolve(args.robot_config)
    output_dir = _resolve(args.output, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    rc = _load_robot_config(robot_config_path)
    _set_robot_dataset_transform(rc.get("indy_robot_from_dataset_transform", None))
    print("robot_from_dataset_transform:")
    print(np.array2string(eval_indy._ROBOT_FROM_DATASET_T, precision=6))

    bundle = _load_policy(args, rc)
    print("device:", bundle.device)
    print("obs_pose_repr:", bundle.obs_pose_repr)
    print("action_pose_repr:", bundle.action_pose_repr)

    _register_image_codecs()
    rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    pred_robot_all = []
    demo_robot_all = []
    pred_dataset_all = []
    demo_dataset_all = []
    raw_all = []

    with zarr.ZipStore(str(dataset_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        start_idx, end_idx, episode_ends = _episode_bounds(root, args.episode)
        episode_len = end_idx - start_idx
        action_horizon = int(bundle.cfg.task.shape_meta.action.horizon)
        action_down = int(bundle.cfg.task.shape_meta.action.down_sample_steps)
        obs_down = int(bundle.cfg.task.shape_meta.obs.robot0_eef_pos.down_sample_steps)
        max_current_idx = end_idx - (action_horizon - 1) * action_down - 1
        if max_current_idx < start_idx:
            raise ValueError(
                f"episode {args.episode} too short for action horizon {action_horizon}"
            )

        max_policy_iters = args.max_policy_iters
        current_control_step = 0
        current_idx = start_idx

        first_obs = _make_env_obs_robot(
            root,
            bundle.cfg,
            current_idx,
            start_idx,
            end_idx,
            data_frequency=args.data_frequency,
        )
        episode_start_pose = [
            np.concatenate(
                [
                    first_obs["robot0_eef_pos"],
                    first_obs["robot0_eef_rot_axis_angle"],
                ],
                axis=-1,
            )[-1]
        ]
        episode_start_pose_for_model = _apply_slam_frame_fix_to_start_pose(
            episode_start_pose
        )

        bundle.policy.reset()
        iter_count = 0
        t0 = time.time()
        while current_idx <= max_current_idx:
            if max_policy_iters is not None and iter_count >= int(max_policy_iters):
                break
            env_obs_robot = _make_env_obs_robot(
                root,
                bundle.cfg,
                current_idx,
                start_idx,
                end_idx,
                data_frequency=args.data_frequency,
            )
            obs_for_model = _apply_slam_frame_fix_to_obs(env_obs_robot, 1)
            obs_dict_np = get_real_umi_obs_dict(
                env_obs=obs_for_model,
                shape_meta=bundle.cfg.task.shape_meta,
                obs_pose_repr=bundle.obs_pose_repr,
                tx_robot1_robot0=None,
                episode_start_pose=episode_start_pose_for_model,
            )
            with torch.no_grad():
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(bundle.device),
                )
                result = bundle.policy.predict_action(obs_dict)
                raw_action = result["action_pred"][0].detach().to("cpu").numpy()
            _check_finite_array(f"[iter={current_control_step}] raw_action", raw_action)
            raw_action = _apply_slam_frame_fix(raw_action, 1)
            action_dataset = _decode_real_umi_action_checked(
                raw_action,
                obs_for_model,
                bundle.action_pose_repr,
                f"[iter={current_control_step} dataset]",
            )
            action_robot = _transform_tcp7_action(
                action_dataset, eval_indy._ROBOT_FROM_DATASET_T, 1
            )
            action_robot = _apply_policy_tcp7_rot_roundtrip(
                action_robot,
                enabled=bundle.policy_rot_rt,
                euler_seq=bundle.policy_rot_seq,
                euler_extrinsic=bundle.policy_rot_ext,
                n_robots=1,
            )

            n_exec = min(int(args.steps_per_inference), len(action_robot))
            this_pred_robot = np.asarray(action_robot[:n_exec], dtype=np.float64)
            this_pred_dataset = np.asarray(action_dataset[:n_exec], dtype=np.float64)
            future_idx = current_idx + np.arange(n_exec, dtype=np.int64) * action_down
            this_demo_dataset = _tcp7_from_dataset_at(root, future_idx)
            this_demo_robot = _dataset_tcp7_to_robot(this_demo_dataset)

            pred_robot_all.append(this_pred_robot)
            demo_robot_all.append(this_demo_robot)
            pred_dataset_all.append(this_pred_dataset)
            demo_dataset_all.append(this_demo_dataset)
            raw_all.append(raw_action[:n_exec])

            obs_tcp6 = np.concatenate(
                [
                    env_obs_robot["robot0_eef_pos"][-1],
                    env_obs_robot["robot0_eef_rot_axis_angle"][-1],
                ]
            )
            pos_err = this_pred_robot[:, :3] - this_demo_robot[:, :3]
            rot_err = _rot_error_rad(this_pred_robot[:, 3:6], this_demo_robot[:, 3:6])
            grip_err = this_pred_robot[:, 6] - this_demo_robot[:, 6]

            for k in range(n_exec):
                rel_t = (current_control_step + k) / float(args.frequency)
                row: dict[str, Any] = {
                    "episode": int(args.episode),
                    "policy_iter": int(iter_count),
                    "control_step": int(current_control_step),
                    "source_idx": int(current_idx),
                    "future_idx": int(future_idx[k]),
                    "exec_row": int(k),
                    "rel_time_s": float(rel_t),
                    "pos_err_norm_m": float(np.linalg.norm(pos_err[k])),
                    "rot_err_rad": float(rot_err[k]),
                    "grip_err_m": float(grip_err[k]),
                }
                row.update(_flatten("obs_tcp6", obs_tcp6, 6))
                row.update(_flatten("raw_action10", raw_action[k], 10))
                row.update(_flatten("pred_dataset_tcp7", this_pred_dataset[k], 7))
                row.update(_flatten("pred_robot_tcp7", this_pred_robot[k], 7))
                row.update(_flatten("demo_dataset_tcp7", this_demo_dataset[k], 7))
                row.update(_flatten("demo_robot_tcp7", this_demo_robot[k], 7))
                rows.append(row)

                sim_row = {
                    "rel_time_s": float(rel_t),
                    "episode": int(args.episode),
                    "policy_iter": int(iter_count),
                    "source_idx": int(current_idx),
                    "exec_row": int(k),
                }
                sim_row.update(_flatten("target_tcp7", this_pred_robot[k], 7))
                sim_rows.append(sim_row)

            if iter_count % max(1, int(args.print_every)) == 0:
                print(
                    f"iter {iter_count:04d} source_idx={current_idx} "
                    f"mean_pos_err={float(np.mean(np.linalg.norm(pos_err, axis=1))):.4f}m "
                    f"mean_rot_err={float(np.mean(rot_err)):.4f}rad"
                )

            iter_count += 1
            current_control_step += int(args.steps_per_inference)
            current_idx = start_idx + current_control_step * obs_down

    pred_robot = np.concatenate(pred_robot_all, axis=0) if pred_robot_all else np.zeros((0, 7))
    demo_robot = np.concatenate(demo_robot_all, axis=0) if demo_robot_all else np.zeros((0, 7))
    pred_dataset = np.concatenate(pred_dataset_all, axis=0) if pred_dataset_all else np.zeros((0, 7))
    demo_dataset = np.concatenate(demo_dataset_all, axis=0) if demo_dataset_all else np.zeros((0, 7))
    raw = np.concatenate(raw_all, axis=0) if raw_all else np.zeros((0, 10))
    pos_norm = np.linalg.norm(pred_robot[:, :3] - demo_robot[:, :3], axis=1)
    rot_err = _rot_error_rad(pred_robot[:, 3:6], demo_robot[:, 3:6]) if len(pred_robot) else np.zeros(0)
    grip_abs = np.abs(pred_robot[:, 6] - demo_robot[:, 6]) if len(pred_robot) else np.zeros(0)

    csv_path = output_dir / f"offline_policy_ep{int(args.episode):03d}.csv"
    sim_csv_path = output_dir / f"sim_tcp7_targets_ep{int(args.episode):03d}.csv"
    npz_path = output_dir / f"offline_policy_ep{int(args.episode):03d}.npz"
    summary_path = output_dir / f"summary_ep{int(args.episode):03d}.json"
    _write_csv(csv_path, rows)
    _write_csv(sim_csv_path, sim_rows)
    np.savez_compressed(
        npz_path,
        pred_robot_tcp7=pred_robot,
        demo_robot_tcp7=demo_robot,
        pred_dataset_tcp7=pred_dataset,
        demo_dataset_tcp7=demo_dataset,
        raw_action10=raw,
    )
    summary = {
        "episode": int(args.episode),
        "dataset": str(dataset_path),
        "checkpoint": str(_resolve(args.input)),
        "robot_config": str(robot_config_path),
        "episode_len_frames": int(episode_len),
        "n_policy_iters": int(iter_count),
        "n_submitted_waypoints": int(len(pred_robot)),
        "duration_s": float(sim_rows[-1]["rel_time_s"] if sim_rows else 0.0),
        "frequency": float(args.frequency),
        "data_frequency": float(args.data_frequency),
        "steps_per_inference": int(args.steps_per_inference),
        "robot_from_dataset_transform": eval_indy._ROBOT_FROM_DATASET_T.tolist(),
        "pos_err_norm_m": {
            "mean": float(np.mean(pos_norm)) if len(pos_norm) else None,
            "median": float(np.median(pos_norm)) if len(pos_norm) else None,
            "max": float(np.max(pos_norm)) if len(pos_norm) else None,
        },
        "rot_err_rad": {
            "mean": float(np.mean(rot_err)) if len(rot_err) else None,
            "median": float(np.median(rot_err)) if len(rot_err) else None,
            "max": float(np.max(rot_err)) if len(rot_err) else None,
        },
        "grip_abs_err_m": {
            "mean": float(np.mean(grip_abs)) if len(grip_abs) else None,
            "max": float(np.max(grip_abs)) if len(grip_abs) else None,
        },
        "outputs": {
            "policy_csv": str(csv_path),
            "sim_tcp7_csv": str(sim_csv_path),
            "npz": str(npz_path),
        },
        "elapsed_s": float(time.time() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote:", csv_path)
    print("wrote:", sim_csv_path)
    print("wrote:", npz_path)
    print("wrote:", summary_path)
    print("summary:")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run eval_real_indy-style model input/output on one zarr episode "
            "and emit TCP7 targets for simulation."
        )
    )
    p.add_argument("-i", "--input", default=str(DEFAULT_CKPT), help="Checkpoint path")
    p.add_argument("-d", "--dataset", default=str(DEFAULT_DATASET), help="Zarr zip path")
    p.add_argument("-rc", "--robot_config", default=str(DEFAULT_ROBOT_CONFIG))
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("-e", "--episode", type=int, default=0)
    p.add_argument("--frequency", type=float, default=19.98)
    p.add_argument("--data_frequency", type=float, default=59.94)
    p.add_argument("--steps_per_inference", type=int, default=6)
    p.add_argument("--max_policy_iters", type=int, default=None)
    p.add_argument("--policy_num_inference_steps", type=int, default=16)
    p.add_argument("--disable_eval_image_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p.add_argument("--print_every", type=int, default=5)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
