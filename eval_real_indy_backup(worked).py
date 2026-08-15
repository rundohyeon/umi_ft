"""
Usage:
(umi): python3 scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data
python3 eval_real_indy.py   --robot_config example/eval_robots_config_indy.yaml   -i path/to.ckpt -o data/eval_cup_wild_example  --print_policy_output

Offline ckpt pose_repr / dataset z (no robot):
python3 scripts/indy_umi/inspect_ckpt_pose_eval.py -i path/to/latest.ckpt --zarr auto --stride 20 --episode_z 8

Live vs train z / raw model scale:
python3 eval_real_indy.py ... --pose_eval_audit --dataset_zarr auto

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
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

"""
"""

# %%
import os
import pathlib
import time
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
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from umi.real_world.umi_env import UmiEnv

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _scipy_euler_seq(seq: str, extrinsic: bool) -> str:
    """Match IndyInterpolationController: extrinsic -> UPPER, intrinsic -> lower."""
    es = str(seq)
    return es.upper() if extrinsic else es.lower()


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


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--no_spacemouse', is_flag=True, default=True, help="Disable SpaceMouse and use keyboard teleop only.")
@click.option('--no_gripper', is_flag=True, default=True, help="Run without connecting to gripper hardware.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
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
    "--print_model_input",
    is_flag=True,
    default=False,
    help=(
        "After get_real_umi_obs_dict: print raw env TCP vs policy_obs tensors "
        "(input to predict_action)."
    ),
)
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, no_spacemouse, no_gripper,
    no_mirror, sim_fov, camera_intrinsics, mirror_swap, print_policy_output,
    pose_eval_audit, dataset_zarr, dataset_z_stride, print_model_input):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file (single arm: one robot + one gripper)
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data.get('grippers', [])
    if len(robots_config) != 1:
        raise ValueError('eval_real_indy expects exactly one robot in robot_config YAML.')
    if (not no_gripper) and len(grippers_config) != 1:
        raise ValueError(
            'When --no_gripper is not set, eval_real_indy expects exactly one gripper in robot_config YAML.'
        )
    rc = robots_config[0]
    gc = grippers_config[0] if len(grippers_config) > 0 else {}
    n_robots = 1

    # Human teleop rotation deltas (keyboard / SpaceMouse) use this Euler order.
    teleop_euler_seq = rc.get("indy_teleop_rot_euler_seq", "xyz")
    teleop_euler_extrinsic = rc.get("indy_teleop_rot_euler_extrinsic", False)
    # Policy tcp7 rotvec: optional round-trip through Indy's task Euler chart
    # (indy_task_rot_*) so commands align with movetelel_abs / ActualTCPPose.
    policy_rot_rt = rc.get("indy_policy_tcp7_rot_euler_roundtrip", True)
    policy_rot_seq = rc.get("indy_policy_tcp7_rot_euler_seq")
    policy_rot_ext = rc.get("indy_policy_tcp7_rot_euler_extrinsic")
    if policy_rot_seq is None:
        policy_rot_seq = rc.get("indy_task_rot_euler_seq", "zxz")
    if policy_rot_ext is None:
        policy_rot_ext = rc.get("indy_task_rot_euler_extrinsic", False)

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

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
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
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
        with sm_ctx as sm, UmiEnv(
                output_dir=output,
                robot_ip=rc['robot_ip'],
                gripper_ip=gc.get('gripper_ip'),
                gripper_port=gc.get('gripper_port', 502),
                gripper_slave_id=gc.get('gripper_slave_id', 65),
                gripper_type=gc.get('gripper_type', 'rg2ft'),
                gripper_serial_port=gc.get('gripper_serial_port'),
                dynamixel_id=gc.get('dynamixel_id', 1),
                dynamixel_baudrate=gc.get('dynamixel_baudrate', 57600),
                dynamixel_open_position=gc.get('dynamixel_open_position', 200),
                dynamixel_close_position=gc.get('dynamixel_close_position', 1600),
                dynamixel_max_gripper_width=gc.get(
                    'dynamixel_max_gripper_width', gc.get('max_gripper_width', 0.09)
                ),
                dynamixel_profile_velocity=gc.get('dynamixel_profile_velocity', 30),
                dynamixel_profile_acceleration=gc.get('dynamixel_profile_acceleration', 15),
                dynamixel_move_max_speed=gc.get('dynamixel_move_max_speed', 0.05),
                use_gripper=(not no_gripper),
                robot_type=rc['robot_type'],
                tcp_offset=rc['tcp_offset'],
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                camera_obs_latency=0.17,
                robot_obs_latency=rc['robot_obs_latency'],
                gripper_obs_latency=gc.get('gripper_obs_latency', 0.01),
                robot_action_latency=rc.get('robot_action_latency', 0.1),
                gripper_action_latency=gc.get('gripper_action_latency', 0.1),
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                indy_task_rot_is_euler=rc.get("indy_task_rot_is_euler", True),
                indy_task_rot_euler_seq=rc.get("indy_task_rot_euler_seq", "zxz"),
                indy_task_rot_euler_in_degrees=rc.get(
                    "indy_task_rot_euler_in_degrees", True
                ),
                indy_task_rot_euler_extrinsic=rc.get(
                    "indy_task_rot_euler_extrinsic", False
                ),
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
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
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)
            if pose_eval_audit:
                _print_ckpt_pose_eval_contract(cfg)

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
            episode_start_pose = [
                np.concatenate([
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_rot_axis_angle'],
                ], axis=-1)[-1]
            ]
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose)
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
                action = get_real_umi_action(raw_pred, obs, action_pose_repr)
                action = _apply_policy_tcp7_rot_roundtrip(
                    action,
                    enabled=policy_rot_rt,
                    euler_seq=policy_rot_seq,
                    euler_extrinsic=policy_rot_ext,
                    n_robots=n_robots,
                )
                assert action.shape[-1] == 7 * n_robots
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
            print('- q: quit, c: start policy, e/w: next/prev match episode, g: move to matched start pose')
            print('- a/d: x-,x+ | y/h: y+,y- | r/f: z+,z-')
            print('- j/l: roll-/+ | i/k: pitch+/+ | u/o: yaw-/+')
            if not no_gripper:
                print('- z/x: gripper close/open')
            print(
                "Safety: Ctrl+C ends the script and stops the env controller "
                "(use the robot E-stop if motion does not stop immediately)."
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
                    gripper_target_pos = np.asarray([0.0], dtype=np.float32)

                t_start = time.monotonic()
                iter_idx = 0
                teleop_motion_latch_armed = True
                keyboard_motion_keys = frozenset(
                    {
                        ord("a"),
                        ord("d"),
                        ord("y"),
                        ord("h"),
                        ord("r"),
                        ord("f"),
                        ord("j"),
                        ord("l"),
                        ord("i"),
                        ord("k"),
                        ord("u"),
                        ord("o"),
                    }
                )
                if not no_gripper:
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
    
                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera{match_camera}_rgb'][-1]
                        match_episode_id = episode_id
                        if match_episode is not None:
                            match_episode_id = match_episode
                        if match_episode_id in episode_first_frame_map:
                            match_img = episode_first_frame_map[match_episode_id]
                            ih, iw, _ = match_img.shape
                            oh, ow, _ = vis_img.shape
                            tf = get_image_transform(
                                input_res=(iw, ih), 
                                output_res=(ow, oh), 
                                bgr_to_rgb=False)
                            match_img = tf(match_img).astype(np.float32) / 255
                            vis_img = (vis_img + match_img) / 2
                        # single-arm: one live view (+ optional match overlay above)
                        
                        text = f'Episode: {episode_id}'
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            lineType=cv2.LINE_AA,
                            thickness=3,
                            color=(0,0,0)
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow("default", vis_img[..., ::-1])
                        key = cv2.pollKey() & 0xFF
                        start_policy = False
                        if key == ord("q"):
                            env.end_episode()
                            exit(0)
                        elif key == ord("c"):
                            start_policy = True
                        elif key == ord("e"):
                            if match_episode is not None:
                                match_episode = min(
                                    match_episode + 1,
                                    env.replay_buffer.n_episodes - 1,
                                )
                        elif key == ord("w"):
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key == ord("g") and match_replay_buffer is not None:
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            pos = ep["robot0_eef_pos"][0]
                            rot = ep["robot0_eef_rot_axis_angle"][0]
                            grip = ep["robot0_gripper_width"][0]
                            pose = np.concatenate([pos, rot])
                            t_goal = time.time() + duration
                            if hasattr(env.robot, "servoL"):
                                env.robot.servoL(pose, duration=duration)
                            else:
                                env.robot.schedule_waypoint(pose, target_time=t_goal)
                            if not no_gripper:
                                env.gripper.schedule_waypoint(grip, target_time=t_goal)
                            target_pose[0] = pose
                            gripper_target_pos[0] = grip
                            time.sleep(duration)
                        elif key == 8:
                            if click.confirm("Are you sure to drop an episode?"):
                                env.drop_episode()

                        if key not in keyboard_motion_keys:
                            teleop_motion_latch_armed = True

                        if start_policy:
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
                            if sm.is_button_pressed(0):
                                grip_delta = -gripper_speed / frequency
                            if sm.is_button_pressed(1):
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
                                    dpos[0] -= pos_step
                                elif key == ord("d"):
                                    dpos[0] += pos_step
                                elif key == ord("y"):
                                    dpos[1] += pos_step
                                elif key == ord("h"):
                                    dpos[1] -= pos_step
                                elif key == ord("r"):
                                    dpos[2] += pos_step
                                elif key == ord("f"):
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
                                elif (not no_gripper) and key == ord("z"):
                                    grip_delta = -gripper_speed / frequency
                                elif (not no_gripper) and key == ord("x"):
                                    grip_delta = gripper_speed / frequency

                        target_pose[0, :3] += dpos
                        target_pose[0, 3:] = _human_teleop_compose_rotvec(
                            target_pose[0, 3:],
                            drot_xyz,
                            teleop_euler_seq,
                            teleop_euler_extrinsic,
                        )
    
                        if not no_gripper:
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
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = [
                        np.concatenate([
                            obs['robot0_eef_pos'],
                            obs['robot0_eef_rot_axis_angle'],
                        ], axis=-1)[-1]
                    ]

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=None,
                                episode_start_pose=episode_start_pose)
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
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            action = _apply_policy_tcp7_rot_roundtrip(
                                action,
                                enabled=policy_rot_rt,
                                euler_seq=policy_rot_seq,
                                euler_extrinsic=policy_rot_ext,
                                n_robots=n_robots,
                            )
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
                        
                        # convert policy action to env actions
                        this_target_poses = action
                        assert this_target_poses.shape[1] == 7 * n_robots

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
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

                        if print_policy_output:
                            _print_policy_action_debug(
                                f"[policy iter={iter_idx} -> exec]",
                                raw_action,
                                action,
                                submitted=this_target_poses,
                            )

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs['camera0_rgb'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow("default", vis_img[..., ::-1])
                        key = cv2.pollKey() & 0xFF
                        stop_episode = False
                        if key == ord("s"):
                            print("Stopped.")
                            stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()