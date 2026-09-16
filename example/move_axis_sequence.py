"""
Sends a fixed +/-10cm translation sequence to the robot, using the exact same
tcp7 action format + env.exec_actions() path that eval_real_indy.py uses for
policy output. No checkpoint/policy involved - this is for verifying real-world
axis directions/signs (e.g. indy_task_frame_xyz_signs) against UMI's training
frame convention, independent of any trained model.

Sequence: 1. x+10cm  2. y+10cm  3. z+10cm  4. x-10cm  5. y-10cm  6. z-10cm

python3 move_axis_sequence.py --robot_config example/eval_robots_config_indy.yaml -o data/axis_test
"""

# %%
import os
import pathlib
import time
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import yaml

from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env import UmiEnv


def _overlay_text(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_text_panel(lines, size=320, bg_color=(30, 30, 30)):
    panel = np.full((size, size, 3), bg_color, dtype=np.uint8)
    y = 20
    for line in lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return panel


def _camera_panel_bgr(obs, panel_size=320):
    img = np.asarray(obs["camera0_rgb"][-1])
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    return cv2.resize(img_bgr, (panel_size, panel_size))


def move_axis_sequence(env, step_m=0.10, settle_time=2.5, frequency=10.0,
        video_path=None, panel_size=320):
    """1. x+step 2. y+step 3. z+step 4. x-step 5. y-step 6. z-step, each
    measured against the actual pose right before that step (not compounded).
    If video_path is given, records a left(camera)/right(cm-unit stats) video
    continuously through the whole sequence."""
    steps = [("x", +1), ("y", +1), ("z", +1), ("x", -1), ("y", -1), ("z", -1)]
    axis_idx = {"x": 0, "y": 1, "z": 2}
    dt = 1.0 / frequency

    video_writer = None
    sep = np.full((panel_size, 4, 3), (255, 255, 255), dtype=np.uint8)

    def _write_frame(obs, lines, label):
        nonlocal video_writer
        left = _overlay_text(_camera_panel_bgr(obs, panel_size), label)
        right = _render_text_panel(lines, size=panel_size)
        frame = np.concatenate([left, sep, right], axis=1)
        if video_path is not None:
            if video_writer is None:
                fh, fw = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), frequency, (fw, fh))
            video_writer.write(frame)

    obs = env.get_obs()
    start_pos_cm = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).copy() * 100.0
    print(f"[move_axis_sequence] start pos(cm)={start_pos_cm}")

    for i, (axis, sign) in enumerate(steps):
        obs = env.get_obs()
        cur_pos = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).copy()
        cur_rot = np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64).copy()
        cur_grip = (float(obs["robot0_gripper_width"][-1, 0])
            if "robot0_gripper_width" in obs else 0.0)

        target_pos = cur_pos.copy()
        target_pos[axis_idx[axis]] += sign * step_m
        action = np.concatenate([target_pos, cur_rot, [cur_grip]])
        target_time = time.time() + settle_time
        expected_cm = np.zeros(3)
        expected_cm[axis_idx[axis]] = sign * step_m * 100.0

        label = f"{axis}{'+' if sign > 0 else '-'}{step_m * 100:.0f}cm"
        print(f"[move_axis_sequence] step {i + 1}/6 ({label}): "
            f"cur(cm)={cur_pos * 100} -> target(cm)={target_pos * 100}")
        env.exec_actions(actions=[action], timestamps=[target_time], compensate_latency=False)

        end_t = target_time + 0.5
        while time.time() < end_t:
            loop_obs = env.get_obs()
            live_pos_cm = np.asarray(loop_obs["robot0_eef_pos"][-1], dtype=np.float64) * 100.0
            lines = [
                f"step {i + 1}/6: {label}",
                f"start (cm): {cur_pos * 100}",
                f"target (cm): {target_pos * 100}",
                f"live   (cm): {live_pos_cm}",
                f"delta so far (cm): {live_pos_cm - cur_pos * 100}",
                f"expected (cm): {expected_cm}",
            ]
            _write_frame(loop_obs, lines, f"step {i + 1}/6: {label}")
            precise_wait(time.time() + dt, time_func=time.time)

        obs = env.get_obs()
        actual_pos_cm = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64) * 100.0
        delta_cm = actual_pos_cm - cur_pos * 100.0
        print(
            f"[move_axis_sequence] step {i + 1} result (cm): actual_delta={delta_cm} "
            f"expected={expected_cm} error={delta_cm - expected_cm}"
        )

    obs = env.get_obs()
    end_pos_cm = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64) * 100.0
    print(
        f"[move_axis_sequence] end pos(cm)={end_pos_cm} (start was {start_pos_cm}, "
        f"net drift(cm)={end_pos_cm - start_pos_cm})"
    )

    if video_writer is not None:
        video_writer.release()
        # chown to host user (uid/gid 1000), container runs as root
        try:
            os.chown(video_path, 1000, 1000)
        except Exception:
            pass
        print(f"[move_axis_sequence] saved video to {video_path}")


@click.command()
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--robot_config", "-rc", required=True, help="Path to robot_config yaml file")
@click.option("--camera_reorder", "-cr", default="0")
@click.option("--no_gripper", is_flag=True, default=True, help="Run without connecting to gripper hardware.")
@click.option("--no_mirror", "-nm", is_flag=True, default=False)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz.")
@click.option("--step_m", default=0.10, type=float, help="Step size in meters for each axis move.")
@click.option("--settle_time", default=2.5, type=float, help="Seconds allotted for each step's motion.")
def main(output, robot_config, camera_reorder, no_gripper, no_mirror, mirror_swap,
        frequency, step_m, settle_time):
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), "r"))
    robots_config = robot_config_data["robots"]
    grippers_config = robot_config_data.get("grippers", [])
    if len(robots_config) != 1:
        raise ValueError("move_axis_sequence expects exactly one robot in robot_config YAML.")
    rc = robots_config[0]
    gc = grippers_config[0] if len(grippers_config) > 0 else {}

    obs_res = (224, 224)
    with SharedMemoryManager() as shm_manager:
        with nullcontext(None) as sm, UmiEnv(
                output_dir=output,
                robot_ip=rc["robot_ip"],
                gripper_ip=gc.get("gripper_ip"),
                gripper_port=gc.get("gripper_port", 502),
                gripper_slave_id=gc.get("gripper_slave_id", 65),
                gripper_type=gc.get("gripper_type", "rg2ft"),
                gripper_serial_port=gc.get("gripper_serial_port"),
                dynamixel_id=gc.get("dynamixel_id", 1),
                dynamixel_baudrate=gc.get("dynamixel_baudrate", 57600),
                dynamixel_open_position=gc.get("dynamixel_open_position", 1600),
                dynamixel_close_position=gc.get("dynamixel_close_position", 200),
                dynamixel_max_gripper_width=gc.get(
                    "dynamixel_max_gripper_width", gc.get("max_gripper_width", 0.09)
                ),
                dynamixel_profile_velocity=gc.get("dynamixel_profile_velocity", 30),
                dynamixel_profile_acceleration=gc.get("dynamixel_profile_acceleration", 15),
                dynamixel_move_max_speed=gc.get("dynamixel_move_max_speed", 0.05),
                use_gripper=(not no_gripper),
                robot_type=rc["robot_type"],
                tcp_offset=rc["tcp_offset"],
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=False,
                enable_multi_cam_vis=True,
                camera_obs_latency=0.17,
                robot_obs_latency=rc["robot_obs_latency"],
                gripper_obs_latency=gc.get("gripper_obs_latency", 0.01),
                robot_action_latency=rc.get("robot_action_latency", 0.1),
                gripper_action_latency=gc.get("gripper_action_latency", 0.1),
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                gripper_obs_horizon=2,
                no_mirror=no_mirror,
                fisheye_converter=None,
                mirror_swap=mirror_swap,
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                indy_task_rot_is_euler=rc.get("indy_task_rot_is_euler", True),
                indy_task_rot_euler_seq=rc.get("indy_task_rot_euler_seq", "zxz"),
                indy_task_rot_euler_in_degrees=rc.get("indy_task_rot_euler_in_degrees", True),
                indy_task_rot_euler_extrinsic=rc.get("indy_task_rot_euler_extrinsic", False),
                indy_task_frame_xyz_signs=tuple(rc.get("indy_task_frame_xyz_signs", [1, 1, 1])),
                shm_manager=shm_manager) as env:
            print("Waiting for camera")
            time.sleep(1.0)
            env.start_episode(time.time() + 1.0)

            log_dir = pathlib.Path(output).joinpath(
                "axis_test_logs", time.strftime("%Y%m%d_%H%M%S"))
            log_dir.mkdir(parents=True, exist_ok=True)
            video_path = log_dir.joinpath("axis_test.mp4")
            print(f"[move_axis_sequence] recording video to {video_path}")

            try:
                move_axis_sequence(env, step_m=step_m, settle_time=settle_time,
                    frequency=frequency, video_path=video_path)
            finally:
                env.end_episode()


# %%
if __name__ == "__main__":
    main()
