from typing import Optional
import os
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
import yaml
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.indy_interpolation_controller import IndyInterpolationController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import (
    draw_predefined_mask, 
    get_mirror_crop_slices,
    inpaint_tag,
    get_image_transform as get_crop_ratio_image_transform,
    parse_aruco_config,
    _aruco_make_detector_parameters,
    _aruco_detect_markers,
)
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform as get_resize_crop_image_transform,
    optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
<<<<<<< HEAD
from umi.real_world.rg2ft_obs import causal_ft_history_from_streams
from umi.real_world.rg2ft_startup_bias import subtract_startup_bias
=======
from umi.real_world.rg2ft_obs import (
    causal_ft_history_from_streams,
    compute_ft_tare_offset,
)
>>>>>>> 1ba40c3 (inference debugged)


def _camera_capture_profile(dev_path: str):
    """Return ((width, height), fps, cap_buffer, width/height) for a v4l path."""
    if "HD60" in dev_path or "Game_Capture" in dev_path:
        return (1920, 1080), 60, 3, 16 / 9
    if "Cam_Link_4K" in dev_path:
        return (3840, 2160), 30, 3, 16 / 9
    if "Cam_Link" in dev_path:
        return (1920, 1080), 60, 3, 16 / 9
    # GoPro USB / other UVC (training default)
    return (4000, 3000), 30, 1, 4 / 3


def get_image_transform(
        input_res,
        output_res,
        *,
        crop_ratio: float = 1.0,
        bgr_to_rgb: bool = False):
    if abs(float(crop_ratio) - 1.0) < 1e-9:
        return get_resize_crop_image_transform(
            input_res=input_res,
            output_res=output_res,
            bgr_to_rgb=bgr_to_rgb,
        )
    return get_crop_ratio_image_transform(
        in_res=input_res,
        out_res=output_res,
        crop_ratio=float(crop_ratio),
        bgr_to_rgb=bgr_to_rgb,
    )


class UmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            gripper_ip=None,
            gripper_port=502,
            gripper_slave_id=65,
            gripper_type='rg2ft',
            rg2ft_frequency=100,
            rg2ft_force=20.0,
            rg2ft_home_to_open=False,
            rg2ft_move_max_speed=0.2,
            rg2ft_open_tolerance=0.005,
            rg2ft_zero_on_start=False,
            rg2ft_zero_samples=25,
            gripper_commands_enabled=True,
            gripper_serial_port=None,
            dynamixel_id=1,
            dynamixel_baudrate=57600,
            dynamixel_protocol_version=2.0,
            dynamixel_open_position=2600,
            dynamixel_close_position=200,
            dynamixel_max_gripper_width=0.09,
            dynamixel_profile_velocity=30,
            dynamixel_profile_acceleration=15,
            dynamixel_home_to_open=False,
            dynamixel_current_limit=None,
            dynamixel_pwm_limit=None,
            dynamixel_move_max_speed=0.05,
            use_gripper=True,
            # env params
            frequency=20,
            robot_type='ur5',
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            policy_image_crop_ratio=1.0,
            mask_before_image_transform=False,
            inpaint_aruco_tags=False,
            aruco_config_path=None,
            mirror_crop=False,
            mirror_swap=False,
            # timing
            align_camera_idx=0,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            ft_obs_horizon=0,
            ft_obs_stride=1,
            ft_obs_frequency=100.0,
            ft_max_age=None,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            indy_command_timeout_s=0.3,
            # robot
            tcp_offset=0.235,
            init_joints=False,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # Indy: Neuromeka task UVW are Euler (deg), not rotvec; see robot YAML keys.
            indy_task_rot_is_euler=True,
            indy_task_rot_euler_seq="xyz",
            indy_task_rot_euler_in_degrees=True,
            indy_task_rot_euler_extrinsic=True,
            indy_task_frame_xyz_signs=(1.0, 1.0, 1.0),
            indy_tool_rot_offset_deg=(0.0, 0.0, 0.0),
            # shared memory
            shm_manager=None
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path_obj = output_dir.joinpath('replay_buffer.zarr')
        zarr_path = str(zarr_path_obj.absolute())
        try:
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')
        except (AssertionError, KeyError, ValueError) as exc:
            if not zarr_path_obj.exists():
                raise
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = output_dir.joinpath(
                f"replay_buffer.corrupt_{stamp}.zarr")
            suffix = 1
            while backup_path.exists():
                backup_path = output_dir.joinpath(
                    f"replay_buffer.corrupt_{stamp}_{suffix}.zarr")
                suffix += 1
            shutil.move(str(zarr_path_obj), str(backup_path))
            print(
                "[WARN] Existing replay_buffer.zarr is inconsistent "
                f"({type(exc).__name__}: {exc}) and was moved to "
                f"{backup_path}. Starting a fresh replay buffer."
            )
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        policy_image_crop_ratio = float(policy_image_crop_ratio)
        if not (0.0 < policy_image_crop_ratio <= 1.0):
            raise ValueError(
                "policy_image_crop_ratio must be in (0, 1]. "
                "Use <1 to zoom in; use the matching camera/lens mode if live FOV is too narrow."
            )

        aruco_dict = None
        aruco_params = None
        if inpaint_aruco_tags:
            if aruco_config_path is None:
                raise ValueError(
                    "aruco_config_path must be provided when inpaint_aruco_tags=True"
                )
            with open(os.path.expanduser(str(aruco_config_path)), "r") as f:
                aruco_config = parse_aruco_config(yaml.safe_load(f))
            aruco_dict = aruco_config["aruco_dict"]
            aruco_params = _aruco_make_detector_parameters()
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        def apply_training_image_preprocess(img):
            if aruco_dict is not None:
                corners, ids, _ = _aruco_detect_markers(
                    img, aruco_dict, aruco_params
                )
                if ids is not None:
                    for this_corners in corners:
                        img = inpaint_tag(img, this_corners.squeeze())
            if mask_before_image_transform:
                img = draw_predefined_mask(
                    img,
                    color=(0, 0, 0),
                    mirror=no_mirror,
                    gripper=True,
                    finger=False,
                    use_aa=False,
                )
            return img

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug.
        reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths

        camera_profiles = [_camera_capture_profile(p) for p in v4l_paths]
        print("[camera] selected UVC devices:")
        for cam_idx, (path, profile) in enumerate(zip(v4l_paths, camera_profiles)):
            res, fps, buf, wh_ratio = profile
            print(
                f"  [{cam_idx}] {path} | res={res[0]}x{res[1]} "
                f"fps={fps} buffer={buf} aspect={wh_ratio:.4f}"
            )
        print(f"[camera] policy_image_crop_ratio={policy_image_crop_ratio:.4f}")
        # Match vis tile aspect to capture cards (HD60/HDMI is 16:9, not 4:3).
        vis_in_wh_ratio = camera_profiles[0][3] if camera_profiles else (4 / 3)
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=vis_in_wh_ratio,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Elgato HD60 X records at 1080p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for idx, path in enumerate(v4l_paths):
            res, fps, buf, _wh_ratio = camera_profiles[idx]
            is_hd60 = "HD60" in path or "Game_Capture" in path
            is_cam_link_4k = "Cam_Link_4K" in path
            if is_hd60:
                stack_crop = (idx == 0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224, 224, 3), dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False
                    )
                    is_mirror = mirror_mask[..., 0] == 0

                def tf_hd60(data, stack_crop=stack_crop, is_mirror=is_mirror):
                    img = data['color']
                    img = apply_training_image_preprocess(img)
                    if fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:, ::-1, ::-1]
                        f = get_image_transform(
                            input_res=(img.shape[1], img.shape[0]),
                            output_res=obs_image_resolution,
                            crop_ratio=policy_image_crop_ratio,
                            bgr_to_rgb=True,
                        )
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:, ::-1, :][is_mirror]
                        if not mask_before_image_transform:
                            img = draw_predefined_mask(
                                img, color=(0, 0, 0), mirror=no_mirror, gripper=True, finger=False, use_aa=True
                            )
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[..., ::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data

                transform.append(tf_hd60)
            elif is_cam_link_4k:
                def tf4k(data):
                    img = data['color']
                    img = apply_training_image_preprocess(img)
                    input_res = (img.shape[1], img.shape[0])
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        crop_ratio=policy_image_crop_ratio,
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                stack_crop = (idx==0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)

                def tf(data, stack_crop=stack_crop, is_mirror=is_mirror):
                    img = data['color']
                    img = apply_training_image_preprocess(img)
                    if fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:,::-1,::-1] # bgr to rgb
                        f = get_image_transform(
                            input_res=(img.shape[1], img.shape[0]),
                            output_res=obs_image_resolution, 
                            crop_ratio=policy_image_crop_ratio,
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        if not mask_before_image_transform:
                            img = draw_predefined_mask(img, color=(0,0,0), 
                                mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            # NOTE: Use CPU H.264 encoder by default for broader compatibility.
            # Some environments expose CUDA for torch but still fail to open
            # FFmpeg NVENC (hevc_nvenc) in PyAV.
            video_recorder.append(VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=21,
                buffer_size=512,
            ))

            vis_out_res = (rw, rh)

            def vis_tf(data, vis_out_res=vis_out_res):
                img = data['color']
                if fisheye_converter is not None:
                    img = apply_training_image_preprocess(img)
                    img = fisheye_converter.forward(img)
                f = get_image_transform(
                    input_res=(img.shape[1], img.shape[0]),
                    output_res=vis_out_res,
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None

        if robot_type.startswith('ur5'):
            from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
            robot = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=500, # UR5 CB3 RTDE
                lookahead_time=0.1,
                gain=300,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                launch_timeout=3,
                tcp_offset_pose=[0,0,tcp_offset,0,0,0],
                payload_mass=None,
                payload_cog=None,
                joints_init=j_init,
                joints_init_speed=1.05,
                soft_real_time=False,
                verbose=False,
                receive_keys=None,
                receive_latency=robot_obs_latency
                )
        elif robot_type.startswith('franka'):
            from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
            robot = FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=200,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                verbose=False,
                receive_latency=robot_obs_latency
            )
        elif ('indy' in robot_type) or ('neuromeka' in robot_type):
            robot = IndyInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                robot_type=robot_type,
                frequency=30,
                launch_timeout=3,
                receive_latency=robot_obs_latency,
                verbose=True,
                vel_ratio=0.1,
                acc_ratio=0.5,
                startup_timeout=15.0,
                task_rot_is_euler=indy_task_rot_is_euler,
                task_rot_euler_seq=indy_task_rot_euler_seq,
                task_rot_euler_in_degrees=indy_task_rot_euler_in_degrees,
                task_rot_euler_extrinsic=indy_task_rot_euler_extrinsic,
                task_frame_xyz_signs=indy_task_frame_xyz_signs,
                tool_rot_offset_deg=indy_tool_rot_offset_deg,
                flange_to_tcp_pose=(
                    0.0, 0.0, float(tcp_offset), 0.0, 0.0, 0.0
                ),
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                command_timeout_s=indy_command_timeout_s,
            )
        else:
            raise ValueError(
                f"Unsupported robot_type '{robot_type}'. "
                "Supported: ur5/ur5e, franka, indy*."
            )
        
        gripper = None
        if use_gripper:
            if gripper_type == 'dynamixel':
                from umi.real_world.dynamixel_gripper_controller import (
                    DynamixelGripperController,
                )
                serial_port = gripper_serial_port or gripper_ip
                if serial_port is None:
                    raise ValueError(
                        "gripper_serial_port (or gripper_ip) must be set when gripper_type='dynamixel'"
                    )
                gripper = DynamixelGripperController(
                    shm_manager=shm_manager,
                    port=str(serial_port),
                    baudrate=dynamixel_baudrate,
                    protocol_version=dynamixel_protocol_version,
                    dxl_id=dynamixel_id,
                    open_position=dynamixel_open_position,
                    close_position=dynamixel_close_position,
                    max_gripper_width=dynamixel_max_gripper_width,
                    profile_velocity=dynamixel_profile_velocity,
                    profile_acceleration=dynamixel_profile_acceleration,
                    home_to_open=dynamixel_home_to_open,
                    current_limit=dynamixel_current_limit,
                    pwm_limit=dynamixel_pwm_limit,
                    move_max_speed=dynamixel_move_max_speed,
                    receive_latency=gripper_obs_latency,
                )
            else:
                from umi.real_world.rg2ft_controller import RG2FTController
                if gripper_ip is None:
                    raise ValueError("gripper_ip must be provided when use_gripper=True")
                gripper = RG2FTController(
                    shm_manager=shm_manager,
                    hostname=gripper_ip,
                    port=gripper_port,
                    slave_id=gripper_slave_id,
                    gripper_type=gripper_type,
                    frequency=rg2ft_frequency,
                    force_n=rg2ft_force,
                    home_to_open=rg2ft_home_to_open,
                    move_max_speed=rg2ft_move_max_speed,
                    open_tolerance=rg2ft_open_tolerance,
                    receive_latency=gripper_obs_latency,
                    verbose=True,
                )

        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.use_gripper = use_gripper
        self.gripper_commands_enabled = bool(gripper_commands_enabled)
        self.rg2ft_zero_on_start = bool(
            rg2ft_zero_on_start and use_gripper and gripper_type != 'dynamixel'
        )
        self.rg2ft_zero_samples = int(rg2ft_zero_samples)
        if self.rg2ft_zero_samples <= 0:
            raise ValueError("rg2ft_zero_samples must be positive")
        self.rg2ft_ft_offset = np.zeros(12, dtype=np.float64)
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.ft_obs_horizon = int(ft_obs_horizon)
        self.ft_obs_stride = int(ft_obs_stride)
        self.ft_obs_frequency = float(ft_obs_frequency)
        self.ft_max_age = None if ft_max_age is None else float(ft_max_age)
        self.ft_startup_bias_12d = None
        if self.ft_obs_horizon < 0:
            raise ValueError("ft_obs_horizon must be non-negative")
        if self.ft_max_age is not None and self.ft_max_age < 0:
            raise ValueError("ft_max_age must be non-negative or None")
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        gripper_ready = True
        if self.use_gripper and self.gripper is not None:
            gripper_ready = self.gripper.is_ready
        return self.camera.is_ready and self.robot.is_ready and gripper_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        if self.use_gripper and self.gripper is not None:
            self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # If a worker process crashed, is_ready can become false.
        # Still try to flush episode buffers instead of asserting.
        def _safe_stop(name, fn):
            try:
                fn()
            except Exception as exc:
                print(f"[WARN] UmiEnv.stop: {name} stop failed: {type(exc).__name__}: {exc}")

        _safe_stop("episode", self.end_episode)
        if self.multi_cam_vis is not None:
            _safe_stop("multi_cam_vis", lambda: self.multi_cam_vis.stop(wait=False))
        _safe_stop("robot", lambda: self.robot.stop(wait=False))
        if self.use_gripper and self.gripper is not None:
            _safe_stop("gripper", lambda: self.gripper.stop(wait=False))
        _safe_stop("camera", lambda: self.camera.stop(wait=False))
        if wait:
            _safe_stop("stop_wait", self.stop_wait)

    def start_wait(self):
        camera_timeout_s = 15.0
        camera_deadline = time.monotonic() + camera_timeout_s
        for camera_idx, camera in self.camera.cameras.items():
            remaining = max(0.0, camera_deadline - time.monotonic())
            camera.ready_event.wait(remaining)
            if not camera.is_ready:
                alive = camera.is_alive()
                raise RuntimeError(
                    "UmiEnv.start_wait: camera did not become ready "
                    f"(camera_idx={camera_idx}, alive={alive}, "
                    f"timeout_s={camera_timeout_s}). Check /dev/video*, "
                    "camera permissions, and capture format."
                )
            camera.video_recorder.start_wait()
        if self.use_gripper and self.gripper is not None:
            self.gripper.start_wait()
            if self.rg2ft_zero_on_start:
                self._zero_rg2ft_from_recent_samples()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def _zero_rg2ft_from_recent_samples(self):
        """Match collection-time software tare using recent raw samples."""
        deadline = time.monotonic() + max(
            2.0,
            2.0 * self.rg2ft_zero_samples / max(self.ft_obs_frequency, 1.0),
        )
        samples = None
        last_error = None
        while time.monotonic() < deadline:
            try:
                state = self.gripper.get_all_state()
                samples = np.asarray(state['gripper_ft'], dtype=np.float64)
                if len(samples) >= self.rg2ft_zero_samples:
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(0.01)
        if samples is None or len(samples) == 0:
            detail = "" if last_error is None else f": {last_error}"
            raise RuntimeError(f"F/T auto-zero received no sensor samples{detail}")
        self.rg2ft_ft_offset = compute_ft_tare_offset(
            samples,
            n_avg=self.rg2ft_zero_samples,
        )
        left = np.array2string(self.rg2ft_ft_offset[:6], precision=4)
        right = np.array2string(self.rg2ft_ft_offset[6:], precision=4)
        print(
            "[F/T auto-zero] software tare applied from "
            f"{min(len(samples), self.rg2ft_zero_samples)} samples; "
            f"left={left} right={right}"
        )
    
    def stop_wait(self):
        self.robot.stop_wait()
        if self.use_gripper and self.gripper is not None:
            self.gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ========= async env API ===========
    def set_ft_startup_bias(self, bias_12d):
        bias = np.asarray(bias_12d, dtype=np.float64)
        if bias.shape != (12,) or np.any(~np.isfinite(bias)):
            raise ValueError("live F/T startup bias must be a finite [12] vector")
        self.ft_startup_bias_12d = bias.copy()

    def hold_robot(self):
        """Cancel pending robot/gripper waypoints and hold measured targets."""
        hold = getattr(self.robot, "cancel_and_hold", None)
        if hold is not None:
            hold()
        if self.gripper_commands_enabled and self.gripper is not None:
            gripper_hold = getattr(self.gripper, "cancel_and_hold", None)
            if gripper_hold is not None:
                gripper_hold()

    def get_latest_ft_state(self):
        """Read the newest native F/T sample for feedback/safety, not the camera anchor."""
        if self.gripper is None or self.ft_startup_bias_12d is None:
            raise RuntimeError("latest F/T requested before gripper/bias is ready")
        state = self.gripper.get_state()
        left_raw = np.asarray(state['gripper_ft_left'], dtype=np.float64).reshape(6)
        right_raw = np.asarray(state['gripper_ft_right'], dtype=np.float64).reshape(6)
        left, right = subtract_startup_bias(
            left_raw[None], right_raw[None], self.ft_startup_bias_12d
        )
        return {
            'left_raw': left_raw,
            'right_raw': right_raw,
            'left': left[0],
            'right': right[0],
            'timestamp': float(state['gripper_timestamp']),
        }

    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        if not self.is_ready:
            cam = self.camera.is_ready
            rob = self.robot.is_ready
            g_ok = True
            if self.use_gripper and self.gripper is not None:
                g_ok = self.gripper.is_ready
            cam_alive = all(
                getattr(c, "is_alive", lambda: True)()
                for c in getattr(self.camera, "cameras", {}).values()
            ) if hasattr(self.camera, "cameras") else True
            rob_alive = getattr(self.robot, "is_alive", lambda: True)()
            raise RuntimeError(
                "UmiEnv.get_obs: environment not ready "
                f"(camera_ready={cam}, robot_ready={rob}, gripper_ready={g_ok}; "
                f"camera_workers_alive={cam_alive}, robot_alive={rob_alive}). "
                "Usually a worker process crashed (e.g. UVC queue.Full) or "
                "start_wait was skipped. Restart the env / script."
            )

        # get data
        # 60 Hz, camera_calibrated_timestamp
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 125/500 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = None
        if self.use_gripper and self.gripper is not None:
            last_gripper_data = self.gripper.get_all_state()
        tared_gripper_ft = None
        if last_gripper_data is not None:
            tared_gripper_ft = (
                np.asarray(last_gripper_data['gripper_ft'], dtype=np.float64)
                - self.rg2ft_ft_offset
            )

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            # remap key
            if camera_idx == 0 and self.mirror_crop:
                camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
            else:
                camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        if self.use_gripper and last_gripper_data is not None:
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_width = gripper_interpolator(gripper_obs_timestamps)
            ft_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=tared_gripper_ft,
            )
            robot0_ft = ft_interpolator(gripper_obs_timestamps)
        else:
            # Keep observation key stable for policies trained with gripper channels.
            gripper_width = np.zeros((len(gripper_obs_timestamps), 1), dtype=np.float32)
            robot0_ft = np.zeros((len(gripper_obs_timestamps), 12), dtype=np.float32)
        gripper_obs = {
            'robot0_gripper_width': gripper_width,
            'robot0_ft': robot0_ft
        }
        if self.ft_obs_horizon > 0:
            if last_gripper_data is None:
                raise RuntimeError(
                    "dual-F/T policy requires live RG2-FT data; no gripper "
                    "sensor stream is available"
                )
            if (
                'gripper_ft_left' not in last_gripper_data
                or 'gripper_ft_right' not in last_gripper_data
            ):
                raise RuntimeError(
                    "dual-F/T policy requires distinct left/right F/T streams "
                    "from the live RG2-FT controller"
                )
            if self.ft_startup_bias_12d is None:
                raise RuntimeError(
                    "dual-F/T observation requested before live startup bias calibration"
                )
            # RG2-FT returns both finger wrenches atomically, therefore both
            # side-specific streams currently share this wall-clock timestamp.
            # The assembler nevertheless samples them independently, matching
            # the training dataset's two causal lookups.
            causal_raw = causal_ft_history_from_streams(
                last_gripper_data['gripper_timestamp'],
                np.asarray(last_gripper_data['gripper_ft_left'])
                - self.rg2ft_ft_offset[:6],
                last_gripper_data['gripper_timestamp'],
                np.asarray(last_gripper_data['gripper_ft_right'])
                - self.rg2ft_ft_offset[6:],
                anchor_timestamp=last_timestamp,
                num_steps=self.ft_obs_horizon,
                stride=self.ft_obs_stride,
                frequency=self.ft_obs_frequency,
                max_age=self.ft_max_age,
            )
            corrected_left, corrected_right = subtract_startup_bias(
                causal_raw['robot0_ft_left'],
                causal_raw['robot0_ft_right'],
                self.ft_startup_bias_12d,
            )
            causal = dict(causal_raw)
            causal['robot0_ft_left'] = corrected_left.astype(np.float32)
            causal['robot0_ft_right'] = corrected_right.astype(np.float32)
            causal['robot0_ft_left_raw'] = causal_raw['robot0_ft_left'].copy()
            causal['robot0_ft_right_raw'] = causal_raw['robot0_ft_right'].copy()
            causal['robot0_ft_startup_bias'] = self.ft_startup_bias_12d.copy()
            latest_left_raw = np.asarray(
                last_gripper_data['gripper_ft_left'][-1], dtype=np.float64
            )
            latest_right_raw = np.asarray(
                last_gripper_data['gripper_ft_right'][-1], dtype=np.float64
            )
            latest_left, latest_right = subtract_startup_bias(
                latest_left_raw[None],
                latest_right_raw[None],
                self.ft_startup_bias_12d,
            )
            causal['robot0_ft_left_latest_raw'] = latest_left_raw
            causal['robot0_ft_right_latest_raw'] = latest_right_raw
            causal['robot0_ft_left_latest'] = latest_left[0]
            causal['robot0_ft_right_latest'] = latest_right[0]
            causal['robot0_ft_latest_timestamp'] = float(
                last_gripper_data['gripper_timestamp'][-1]
            )
            gripper_obs.update(causal)

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            if self.use_gripper and last_gripper_data is not None:
                self.obs_accumulator.put(
                    data={
                        'robot0_gripper_width': last_gripper_data['gripper_position'][...,None],
                        'robot0_ft': tared_gripper_ft,
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.ndim == 0:
            timestamps = timestamps.reshape(1)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        is_new = timestamps > receive_time
        # If every timestamp is already in the past (loop jitter / load), the robot
        # would get zero waypoints while callers still advance local targets → bad
        # teleop. Force-schedule the last command slightly in the future instead.
        if len(actions) > 0 and not np.any(is_new):
            actions = actions[[-1]]
            timestamps = np.array([receive_time + 0.002], dtype=np.float64)
            is_new = np.array([True], dtype=bool)

        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i, :6]
            g_actions = new_actions[i, 6:]
            # Clamp so the controller never receives a stale wall-clock target.
            t_cmd = max(float(new_timestamps[i]), receive_time + 1e-3)
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=t_cmd - r_latency,
            )
            if (
                self.use_gripper
                and self.gripper is not None
                and self.gripper_commands_enabled
            ):
                self.gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=t_cmd - g_latency,
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        if not self.camera.is_ready:
            # Camera process already down; nothing to stop cleanly.
            self.obs_accumulator = None
            self.action_accumulator = None
            return
        
        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            if len(self.action_accumulator.timestamps) == 0:
                self.obs_accumulator = None
                self.action_accumulator = None
                return

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                if self.use_gripper and 'robot0_gripper_width' in self.obs_accumulator.timestamps:
                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                        x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                    )
                    episode['robot0_gripper_width'] = gripper_interpolator(timestamps)
                else:
                    episode['robot0_gripper_width'] = np.zeros((len(timestamps), 1), dtype=np.float32)

                if self.use_gripper and 'robot0_ft' in self.obs_accumulator.timestamps:
                    ft_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps['robot0_ft']),
                        x=np.array(self.obs_accumulator.data['robot0_ft'])
                    )
                    episode['robot0_ft'] = ft_interpolator(timestamps)
                else:
                    episode['robot0_ft'] = np.zeros((len(timestamps), 12), dtype=np.float32)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
