"""Dynamixel gripper controller with the same API as RG2FTController / WSGController."""

from __future__ import annotations

import enum
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.precise_sleep import precise_wait
from umi.real_world.dynamixel_controller import (
    PROTOCOL_2_0,
    DynamixelConfig,
    DynamixelPositionController,
)
from umi.shared_memory.shared_memory_queue import Empty, SharedMemoryQueue
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


@dataclass(frozen=True)
class DynamixelGripperConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    protocol_version: float = PROTOCOL_2_0
    dxl_id: int = 1
    open_position: int = 0
    close_position: int = 90
    max_gripper_width: float = 0.09
    profile_velocity: int = 30
    profile_acceleration: int = 15
    current_limit: float | None = None
    pwm_limit: float | None = None
    move_max_speed: float = 0.05


def gripper_width_to_ticks(
    width_m: float,
    *,
    max_gripper_width: float,
    open_position: int,
    close_position: int,
) -> int:
    """Map UMI gripper width (m) to Dynamixel goal ticks.

    width=0 -> closed, width=max_gripper_width -> open.
  """
    if max_gripper_width <= 0:
        raise ValueError("max_gripper_width must be positive")
    open_ratio = float(np.clip(width_m / max_gripper_width, 0.0, 1.0))
    close_ratio = 1.0 - open_ratio
    return int(round(open_position + close_ratio * (close_position - open_position)))


def gripper_ticks_to_width(
    ticks: int,
    *,
    max_gripper_width: float,
    open_position: int,
    close_position: int,
) -> float:
    """Inverse of gripper_width_to_ticks."""
    span = close_position - open_position
    if span == 0:
        return 0.0
    close_ratio = float(np.clip((ticks - open_position) / span, 0.0, 1.0))
    open_ratio = 1.0 - close_ratio
    return open_ratio * max_gripper_width


class DynamixelGripperController(mp.Process):
    """Gripper process compatible with UmiEnv / eval_real_indy."""

    def __init__(
        self,
        shm_manager,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        protocol_version: float = PROTOCOL_2_0,
        dxl_id: int = 1,
        open_position: int = 0,
        close_position: int = 90,
        max_gripper_width: float = 0.09,
        profile_velocity: int = 30,
        profile_acceleration: int = 15,
        frequency: int = 30,
        home_to_open: bool = False,
        current_limit: float | None = None,
        pwm_limit: float | None = None,
        move_max_speed: float = 0.05,
        get_max_k=None,
        command_queue_size: int = 1024,
        launch_timeout: float = 3.0,
        receive_latency: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(name="DynamixelGripperController")
        self.gripper_config = DynamixelGripperConfig(
            port=port,
            baudrate=baudrate,
            protocol_version=protocol_version,
            dxl_id=dxl_id,
            open_position=open_position,
            close_position=close_position,
            max_gripper_width=max_gripper_width,
            profile_velocity=profile_velocity,
            profile_acceleration=profile_acceleration,
            current_limit=current_limit,
            pwm_limit=pwm_limit,
            move_max_speed=move_max_speed,
        )
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        example_cmd = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=command_queue_size,
        )

        example_state = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": 0.0,
            "gripper_ft": np.zeros(12, dtype=np.float64),
            "gripper_busy": 0,
            "gripper_grip_det": 0,
            "gripper_measure_timestamp": time.time(),
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )
        self.ready_event = mp.Event()

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DynamixelGripperController] spawned at {self.pid}")

    def stop(self, wait=True):
        self.input_queue.put({"cmd": Command.SHUTDOWN.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pos, target_time: float):
        if isinstance(pos, np.ndarray):
            pos = pos.item()
        self.input_queue.put(
            {
                "cmd": Command.SCHEDULE_WAYPOINT.value,
                "target_pos": float(pos),
                "target_time": target_time,
            }
        )

    def restart_put(self, start_time: float):
        self.input_queue.put(
            {"cmd": Command.RESTART_PUT.value, "target_time": start_time}
        )

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def _width_to_ticks(self, width_m: float) -> int:
        cfg = self.gripper_config
        return gripper_width_to_ticks(
            width_m,
            max_gripper_width=cfg.max_gripper_width,
            open_position=cfg.open_position,
            close_position=cfg.close_position,
        )

    def _ticks_to_width(self, ticks: int) -> float:
        cfg = self.gripper_config
        return gripper_ticks_to_width(
            ticks,
            max_gripper_width=cfg.max_gripper_width,
            open_position=cfg.open_position,
            close_position=cfg.close_position,
        )

    def run(self):
        cfg = self.gripper_config
        dxl_cfg = DynamixelConfig(
            port=cfg.port,
            baudrate=cfg.baudrate,
            protocol_version=cfg.protocol_version,
            dxl_ids=(cfg.dxl_id,),
            profile_velocity=cfg.profile_velocity,
            profile_acceleration=cfg.profile_acceleration,
            current_limit=cfg.current_limit,
            pwm_limit=cfg.pwm_limit,
        )
        controller = DynamixelPositionController(dxl_cfg)
        controller.connect()
        try:
            controller.configure_position_mode()
            # configure_position_mode may disable torque while changing mode or
            # EEPROM-backed force limits. Re-enable it before any goal command.
            controller.enable_torque()
            curr_ticks = controller.get_present_position(cfg.dxl_id)
            curr_pos = self._ticks_to_width(curr_ticks)
            if self.home_to_open:
                curr_pos = cfg.max_gripper_width
                controller.set_goal_position(cfg.dxl_id, self._width_to_ticks(curr_pos))

            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos, 0, 0, 0, 0, 0]],
            )
            last_pos = curr_pos
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0

            while keep_running:
                t_now = time.monotonic()
                dt = 1 / self.frequency

                try:
                    curr_ticks = controller.get_present_position(cfg.dxl_id)
                    pos = self._ticks_to_width(curr_ticks)
                    busy = int(controller.is_moving(cfg.dxl_id))
                except RuntimeError:
                    pos = last_pos
                    busy = 1

                vel = (pos - last_pos) / dt
                last_pos = pos
                state = {
                    "gripper_state": busy,
                    "gripper_position": pos,
                    "gripper_velocity": vel,
                    "gripper_force": 0.0,
                    "gripper_ft": np.zeros(12, dtype=np.float64),
                    "gripper_busy": busy,
                    "gripper_grip_det": 0,
                    "gripper_measure_timestamp": time.time(),
                    "gripper_receive_timestamp": time.time(),
                    "gripper_timestamp": time.time() - self.receive_latency,
                }
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command["cmd"]
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command["target_pos"]
                        target_time = command["target_time"]
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=cfg.move_max_speed,
                            max_rot_speed=cfg.move_max_speed,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command["target_time"] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                target_width = pose_interp(t_now)[0]
                target_ticks = self._width_to_ticks(target_width)
                try:
                    controller.set_goal_position(cfg.dxl_id, target_ticks)
                except RuntimeError as exc:
                    if self.verbose:
                        print(f"[DynamixelGripperController] set_goal failed: {exc}")

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
        finally:
            controller.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[DynamixelGripperController] disconnected: {cfg.port}")


def run_unit_tests() -> None:
  """Pure unit tests (no hardware)."""
  cfg = dict(max_gripper_width=0.09, open_position=0, close_position=90)

  assert gripper_width_to_ticks(0.0, **cfg) == 90
  assert gripper_width_to_ticks(0.09, **cfg) == 0
  assert gripper_width_to_ticks(0.045, **cfg) == 45

  assert abs(gripper_ticks_to_width(90, **cfg) - 0.0) < 1e-9
  assert abs(gripper_ticks_to_width(0, **cfg) - 0.09) < 1e-9
  assert abs(gripper_ticks_to_width(45, **cfg) - 0.045) < 1e-9

  for width in np.linspace(0.0, 0.09, 10):
      ticks = gripper_width_to_ticks(width, **cfg)
      recovered = gripper_ticks_to_width(ticks, **cfg)
      assert abs(recovered - width) < 0.01

  print("[DynamixelGripperController] unit tests passed")
