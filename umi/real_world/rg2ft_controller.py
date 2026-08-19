from __future__ import annotations

import enum
import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from umi.common.precise_sleep import precise_wait
from umi.real_world.rg2ft_protocol import (
    GRIPPER_SPECS,
    RG2FTModbusClient,
    meters_to_width_reg,
    read_ft_status_full,
    width_to_meters,
    write_command,
)
from umi.shared_memory.shared_memory_queue import Empty, SharedMemoryQueue
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


def _schedule_direct_waypoint(pending, *, target_time: float, target_pos: float):
    """Insert a final-width waypoint and replace an overlapping future plan.

    Policy inference periodically replans before all previously queued actions
    have executed.  Match PoseTrajectoryInterpolator's replacement semantics:
    a newly scheduled waypoint removes old waypoints at the same or later
    timestamp, while earlier waypoints remain scheduled.
    """
    target_time = float(target_time)
    target_pos = float(target_pos)
    updated = [item for item in pending if item[0] < target_time]
    updated.append((target_time, target_pos))
    updated.sort(key=lambda item: item[0])
    return updated


def _pop_due_direct_waypoint(pending, *, now: float):
    """Return the newest due target and the still-future waypoint list."""
    due_pos = None
    first_future = 0
    for first_future, (target_time, target_pos) in enumerate(pending):
        if target_time > now:
            break
        due_pos = target_pos
    else:
        first_future = len(pending)
        return [], due_pos

    return pending[first_future:], due_pos


class RG2FTController(mp.Process):
    """Scheduled RG2-FT width control plus width/F/T feedback.

    Widths at this API are metres.  Merely starting the process performs no
    motion command: the controller begins writing grip registers only after a
    waypoint is explicitly scheduled.  Intermediate/closed targets are held,
    while a fully-open target is released after it is reached to avoid the
    open/re-grip oscillation observed during data collection.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        hostname,
        port=502,
        slave_id=65,
        gripper_type="rg2ft",
        frequency=100,
        home_to_open=False,
        move_max_speed=0.2,
        force_n=20.0,
        open_tolerance=0.005,
        reconnect_interval=0.2,
        get_max_k=None,
        command_queue_size=1024,
        launch_timeout=5,
        receive_latency=0.0,
        verbose=False,
    ):
        super().__init__(name="RG2FTController")
        if gripper_type not in GRIPPER_SPECS:
            raise ValueError(
                f"unsupported gripper_type={gripper_type!r}; "
                f"expected one of {sorted(GRIPPER_SPECS)}"
            )
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        if move_max_speed <= 0:
            raise ValueError("move_max_speed must be positive")

        spec = GRIPPER_SPECS[gripper_type]
        max_force_n = spec["max_force"] / 10.0
        self.hostname = str(hostname)
        self.port = int(port)
        self.slave_id = int(slave_id)
        self.gripper_type = gripper_type
        self.frequency = float(frequency)
        self.home_to_open = bool(home_to_open)
        self.move_max_speed = float(move_max_speed)
        self.force_n = float(np.clip(force_n, 0.0, max_force_n))
        self.open_tolerance = max(0.0, float(open_tolerance))
        self.reconnect_interval = max(0.01, float(reconnect_interval))
        self.launch_timeout = float(launch_timeout)
        self.receive_latency = float(receive_latency)
        self.verbose = bool(verbose)

        if get_max_k is None:
            get_max_k = int(self.frequency * 10)

        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                "cmd": Command.SCHEDULE_WAYPOINT.value,
                "target_pos": 0.0,
                "target_time": 0.0,
            },
            buffer_size=command_queue_size,
        )

        state_example = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": self.force_n,
            "gripper_ft": np.zeros(12, dtype=np.float64),
            "gripper_ft_left": np.zeros(6, dtype=np.float64),
            "gripper_ft_right": np.zeros(6, dtype=np.float64),
            "gripper_busy": 0,
            "gripper_grip_det": 0,
            "gripper_measure_timestamp": time.time(),
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency,
        )

        self.ready_event = mp.Event()
        self.startup_error_queue = mp.Queue(maxsize=1)

    # ========= launch methods ==========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RG2FTController] process spawned pid={self.pid}")

    def stop(self, wait=True):
        if self.is_alive():
            self.input_queue.put({"cmd": Command.SHUTDOWN.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        if not self.ready_event.wait(self.launch_timeout):
            raise TimeoutError(
                f"RG2-FT did not become ready at {self.hostname}:{self.port} "
                f"within {self.launch_timeout}s"
            )
        try:
            startup_error = self.startup_error_queue.get_nowait()
        except Empty:
            startup_error = None
        if startup_error is not None:
            self.join(timeout=0.5)
            raise RuntimeError(f"RG2-FT startup failed: {startup_error}")
        if not self.is_alive():
            raise RuntimeError("RG2-FT controller exited during startup")

    def stop_wait(self):
        self.join(timeout=max(2.0, self.launch_timeout))
        if self.is_alive():
            self.terminate()
            self.join(timeout=1.0)

    @property
    def is_ready(self):
        return self.ready_event.is_set() and self.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods =========
    def schedule_waypoint(self, pos, target_time: float):
        pos_arr = np.asarray(pos, dtype=np.float64)
        if pos_arr.size != 1:
            raise ValueError(f"RG2-FT width command must be scalar, got {pos_arr.shape}")
        self.input_queue.put(
            {
                "cmd": Command.SCHEDULE_WAYPOINT.value,
                "target_pos": float(pos_arr.item()),
                "target_time": float(target_time),
            }
        )

    def restart_put(self, start_time):
        self.input_queue.put(
            {
                "cmd": Command.RESTART_PUT.value,
                "target_time": float(start_time),
            }
        )

    # ========= receive APIs ============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= process loop =============
    def run(self):
        spec = GRIPPER_SPECS[self.gripper_type]
        max_width_reg = int(spec["max_width"])
        max_width_m = width_to_meters(max_width_reg)
        force_reg = int(round(self.force_n * 10.0))
        client = RG2FTModbusClient(
            self.hostname,
            port=self.port,
            slave_id=self.slave_id,
            timeout=1.0,
        )

        try:
            client.connect()
            ft, width_reg, busy, grip_det, sta = read_ft_status_full(client)
        except Exception as exc:
            try:
                self.startup_error_queue.put_nowait(repr(exc))
            except Exception:
                pass
            self.ready_event.set()
            client.close()
            return

        curr_pos = float(np.clip(width_to_meters(width_reg), 0.0, max_width_m))
        pending_waypoints = []
        target_active = False
        final_target_pos = curr_pos
        if self.home_to_open:
            # Kept as an explicit opt-in only; default startup is read-only.
            final_target_pos = max_width_m
            target_active = True

        last_pos = curr_pos
        keep_running = True
        t_start = time.monotonic()
        iter_idx = 0
        self.ready_event.set()
        if self.verbose:
            print(
                f"[RG2FTController] connected {self.hostname}:{self.port} "
                f"slave={self.slave_id} width={curr_pos:.5f}m force={self.force_n:.1f}N"
            )

        try:
            while keep_running:
                t_now = time.monotonic()
                dt = 1.0 / self.frequency

                try:
                    ft, width_reg, busy, grip_det, sta = read_ft_status_full(client)
                except Exception as exc:
                    if self.verbose:
                        print(f"[RG2FTController] Modbus error: {exc}; reconnecting")
                    client.close()
                    time.sleep(self.reconnect_interval)
                    try:
                        client.connect()
                    except Exception:
                        continue
                    t_start = time.monotonic() - dt * iter_idx
                    continue

                pos = float(np.clip(width_to_meters(width_reg), 0.0, max_width_m))
                vel = (pos - last_pos) / dt
                last_pos = pos
                now_wall = time.time()
                self.ring_buffer.put(
                    {
                        "gripper_state": busy,
                        "gripper_position": pos,
                        "gripper_velocity": vel,
                        "gripper_force": self.force_n,
                        # The RG2-FT status block contains both native finger
                        # wrenches in one atomic Modbus response.  Publish the
                        # legacy combined field for recording compatibility and
                        # side-specific streams for dual-F/T policy assembly.
                        # The arrays are copies so a consumer cannot alias one
                        # finger into the other.
                        "gripper_ft": ft,
                        "gripper_ft_left": ft[:6].copy(),
                        "gripper_ft_right": ft[6:].copy(),
                        "gripper_busy": busy,
                        "gripper_grip_det": grip_det,
                        "gripper_measure_timestamp": now_wall,
                        "gripper_receive_timestamp": now_wall,
                        "gripper_timestamp": now_wall - self.receive_latency,
                    }
                )

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    cmd = commands["cmd"][i]
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        requested_pos = float(
                            np.clip(commands["target_pos"][i], 0.0, max_width_m)
                        )
                        target_wall_time = float(commands["target_time"][i])
                        target_mono_time = (
                            t_now + target_wall_time - time.time()
                        )
                        pending_waypoints = _schedule_direct_waypoint(
                            pending_waypoints,
                            target_time=target_mono_time,
                            target_pos=requested_pos,
                        )
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = (
                            float(commands["target_time"][i])
                            - time.time()
                            + time.monotonic()
                        )
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                if not keep_running:
                    break

                pending_waypoints, due_target_pos = _pop_due_direct_waypoint(
                    pending_waypoints,
                    now=t_now,
                )
                if due_target_pos is not None:
                    final_target_pos = float(due_target_pos)
                    target_active = True
                    if self.verbose:
                        print(
                            "[RG2FTController] direct final target "
                            f"{final_target_pos:.5f}m"
                        )

                if target_active:
                    # RG2-FT performs its own internal motion.  Repeatedly send
                    # the same final target; changing intermediate targets at
                    # 100 Hz makes the gripper restart and move stop-start.
                    target_reg = meters_to_width_reg(
                        final_target_pos, max_width_reg
                    )
                    try:
                        if not sta:
                            write_command(
                                client,
                                out_zero=0,
                                r_gfr=force_reg,
                                r_gwd=target_reg,
                                r_ctr=1,
                            )
                    except Exception as exc:
                        if self.verbose:
                            print(f"[RG2FTController] command error: {exc}")
                        client.close()

                    fully_open_target = (
                        final_target_pos >= max_width_m - self.open_tolerance
                    )
                    reached_open = (
                        abs(pos - final_target_pos) <= self.open_tolerance
                    )
                    if fully_open_target and reached_open:
                        target_active = False

                iter_idx += 1
                precise_wait(
                    t_end=t_start + (1.0 / self.frequency) * iter_idx,
                    time_func=time.monotonic,
                )
        finally:
            client.close()
            self.ready_event.set()
            if self.verbose:
                print(f"[RG2FTController] disconnected {self.hostname}")
