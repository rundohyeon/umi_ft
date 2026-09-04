import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import scipy.spatial.transform as st

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty, Full)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.pose_util import pose_to_mat, mat_to_pose
from diffusion_policy.common.precise_sleep import precise_wait


OP_IDLE = 5
OP_MOVING = 6
TELE_OP = 17
OP_RECOVER_STATES = {2, 3, 4, 8, 15, 16}
TELE_TASK_ABSOLUTE = 0


class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    HOLD = 2


def _to_deg(rad_vec):
    return np.degrees(np.asarray(rad_vec, dtype=np.float64))


def _to_rad(deg_vec):
    return np.radians(np.asarray(deg_vec, dtype=np.float64))


class IndyInterpolationController(mp.Process):
    """
    Best-effort Indy controller for RealEnv/UmiEnv compatibility.
    It discovers available Neuromeka API methods at runtime.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        robot_type="indyrp2",
        frequency=30,
        launch_timeout=3,
        get_max_k=None,
        receive_latency=0.0,
        verbose=False,
        vel_ratio=0.2,
        acc_ratio=1.0,
        startup_timeout=15.0,
        # Neuromeka task pose is XYZ + UVW (degrees), with U/V/W rotations about
        # X/Y/Z and R = Rz(W) @ Ry(V) @ Rx(U). This is scipy extrinsic 'xyz'.
        task_rot_is_euler=True,
        task_rot_euler_seq="xyz",
        task_rot_euler_in_degrees=True,
        # scipy: lowercase seq = extrinsic; UPPER = intrinsic.
        task_rot_euler_extrinsic=True,
        # Per-axis sign map between Neuromeka task XYZ and UMI training frame.
        # Example: [-1, 1, -1] when UMI +x,+z corresponds to robot -x,-z.
        task_frame_xyz_signs=(1.0, 1.0, 1.0),
        # Fixed rotation offset between the robot-reported TCP frame and the
        # UMI/SLAM gripper frame, as intrinsic xyz Euler deg applied tool-side:
        # R_reported = R_umi @ R_offset. Feedback strips it, commands re-add it.
        tool_rot_offset_deg=(0.0, 0.0, 0.0),
        # Physical fingertip TCP in the unconfigured flange/tool0 frame:
        # xyz metres + rotation-vector radians. CONTY has no registered TCP.
        flange_to_tcp_pose=(0.0, 0.0, 0.235, 0.0, 0.0, 0.0),
        max_pos_speed=0.5,
        max_rot_speed=1.0,
        command_timeout_s=0.3,
    ):
        super().__init__(name="IndyPositionalController")
        self.robot_ip = robot_ip
        self.robot_type = str(robot_type).lower()
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.robot_dof = 7 if "rp2" in self.robot_type else 6
        self.vel_ratio = float(vel_ratio)
        self.acc_ratio = float(acc_ratio)
        self.startup_timeout = float(startup_timeout)
        self.task_rot_is_euler = bool(task_rot_is_euler)
        self.task_rot_euler_seq = str(task_rot_euler_seq)
        self.task_rot_euler_in_degrees = bool(task_rot_euler_in_degrees)
        self.task_rot_euler_extrinsic = bool(task_rot_euler_extrinsic)
        es = str(task_rot_euler_seq)
        self._task_euler_scipy_seq = (
            es.lower() if self.task_rot_euler_extrinsic else es.upper()
        )
        self.task_frame_xyz_signs = np.asarray(
            task_frame_xyz_signs, dtype=np.float64
        ).reshape(3)
        self.tool_rot_offset_deg = np.asarray(
            tool_rot_offset_deg, dtype=np.float64
        ).reshape(3)
        self._has_tool_rot_offset = bool(np.any(self.tool_rot_offset_deg != 0.0))
        self._tool_rot_offset = st.Rotation.from_euler(
            "xyz", self.tool_rot_offset_deg, degrees=True
        )
        self.flange_to_tcp_pose = np.asarray(
            flange_to_tcp_pose, dtype=np.float64
        ).reshape(6)
        self._tx_flange_tcp = pose_to_mat(self.flange_to_tcp_pose)
        self._tx_tcp_flange = np.linalg.inv(self._tx_flange_tcp)
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.command_timeout_s = float(command_timeout_s)
        if self.max_pos_speed <= 0 or self.max_rot_speed <= 0:
            raise ValueError("Indy speed limits must be positive")
        if self.command_timeout_s <= 0:
            raise ValueError("Indy command timeout must be positive")

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        input_example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=input_example,
            buffer_size=256,
        )

        rb_example = {
            "ActualTCPPose": np.zeros((6,), dtype=np.float64),
            "TargetTCPPose": np.zeros((6,), dtype=np.float64),
            "ActualQ": np.zeros((self.robot_dof,), dtype=np.float64),
            "ActualQd": np.zeros((self.robot_dof,), dtype=np.float64),
            "robot_receive_timestamp": time.time(),
            "robot_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=rb_example,
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
            print(f"[IndyPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        try:
            self.input_queue.put({"cmd": Command.STOP.value})
        except Full:
            print(
                "[IndyPositionalController] stop queue is full; "
                "terminating controller process without queued STOP."
            )
            if self.is_alive():
                self.terminate()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        # Process can crash after ready_event is set.
        return self.ready_event.is_set() and self.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pose, target_time):
        pose = np.asarray(pose, dtype=np.float64)
        assert pose.shape == (6,)
        self.input_queue.put(
            {
                "cmd": Command.SCHEDULE_WAYPOINT.value,
                "target_pose": pose,
                "target_time": float(target_time),
            }
        )

    def cancel_and_hold(self):
        """Discard queued waypoints and disarm task-space streaming."""
        self.input_queue.clear()
        self.input_queue.put({"cmd": Command.HOLD.value})

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    @staticmethod
    def _remap_task_frame_pose(
        pose_m_rad: np.ndarray, signs: np.ndarray
    ) -> np.ndarray:
        """Map task pose between Neuromeka native frame and UMI training frame."""
        signs = np.asarray(signs, dtype=np.float64).reshape(3)
        if np.allclose(signs, 1.0):
            return np.asarray(pose_m_rad, dtype=np.float64)
        pose = np.asarray(pose_m_rad, dtype=np.float64).copy()
        S = np.diag(signs)
        pose[:3] = signs * pose[:3]
        rot = st.Rotation.from_rotvec(pose[3:6])
        pose[3:6] = (st.Rotation.from_matrix(S @ rot.as_matrix() @ S)).as_rotvec()
        return pose

    def _robot_to_umi_pose(self, pose_m_rad: np.ndarray) -> np.ndarray:
        pose = self._remap_task_frame_pose(pose_m_rad, self.task_frame_xyz_signs)
        if self._has_tool_rot_offset:
            pose = pose.copy()
            rot = st.Rotation.from_rotvec(pose[3:6])
            pose[3:6] = (rot * self._tool_rot_offset.inv()).as_rotvec()
        return pose

    def _umi_to_robot_pose(self, pose_m_rad: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_m_rad, dtype=np.float64)
        if self._has_tool_rot_offset:
            pose = pose.copy()
            rot = st.Rotation.from_rotvec(pose[3:6])
            pose[3:6] = (rot * self._tool_rot_offset).as_rotvec()
        return self._remap_task_frame_pose(pose, self.task_frame_xyz_signs)

    def _extract_pose_m_rad(self, control_data):
        pose = None
        pose_candidates = [
            "p",
            "tcp",
            "tcp_pose",
            "task_pos",
            "task_pose",
            "tpos",
            "pos",
            "ActualTCPPose",
        ]
        for key in pose_candidates:
            val = control_data.get(key, None)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).reshape(-1)
                if arr.size >= 6:
                    pose = arr[:6]
                    break
        if pose is None:
            pose = np.zeros((6,), dtype=np.float64)

        # Heuristic conversion: Indy task spaces are often mm + deg.
        if np.max(np.abs(pose[:3])) > 10.0:
            pose[:3] = pose[:3] / 1000.0
        if self.task_rot_is_euler:
            euler = np.asarray(pose[3:6], dtype=np.float64)
            if self.task_rot_euler_in_degrees:
                euler = np.radians(euler)
            rot = st.Rotation.from_euler(
                self._task_euler_scipy_seq, euler, degrees=False
            )
            pose[3:6] = rot.as_rotvec()
        else:
            if np.max(np.abs(pose[3:])) > 2.0 * np.pi:
                pose[3:] = _to_rad(pose[3:])
        # Neuromeka reports the unconfigured flange/tool0 pose. Expose the
        # physical fingertip TCP pose to UMI/model observation code.
        tcp_pose = mat_to_pose(pose_to_mat(pose) @ self._tx_flange_tcp)
        return self._robot_to_umi_pose(tcp_pose)

    def _extract_joint_rad(self, control_data):
        q = np.asarray(control_data.get("q", np.zeros((self.robot_dof,))), dtype=np.float64).reshape(-1)
        qd = np.asarray(control_data.get("qdot", np.zeros((self.robot_dof,))), dtype=np.float64).reshape(-1)
        if q.size < self.robot_dof:
            q = np.pad(q, (0, self.robot_dof - q.size))
        if qd.size < self.robot_dof:
            qd = np.pad(qd, (0, self.robot_dof - qd.size))
        # Neuromeka joint feedback is degree in indy_driver.py
        q = _to_rad(q[: self.robot_dof])
        qd = _to_rad(qd[: self.robot_dof])
        return q, qd

    def _ensure_idle(self, indy, timeout_s: float = 30.0) -> int:
        """Match indy_driver: stop motion/teleop and wait for OP_IDLE."""
        if hasattr(indy, "stop_motion"):
            try:
                indy.stop_motion()
                time.sleep(0.1)
            except Exception:
                pass
        if hasattr(indy, "stop_teleop"):
            try:
                indy.stop_teleop()
                time.sleep(0.3)
            except Exception:
                pass

        deadline = time.time() + timeout_s
        last_op = -1
        while time.time() < deadline:
            try:
                last_op = int(indy.get_control_data().get("op_state", -1))
            except Exception:
                time.sleep(0.1)
                continue
            if last_op == OP_IDLE:
                return last_op
            if last_op in OP_RECOVER_STATES and hasattr(indy, "recover"):
                try:
                    if self.verbose:
                        print(f"[IndyPositionalController] op_state={last_op}, calling recover()")
                    indy.recover()
                    time.sleep(0.5)
                except Exception as exc:
                    if self.verbose:
                        print(f"[IndyPositionalController] recover failed: {exc}")
            time.sleep(0.1)
        return last_op

    def _enter_teleop(self, indy, method: int = TELE_TASK_ABSOLUTE) -> None:
        """Enter TELE_OP before movetele* streaming (indy_driver pattern)."""
        last_op = self._ensure_idle(indy, timeout_s=max(self.startup_timeout, 10.0))
        if last_op != OP_IDLE and self.verbose:
            print(
                f"[IndyPositionalController] warning: op_state={last_op} before teleop "
                f"(wanted {OP_IDLE})"
            )

        deadline = time.time() + self.startup_timeout
        last_start = 0.0
        while time.time() < deadline:
            try:
                last_op = int(indy.get_control_data().get("op_state", -1))
                if last_op == TELE_OP:
                    if self.verbose:
                        print("[IndyPositionalController] entered TELE_OP")
                    return
            except Exception:
                pass

            if hasattr(indy, "start_teleop") and (time.time() - last_start) >= 0.5:
                try:
                    indy.start_teleop(method=method)
                except Exception as exc:
                    if self.verbose:
                        print(f"[IndyPositionalController] start_teleop failed: {exc}")
                last_start = time.time()
            time.sleep(0.2)

        raise RuntimeError(
            f"Failed to enter TELE_OP before control loop (op_state={last_op}, "
            f"wanted {TELE_OP}). Check teach pendant: AUTO mode on, faults cleared, "
            "no other teleop client connected."
        )

    def _find_task_cmd(self, indy):
        # UMI supplies base-frame absolute targets. A relative API must never
        # receive these same six numbers as a displacement.
        teleop_candidates = [
            "movetelel_abs",
            "movetelet_abs",
            "movetele_task_abs",
            "moveteletask_abs",
            "movetelepos_abs",
        ]
        direct_move_candidates = [
            "movel",
            "task_move_abs",
        ]
        for name in teleop_candidates:
            if hasattr(indy, name):
                return getattr(indy, name), name, True
        for name in direct_move_candidates:
            if hasattr(indy, name):
                return getattr(indy, name), name, False
        return None, None, False

    def _send_task_pose(self, cmd_fn, cmd_name, target_pose_m_rad, prefer_mm_deg=True):
        # prefer_mm_deg True: teleop-style APIs (often mm + Euler deg).
        # False: direct movel etc. often expect SI m + axis-angle (rotvec), not Euler(rad).
        # UMI schedules physical fingertip TCP poses. Convert them back to the
        # unconfigured flange/tool0 pose required by Neuromeka/CONTY.
        tcp_pose_m_rad = self._umi_to_robot_pose(
            np.asarray(target_pose_m_rad, dtype=np.float64)
        )
        pose_m_rad = mat_to_pose(
            pose_to_mat(tcp_pose_m_rad) @ self._tx_tcp_flange
        )
        pos_m = pose_m_rad[:3].copy()
        rotvec = pose_m_rad[3:6].copy()
        pose_si_rotvec = pose_m_rad.tolist()

        if self.task_rot_is_euler:
            euler_rad = st.Rotation.from_rotvec(rotvec).as_euler(
                self._task_euler_scipy_seq, degrees=False
            )
            euler_deg = np.degrees(euler_rad)
            mm_deg_euler = np.concatenate([pos_m * 1000.0, euler_deg]).tolist()
            m_euler_rad = np.concatenate([pos_m, euler_rad]).tolist()
            if prefer_mm_deg:
                pose_try_list = [mm_deg_euler, m_euler_rad, pose_si_rotvec]
            else:
                pose_try_list = [pose_si_rotvec, m_euler_rad, mm_deg_euler]
        else:
            mm_deg_legacy = np.concatenate([pos_m * 1000.0, _to_deg(rotvec)]).tolist()
            if prefer_mm_deg:
                pose_try_list = [mm_deg_legacy, pose_si_rotvec]
            else:
                pose_try_list = [pose_si_rotvec, mm_deg_legacy]

        last_exc = None

        def _check_command_response(response):
            """Raise on a controller-side rejection returned by IndyDCP3.

            The Neuromeka ``movetelel_abs`` API returns a mapping such as
            ``{"code": 0, "msg": "..."}`` instead of raising on a rejected
            motion request.  Treat a non-zero response as a failed command;
            otherwise evaluation can log a submitted waypoint while the robot
            remains stationary.  Other command adapters may return ``None``;
            retain their existing success behaviour.
            """
            if not isinstance(response, dict) or "code" not in response:
                return
            try:
                code = int(response["code"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Indy {cmd_name} returned an invalid response: {response!r}"
                ) from exc
            if code != 0:
                raise RuntimeError(
                    f"Indy {cmd_name} rejected task waypoint: "
                    f"code={code}, msg={response.get('msg', '')!r}"
                )

        for pose_list in pose_try_list:
            kwargs_candidates = [
                {"tpos": pose_list, "vel_ratio": self.vel_ratio, "acc_ratio": self.acc_ratio},
                {"pose": pose_list, "vel_ratio": self.vel_ratio, "acc_ratio": self.acc_ratio},
                {"task": pose_list, "vel_ratio": self.vel_ratio, "acc_ratio": self.acc_ratio},
                {"target": pose_list, "vel_ratio": self.vel_ratio, "acc_ratio": self.acc_ratio},
            ]

            for kwargs in kwargs_candidates:
                try:
                    _check_command_response(cmd_fn(**kwargs))
                    return
                except TypeError:
                    continue
                except Exception as exc:
                    last_exc = exc
                    break

            try:
                _check_command_response(cmd_fn(pose_list))
                return
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"Failed to send pose via {cmd_name}: {last_exc}")

    def run(self):
        try:
            from neuromeka import IndyDCP3
        except Exception as exc:
            raise RuntimeError(
                "neuromeka package is required for Indy controller. Install with `pip install neuromeka`."
            ) from exc

        indy = IndyDCP3(self.robot_ip)
        cmd_fn = None
        cmd_name = None
        needs_teleop = False
        pose_interp = None
        last_target_t = None
        armed = False
        armed_deadline = None

        try:
            cmd_fn, cmd_name, needs_teleop = self._find_task_cmd(indy)
            if cmd_fn is None:
                raise RuntimeError(
                    "Could not find an Indy task-space command method. "
                    "Expected one of movetelet_abs/movel variants."
                )
            if self.verbose:
                print(
                    f"[IndyPositionalController] command mode: {cmd_name}, "
                    f"teleop_required={needs_teleop}"
                )

            # Enter TELE_OP only when using movetele* style APIs.
            if needs_teleop:
                self._enter_teleop(indy, method=TELE_TASK_ABSOLUTE)
            else:
                # For direct move APIs (e.g., movel), teleop mode may block commands.
                if hasattr(indy, "stop_teleop"):
                    try:
                        indy.stop_teleop()
                        time.sleep(0.1)
                    except Exception:
                        pass

            curr_data = indy.get_control_data()
            curr_pose = self._extract_pose_m_rad(curr_data)
            curr_t = time.monotonic()
            pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])
            last_target_t = curr_t

            if self.verbose:
                print(f"[IndyPositionalController] Connected to robot: {self.robot_ip}, cmd={cmd_name}")
                if not np.allclose(self.task_frame_xyz_signs, 1.0):
                    print(
                        f"[IndyPositionalController] task_frame_xyz_signs="
                        f"{self.task_frame_xyz_signs.tolist()} "
                        "(UMI training frame <-> Neuromeka task XYZ)"
                    )
                if self._has_tool_rot_offset:
                    print(
                        f"[IndyPositionalController] tool_rot_offset_deg="
                        f"{self.tool_rot_offset_deg.tolist()} "
                        "(R_reported = R_umi @ R_offset; feedback strips, command re-adds)"
                    )
                print(
                    "[IndyPositionalController] flange_to_tcp_pose="
                    f"{self.flange_to_tcp_pose.tolist()} "
                    "(feedback: flange@T; command: tcp@inv(T))"
                )

            dt = 1.0 / float(self.frequency)
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            pose_debug_once = [False]
            while keep_running:
                t_now = time.monotonic()

                try:
                    control_data = indy.get_control_data()
                except Exception as exc:
                    print(f"[IndyPositionalController] get_control_data failed: {exc}")
                    break
                if self.verbose and not pose_debug_once[0]:
                    pose_debug_once[0] = True
                    raw_p = control_data.get("p", None)
                    print(f"[Indy] raw get_control_data()['p']={raw_p!r}")
                    ap = self._extract_pose_m_rad(control_data)
                    print(f"[Indy] interpreted TCP (m, rotvec rad)={np.array2string(ap, precision=5)}")
                    print(
                        f"[Indy] cmd={cmd_name} teleop={needs_teleop} "
                        f"euler_scipy_seq={self._task_euler_scipy_seq!r} "
                        f"rot_is_euler={self.task_rot_is_euler}"
                    )
                actual_pose = self._extract_pose_m_rad(control_data)
                pose_cmd = actual_pose.copy()
                if armed and armed_deadline is not None and t_now > armed_deadline:
                    print(
                        "[IndyPositionalController] command watchdog expired; "
                        "disarming and holding the measured TCP."
                    )
                    armed = False
                    pose_interp = PoseTrajectoryInterpolator(
                        times=[t_now], poses=[actual_pose]
                    )
                    last_target_t = t_now
                if armed:
                    pose_cmd = pose_interp(t_now)
                    try:
                        self._send_task_pose(
                            cmd_fn,
                            cmd_name,
                            pose_cmd,
                            prefer_mm_deg=needs_teleop
                        )
                    except Exception as exc:
                        print(f"[IndyPositionalController] command failed: {exc}")
                        break
                actual_q, actual_qd = self._extract_joint_rad(control_data)
                state = {
                    "ActualTCPPose": actual_pose,
                    "TargetTCPPose": pose_cmd,
                    "ActualQ": actual_q,
                    "ActualQd": actual_qd,
                    "robot_receive_timestamp": time.time(),
                    "robot_timestamp": time.time() - self.receive_latency,
                }
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    cmd = int(commands["cmd"][i])
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    if cmd == Command.HOLD.value:
                        armed = False
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[t_now], poses=[actual_pose]
                        )
                        last_target_t = t_now
                        armed_deadline = None
                        continue
                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        if not armed:
                            # Start motion only after first explicit waypoint.
                            armed = True
                            self._last_sent_pose_m_rad = actual_pose.copy()
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[t_now],
                                poses=[actual_pose]
                            )
                            last_target_t = t_now
                        target_pose = np.asarray(commands["target_pose"][i], dtype=np.float64)
                        target_time = float(commands["target_time"][i])
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=t_now,
                            last_waypoint_time=last_target_t,
                        )
                        last_target_t = target_time
                        armed_deadline = max(t_now, target_time) + self.command_timeout_s

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                precise_wait(t_end=t_start + dt * iter_idx, time_func=time.monotonic)
        finally:
            self.ready_event.clear()
            try:
                if hasattr(indy, "stop_teleop"):
                    indy.stop_teleop()
            except Exception:
                pass
            try:
                if hasattr(indy, "stop_motion"):
                    indy.stop_motion()
            except Exception:
                pass
