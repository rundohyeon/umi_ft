#!/usr/bin/env python3
"""Neuromeka IndyDCP3 motion examples and waypoint runner.

Requires:
    pip install neuromeka pyyaml

Usage:
    # Run predefined waypoints from YAML (robot + optional Dynamixel gripper)
    python example_neuromeka_movej_movel.py run --waypoints waypoints/example_indy.yaml

    # Built-in API demos
    python example_neuromeka_movej_movel.py demo --ip 192.168.1.10
    python example_neuromeka_movej_movel.py demo --demo movej_abs
"""

from __future__ import annotations

import copy
import json
import math
import socket
import sys
import threading
import time
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import click

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from umi.real_world.dynamixel_controller import (  # noqa: E402
        PROTOCOL_2_0,
        DynamixelConfig,
        DynamixelPositionController,
    )
    _HAS_DYNAMIXEL = True
except ImportError:
    _HAS_DYNAMIXEL = False

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "pyyaml is required. Install with: pip install pyyaml"
    ) from exc

try:
    from neuromeka import IndyDCP3, JointBaseType, TaskBaseType
    _HAS_NEUROMEKA = True
except ImportError:
    IndyDCP3 = Any  # type: ignore[misc, assignment]
    JointBaseType = None  # type: ignore[assignment]
    TaskBaseType = None  # type: ignore[assignment]
    _HAS_NEUROMEKA = False

OP_SYSTEM_ON = 1
OP_IDLE = 5
OP_MOVING = 6
OP_TEACHING = 7
OP_COLLISION = 8
OP_STOP_AND_OFF = 9
OP_TELE_OP = 17
# Fault / safety states (includes servo OFF)
OP_RECOVER_STATES = {2, 3, 4, 8, 9, 15, 16}
SERVO_OFF_STATES = {0, 9}

OP_STATE_NAMES = {
    0: "SYSTEM_OFF",
    1: "SYSTEM_ON",
    2: "VIOLATE",
    3: "RECOVER_HARD",
    4: "RECOVER_SOFT",
    5: "IDLE",
    6: "MOVING",
    7: "TEACHING",
    8: "COLLISION",
    9: "STOP_AND_OFF",
    11: "BRAKE_CONTROL",
    15: "VIOLATE_HARD",
    16: "MANUAL_RECOVER",
    17: "TELE_OP",
}

ALL_DEMOS = ("movej_abs", "movej_rel", "movel_abs", "movel_rel")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WAYPOINTS = SCRIPT_DIR / "waypoints" / "example_indy.yaml"

# IndyDCP3 gRPC ports (index=0). get_control_data() uses rtde -> 20004.
DCP_PORTS = {
    "control": 20001,
    "device": 20002,
    "config": 20003,
    "rtde": 20004,
}

if _HAS_NEUROMEKA:
    JOINT_BASE = {
        "absolute": JointBaseType.ABSOLUTE,
        "abs": JointBaseType.ABSOLUTE,
        "relative": JointBaseType.RELATIVE,
        "rel": JointBaseType.RELATIVE,
    }
    TASK_BASE = {
        "absolute": TaskBaseType.ABSOLUTE,
        "abs": TaskBaseType.ABSOLUTE,
        "relative": TaskBaseType.RELATIVE,
        "rel": TaskBaseType.RELATIVE,
        "tcp": TaskBaseType.TCP,
    }
else:
    JOINT_BASE = {}
    TASK_BASE = {}


@dataclass
class GripperConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    dxl_id: int = 1
    open_position: int = 1000
    close_position: int = 0
    profile_velocity: int = 30
    profile_acceleration: int = 15
    current_limit: float | None = None
    pwm_limit: float | None = None
    keep_torque: bool = True
    move_timeout_s: float = 30.0
    move_tolerance: int = 15
    finish_mode: str = "goal"
    finish_time_s: float = 1.0


def _list_serial_ports() -> list[str]:
    return sorted(glob("/dev/ttyUSB*") + glob("/dev/ttyACM*"))


def _format_missing_gripper_port(port: str) -> str:
    found = _list_serial_ports()
    lines = [
        f"Dynamixel gripper port not found: {port}",
        "",
        "Checks:",
        "  1. USB cable plugged into the U2D2 / USB-serial adapter",
        "  2. Inside Docker: attach serial without full rebuild:",
        "       bash docker/container.sh attach-serial",
        "     or recreate: bash docker/container.sh recreate",
        "  3. List ports: ls -l /dev/ttyUSB* /dev/ttyACM*",
    ]
    if found:
        lines.append(f"  Available now: {', '.join(found)}")
        lines.append(f"  Retry with: --gripper-port {found[0]}")
    else:
        lines.append("  No /dev/ttyUSB* or /dev/ttyACM* found on this machine.")
    lines.append(
        "  Arm-only test (skip gripper waypoints): add --no-gripper"
    )
    return "\n".join(lines)


class DynamixelGripperSession:
    """Thin wrapper around DynamixelPositionController for waypoint sequences."""

    def __init__(self, cfg: GripperConfig):
        if not _HAS_DYNAMIXEL:
            raise RuntimeError(
                "dynamixel-sdk / umi dynamixel_controller not available. "
                "Install with: pip install dynamixel-sdk"
            )
        self.cfg = cfg
        self._controller: DynamixelPositionController | None = None

    def __enter__(self) -> "DynamixelGripperSession":
        if not Path(self.cfg.port).exists():
            raise FileNotFoundError(_format_missing_gripper_port(self.cfg.port))
        dxl_cfg = DynamixelConfig(
            port=self.cfg.port,
            baudrate=self.cfg.baudrate,
            protocol_version=PROTOCOL_2_0,
            dxl_ids=(self.cfg.dxl_id,),
            profile_velocity=self.cfg.profile_velocity,
            profile_acceleration=self.cfg.profile_acceleration,
            current_limit=self.cfg.current_limit,
            pwm_limit=self.cfg.pwm_limit,
        )
        self._controller = DynamixelPositionController(dxl_cfg)
        self._controller._disable_torque_on_exit = not self.cfg.keep_torque
        self._controller.__enter__()
        self._controller.configure_position_mode()
        self._controller.enable_torque()
        pos = self._controller.get_present_position(self.cfg.dxl_id)
        print(
            f"Gripper connected on {self.cfg.port} id={self.cfg.dxl_id} "
            f"(present={pos}, open={self.cfg.open_position}, close={self.cfg.close_position}"
            f"{f', pwm_limit={self.cfg.pwm_limit}' if self.cfg.pwm_limit is not None else ''}"
            f"{f', current_limit={self.cfg.current_limit}' if self.cfg.current_limit is not None else ''})"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._controller is not None:
            self._controller.__exit__(exc_type, exc, tb)
            self._controller = None

    def resolve_target(self, target: Any) -> int:
        if isinstance(target, str):
            key = target.strip().lower()
            if key == "open":
                return int(self.cfg.open_position)
            if key == "close":
                return int(self.cfg.close_position)
            if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
                return int(key)
            raise ValueError(
                f"Unknown gripper target '{target}'. Use open, close, or an integer tick value."
            )
        return int(target)

    def move_to(
        self,
        target: Any,
        *,
        wait: bool = True,
        optional: bool = False,
        timeout_s: float | None = None,
        tolerance: int | None = None,
        finish_mode: str | None = None,
        finish_time_s: float | None = None,
        current_limit: float | None = None,
        pwm_limit: float | None = None,
    ) -> int:
        if self._controller is None:
            raise RuntimeError("Gripper session is not connected")
        goal = self.resolve_target(target)
        timeout_s = self.cfg.move_timeout_s if timeout_s is None else float(timeout_s)
        tolerance = self.cfg.move_tolerance if tolerance is None else int(tolerance)
        finish_mode = self.cfg.finish_mode if finish_mode is None else str(finish_mode)
        finish_time_s = self.cfg.finish_time_s if finish_time_s is None else float(finish_time_s)
        if current_limit is not None or pwm_limit is not None:
            self._controller.set_grip_force_limit(
                current_limit=current_limit,
                pwm_limit=pwm_limit,
                reenable_torque=True,
            )

        before = self._controller.get_present_position(self.cfg.dxl_id)
        if finish_mode.strip().lower() == "goal" and abs(before - goal) <= tolerance:
            print(f"  gripper: already at goal={goal} (present={before}), skip")
            return before
        print(
            f"  gripper: moving {before} -> {goal} "
            f"(finish_mode={finish_mode}, timeout={timeout_s}s, tolerance={tolerance})"
        )
        try:
            self._controller.move_to(
                goal,
                wait=wait,
                timeout_s=timeout_s,
                tolerance=tolerance,
                finish_mode=finish_mode,
                finish_time_s=finish_time_s,
            )
        except TimeoutError as exc:
            after = self._controller.get_present_position(self.cfg.dxl_id)
            print(f"  gripper: WARNING timed out ({exc}); present={after}, goal={goal}")
            if not optional:
                raise
            return after
        after = self._controller.get_present_position(self.cfg.dxl_id)
        gap = abs(after - goal)
        if gap > tolerance:
            print(
                f"  gripper: WARNING stopped {gap} ticks from goal "
                f"(present={after}, goal={goal}). "
                f"For full close: finish_mode=goal, raise pwm_limit, or lower close_position."
            )
        print(f"  gripper: done present={after} (goal={goal})")
        return after


def _parse_gripper_config(
    data: dict[str, Any],
    *,
    cli_port: str | None,
    cli_id: int | None,
) -> GripperConfig | None:
    raw = data.get("gripper")
    if raw is None and cli_port is None and cli_id is None:
        return None
    raw = dict(raw or {})
    if cli_port is not None:
        raw["port"] = cli_port
    if cli_id is not None:
        raw["id"] = cli_id
    return GripperConfig(
        port=str(raw.get("port", "/dev/ttyUSB0")),
        baudrate=int(raw.get("baudrate", 57600)),
        dxl_id=int(raw.get("id", raw.get("dxl_id", 1))),
        open_position=int(raw.get("open_position", 1000)),
        close_position=int(raw.get("close_position", 0)),
        profile_velocity=int(raw.get("profile_velocity", 30)),
        profile_acceleration=int(raw.get("profile_acceleration", 15)),
        current_limit=(
            float(raw["current_limit"]) if raw.get("current_limit") is not None else None
        ),
        pwm_limit=(
            float(raw["pwm_limit"]) if raw.get("pwm_limit") is not None else None
        ),
        keep_torque=bool(raw.get("keep_torque", True)),
        move_timeout_s=float(raw.get("move_timeout_s", raw.get("timeout_s", 30.0))),
        move_tolerance=int(raw.get("move_tolerance", raw.get("tolerance", 15))),
        finish_mode=str(raw.get("finish_mode", "goal")),
        finish_time_s=float(raw.get("finish_time_s", 1.0)),
    )


def _sequence_uses_gripper(waypoints: list[Any]) -> bool:
    return any(
        isinstance(wp, dict) and str(wp.get("type", "")).lower() == "gripper"
        for wp in waypoints
    )


def _close_indy(indy: IndyDCP3 | None) -> None:
    if indy is None:
        return
    for attr in (
        "boot_channel",
        "control_channel",
        "device_channel",
        "config_channel",
        "rtde_channel",
        "cri_channel",
    ):
        channel = getattr(indy, attr, None)
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass


def _require_neuromeka() -> None:
    if not _HAS_NEUROMEKA:
        raise RuntimeError(
            "neuromeka package is required for the real robot/DCP backend. "
            "Install with: pip install neuromeka, or use run --sim for ROS2 Gazebo."
        )


def _connect_indy(ip: str, *, index: int = 0) -> IndyDCP3:
    _require_neuromeka()
    return IndyDCP3(ip, index=index)


def _tcp_probe(ip: str, port: int, *, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _format_probe_failure(ip: str, probe: dict[str, Any]) -> str:
    lines = [
        f"IndyDCP3 cannot talk to robot controller at {ip}.",
        "",
        "TCP ports:",
    ]
    for name, port in DCP_PORTS.items():
        ok = probe["ports"].get(name, False)
        lines.append(f"  {ip}:{port} ({name}): {'open' if ok else 'CLOSED'}")
    lines.extend(
        [
            "",
            "RPC:",
            f"  get_control_info ({DCP_PORTS['control']}): "
            f"{'OK' if probe['rpc'].get('get_control_info') else 'FAILED'}",
            f"  get_control_data ({DCP_PORTS['rtde']}): "
            f"{'OK' if probe['rpc'].get('get_control_data') else 'FAILED/TIMEOUT'}",
            "",
            "IndyDCP3() only creates a gRPC client — it does NOT verify the robot is ready.",
            "",
            "If port 20004 is open but get_control_data times out (common after servo OFF):",
            "  1. Teach pendant: clear alarm -> SERVO ON -> AUTO",
            "  2. Power-cycle the robot controller (or restart IndyStudio)",
            "  3. Wait 30-60s after boot, then: python3 ... status --ip ...",
            "",
            "If ports are CLOSED: check Ethernet cable and that this PC is on 192.168.1.x.",
        ]
    )
    return "\n".join(lines)


def _probe_controller(ip: str, *, verbose: bool = False) -> dict[str, Any]:
    probe: dict[str, Any] = {"ip": ip, "ports": {}, "rpc": {}}
    for name, port in DCP_PORTS.items():
        ok = _tcp_probe(ip, port)
        probe["ports"][name] = ok
        if verbose:
            print(f"  TCP {ip}:{port} ({name}) -> {'open' if ok else 'closed'}")

    indy = _connect_indy(ip)
    if hasattr(indy, "get_control_info"):
        info = _rpc_with_timeout(
            indy.get_control_info,
            timeout_s=5.0,
            label=f"get_control_info:{DCP_PORTS['control']}",
            default=None,
        )
        probe["rpc"]["get_control_info"] = info is not None
        if verbose:
            state = "OK" if info else "TIMEOUT"
            print(f"  RPC get_control_info (port {DCP_PORTS['control']}) -> {state}")

    data = _rpc_with_timeout(
        indy.get_control_data,
        timeout_s=8.0,
        label=f"get_control_data:{DCP_PORTS['rtde']}",
        default=None,
    )
    probe["rpc"]["get_control_data"] = data is not None
    if verbose:
        state = "OK" if data else "TIMEOUT"
        print(f"  RPC get_control_data (port {DCP_PORTS['rtde']}) -> {state}")
    if data:
        probe["op_state"] = int(data.get("op_state", -1))
        if verbose:
            op = probe["op_state"]
            print(f"  op_state={op} ({_op_state_hint(op)})")
    _close_indy(indy)
    return probe


def _rpc_with_timeout(
    fn,
    *,
    timeout_s: float = 8.0,
    label: str = "RPC",
    default: Any = None,
) -> Any:
    """Run a blocking IndyDCP3 call without hanging forever on a stuck controller."""
    result: list[Any] = [default]
    error: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = fn()
        except BaseException as exc:  # noqa: BLE001
            error[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        print(
            f"  WARNING: {label} timed out after {timeout_s:.0f}s "
            "(controller gRPC may be stuck). Check teach pendant: SERVO ON, AUTO, no fault."
        )
        return default
    if error[0] is not None:
        raise error[0]
    return result[0]


def _read_op_state(indy: IndyDCP3, *, ip: str | None = None) -> tuple[int, IndyDCP3]:
    data = _rpc_with_timeout(
        indy.get_control_data,
        timeout_s=8.0,
        label=f"get_control_data:{DCP_PORTS['rtde']}",
        default=None,
    )
    if data is None:
        if ip is not None:
            print("  recreating IndyDCP3 client (gRPC channel stale)...")
            _close_indy(indy)
            time.sleep(2.0)
            indy = _connect_indy(ip)
        return -1, indy
    return int(data.get("op_state", -1)), indy


def _op_state_hint(op_state: int) -> str:
    name = OP_STATE_NAMES.get(op_state, f"unknown({op_state})")
    hints = {
        OP_SYSTEM_ON: "select AUTO on teach pendant and wait for IDLE",
        OP_MOVING: "arm still moving; stop_motion or wait",
        OP_TEACHING: "exit TEACH mode on teach pendant",
        OP_TELE_OP: "another teleop client active; stop eval/ROS teleop or stop_teleop",
        OP_STOP_AND_OFF: "servo OFF after safety stop; clear alarm and SERVO ON",
        0: "servo OFF; turn SERVO ON on teach pendant",
    }
    extra = hints.get(op_state, "")
    if extra:
        return f"{name} — {extra}"
    return name


def _is_grpc_unavailable(exc: BaseException) -> bool:
    msg = str(exc)
    needles = (
        "UNAVAILABLE",
        "Connection reset by peer",
        "Connection refused",
        "failed to connect to all addresses",
        "InactiveRpcError",
    )
    return any(n in msg for n in needles)


def _reconnect_indy(ip: str) -> IndyDCP3:
    """DCP gRPC (port 20004) can drop during long movel; reconnect and wait IDLE."""
    print(f"  Reconnecting IndyDCP3 @ {ip} ...")
    time.sleep(3.0)
    return _ensure_ready(_connect_indy(ip), ip=ip)


def _get_control_data(indy: IndyDCP3, ip: str | None) -> tuple[dict, IndyDCP3]:
    try:
        return indy.get_control_data(), indy
    except Exception as exc:
        if ip is not None and _is_grpc_unavailable(exc):
            print(f"  gRPC lost during poll ({exc!r}); reconnecting...")
            indy = _reconnect_indy(ip)
            return indy.get_control_data(), indy
        raise


def _robot_status_line(indy: IndyDCP3, ip: str | None = None) -> str:
    data, _ = _get_control_data(indy, ip)
    op = int(data.get("op_state", -1))
    name = OP_STATE_NAMES.get(op, f"unknown({op})")
    return f"op_state={op} ({name}), tcp p (mm,deg)={data.get('p')}"


def _raise_robot_fault(indy: IndyDCP3, op_state: int, ip: str | None, *, context: str) -> None:
    line = _robot_status_line(indy, ip)
    if op_state in SERVO_OFF_STATES:
        raise RuntimeError(
            f"Robot servo OFF / safety stop during {context}.\n"
            f"  {line}\n"
            "  Teach pendant: clear alarm -> SERVO ON -> AUTO -> recover/reset if needed.\n"
            "  Likely causes: Cartesian workspace limit, collision, or bad linear path\n"
            "  (large base-relative movel after many moves). Use frame: tcp, smaller steps,\n"
            "  lower vel_ratio/acc_ratio, or insert movej to reorient."
        )
    raise RuntimeError(f"Robot fault during {context}: {line}")


def _wait_idle(indy: IndyDCP3, timeout_s: float = 60.0, *, ip: str | None = None) -> IndyDCP3:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data, indy = _get_control_data(indy, ip)
        op_state = int(data.get("op_state", -1))
        if op_state in SERVO_OFF_STATES:
            _raise_robot_fault(indy, op_state, ip, context="wait_idle")
        if op_state == OP_IDLE:
            return indy
        time.sleep(0.05)
    raise TimeoutError(f"Robot did not return to OP_IDLE within {timeout_s:.1f}s")


def _wait_motion(indy: IndyDCP3, timeout_s: float = 120.0, *, ip: str | None = None) -> IndyDCP3:
    """Wait until movej/movel finishes by polling op_state.

    Avoid indy.wait_traj() here: on long movel (e.g. 300 mm) the controller gRPC
    channel (192.168.1.10:20004) often resets while the arm still moves.
    """
    deadline = time.time() + timeout_s
    saw_moving = False
    start_t = time.time()
    while time.time() < deadline:
        data, indy = _get_control_data(indy, ip)
        op_state = int(data.get("op_state", -1))
        if op_state in OP_RECOVER_STATES:
            _raise_robot_fault(indy, op_state, ip, context="motion")
        if op_state == OP_MOVING:
            saw_moving = True
        if saw_moving and op_state == OP_IDLE:
            return indy
        if (not saw_moving) and (time.time() - start_t) > 3.0:
            raise RuntimeError(
                "Motion never started (stuck in IDLE). "
                "movel may have been rejected (wrong frame, unreachable target, "
                "or path blocked). "
                + _robot_status_line(indy, ip)
            )
        time.sleep(0.05)
    raise TimeoutError(
        f"Motion did not finish within {timeout_s:.1f}s; saw_moving={saw_moving}; "
        + _robot_status_line(indy, ip)
    )


def _ensure_ready(indy: IndyDCP3, *, ip: str, timeout_s: float = 45.0) -> IndyDCP3:
    """Match indy_driver / eval startup: exit teleop and wait for IDLE."""
    print("  Waiting for robot ready (op_state=5 IDLE)...")
    last_log = 0.0
    last_stop_teleop = 0.0
    last_stop_motion = 0.0
    consecutive_timeouts = 0

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        op_state, indy = _read_op_state(indy, ip=ip)
        now = time.time()

        if op_state == -1:
            consecutive_timeouts += 1
            if consecutive_timeouts >= 3:
                print("  DCP not responding after 3 attempts. Running connectivity probe...")
                probe = _probe_controller(ip, verbose=True)
                raise RuntimeError(_format_probe_failure(ip, probe))
            if now - last_log >= 2.0:
                print("  still waiting for get_control_data() on port 20004...")
                last_log = now
            time.sleep(0.5)
            continue

        consecutive_timeouts = 0

        if op_state in SERVO_OFF_STATES:
            _raise_robot_fault(indy, op_state, ip, context="startup")

        if op_state == OP_IDLE:
            print("  Robot ready (OP_IDLE).")
            return indy

        if op_state in OP_RECOVER_STATES and op_state not in SERVO_OFF_STATES:
            if hasattr(indy, "recover"):
                print(f"  op_state={op_state}, calling recover()...")
                _rpc_with_timeout(indy.recover, timeout_s=10.0, label="recover()")
                time.sleep(0.5)
                continue
            _raise_robot_fault(indy, op_state, ip, context="startup")

        if op_state == OP_MOVING and hasattr(indy, "stop_motion"):
            if now - last_stop_motion >= 1.0:
                print("  op_state=MOVING, sending stop_motion()...")
                _rpc_with_timeout(indy.stop_motion, timeout_s=5.0, label="stop_motion()")
                last_stop_motion = now
                time.sleep(0.2)
                continue

        if op_state in (OP_TELE_OP, OP_TEACHING) and hasattr(indy, "stop_teleop"):
            if now - last_stop_teleop >= 1.0:
                print(f"  op_state={_op_state_hint(op_state)}, sending stop_teleop()...")
                _rpc_with_timeout(indy.stop_teleop, timeout_s=5.0, label="stop_teleop()")
                last_stop_teleop = now
                time.sleep(0.3)
                continue

        if now - last_log >= 2.0:
            print(f"  op_state={op_state} ({_op_state_hint(op_state)})")
            last_log = now
        time.sleep(0.1)

    op_state, _ = _read_op_state(indy, ip=ip)
    raise RuntimeError(
        f"Robot not ready after {timeout_s:.0f}s (op_state={op_state}, "
        f"wanted {OP_IDLE} IDLE). {_op_state_hint(op_state)}"
    )


def _dof(indy: IndyDCP3) -> int:
    q = indy.get_control_data().get("q", [])
    return len(q)


def _current_joints_deg(indy: IndyDCP3) -> list[float]:
    return [float(v) for v in indy.get_control_data().get("q", [])]


def _pad_joints(
    values: list[float],
    dof: int,
    *,
    fill_from: list[float] | None = None,
) -> list[float]:
    out = [float(v) for v in values[:dof]]
    if len(out) < dof:
        filler = fill_from or [0.0] * dof
        for i in range(len(out), dof):
            out.append(float(filler[i]) if i < len(filler) else 0.0)
    return out


def _pad_task(values: list[float]) -> list[float]:
    out = [float(v) for v in values[:6]]
    if len(out) < 6:
        out.extend([0.0] * (6 - len(out)))
    return out


def _print_state(indy: IndyDCP3, label: str) -> None:
    data = _rpc_with_timeout(
        indy.get_control_data,
        timeout_s=8.0,
        label="get_control_data",
        default=None,
    )
    if data is None:
        print(f"{label}")
        print("  (get_control_data timed out)")
        return
    print(f"{label}")
    print(f"  op_state: {data.get('op_state')}")
    print(f"  joints q (deg): {data.get('q')}")
    print(f"  tcp p (mm, deg): {data.get('p')}")


def _check_motion_response(response: Any, action: str) -> None:
    if isinstance(response, dict):
        if response.get("success") is False:
            raise RuntimeError(f"{action} rejected: {response}")
        if "error" in response and response["error"]:
            raise RuntimeError(f"{action} error: {response['error']}")


def _movej(
    indy: IndyDCP3,
    jtarget: list[float],
    *,
    base_type,
    vel_ratio: int,
    acc_ratio: int,
    ip: str | None = None,
    motion_timeout_s: float | None = None,
) -> IndyDCP3:
    if len(jtarget) != _dof(indy):
        raise ValueError(
            f"movej target has {len(jtarget)} joints but robot has {_dof(indy)} DOF: {jtarget}"
        )
    response = indy.movej(
        jtarget=jtarget,
        base_type=base_type,
        vel_ratio=vel_ratio,
        acc_ratio=acc_ratio,
    )
    _check_motion_response(response, "movej")
    timeout = motion_timeout_s if motion_timeout_s is not None else 120.0
    return _wait_motion(indy, timeout_s=timeout, ip=ip)


def _movel(
    indy: IndyDCP3,
    ttarget: list[float],
    *,
    base_type,
    vel_ratio: int,
    acc_ratio: int,
    bypass_singular: bool = False,
    ip: str | None = None,
    motion_timeout_s: float | None = None,
) -> IndyDCP3:
    kwargs: dict[str, Any] = dict(
        ttarget=ttarget,
        base_type=base_type,
        vel_ratio=vel_ratio,
        acc_ratio=acc_ratio,
    )
    # Neuromeka API: joy/jog often tolerates paths that strict movel rejects.
    try:
        import inspect

        if "bypass_singular" in inspect.signature(indy.movel).parameters:
            kwargs["bypass_singular"] = bypass_singular
    except (TypeError, ValueError):
        pass
    response = indy.movel(**kwargs)
    _check_motion_response(response, "movel")
    if motion_timeout_s is None:
        # Scale wait with Cartesian step size (mm); large moves need longer + gRPC may drop.
        step_mm = math.sqrt(sum(float(v) ** 2 for v in ttarget[:3]))
        motion_timeout_s = max(120.0, 30.0 + step_mm * 0.5)
    return _wait_motion(indy, timeout_s=motion_timeout_s, ip=ip)


def _resolve_joint_target(indy: IndyDCP3, target: Any) -> list[float]:
    dof = _dof(indy)
    current_q = _current_joints_deg(indy)
    if target in ("home", "HOME"):
        home = indy.get_home_pos().get("jpos", current_q)
        return _pad_joints(home, dof, fill_from=current_q)
    if not isinstance(target, (list, tuple)):
        raise ValueError(f"movej target must be a list or 'home', got {target!r}")
    return _pad_joints([float(v) for v in target], dof, fill_from=current_q)


def _resolve_task_target(target: Any) -> list[float]:
    if not isinstance(target, (list, tuple)):
        raise ValueError(f"movel target must be a list, got {target!r}")
    return _pad_task([float(v) for v in target])


def _load_waypoint_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Waypoint file must contain a mapping at top level: {path}")
    if "waypoints" not in data:
        raise ValueError(f"Waypoint file missing 'waypoints' list: {path}")
    if not isinstance(data["waypoints"], list) or not data["waypoints"]:
        raise ValueError(f"'waypoints' must be a non-empty list: {path}")
    return data


def _execute_waypoint(
    indy: IndyDCP3,
    wp: dict[str, Any],
    *,
    default_vel_ratio: int,
    default_acc_ratio: int,
    default_dwell_s: float,
    gripper: DynamixelGripperSession | None = None,
    robot_ip: str | None = None,
) -> IndyDCP3:
    name = str(wp.get("name", "unnamed"))
    move_type = str(wp.get("type", "")).lower()
    frame = str(wp.get("frame", "absolute")).lower()
    target = wp.get("target")
    vel_ratio = int(wp.get("vel_ratio", default_vel_ratio))
    acc_ratio = int(wp.get("acc_ratio", default_acc_ratio))
    dwell_s = float(wp.get("dwell_s", default_dwell_s))

    print(f"\n=== waypoint: {name} ===")
    print(f"  type={move_type} vel_ratio={vel_ratio} acc_ratio={acc_ratio}")

    if move_type == "gripper":
        if gripper is None:
            print(f"  gripper target: {target!r} (skipped, --no-gripper)")
            return indy
        print(f"  gripper target: {target!r}")
        gripper.move_to(
            target,
            wait=bool(wp.get("wait", True)),
            optional=bool(wp.get("optional", False)),
            timeout_s=wp.get("timeout_s", wp.get("move_timeout_s")),
            tolerance=wp.get("tolerance", wp.get("move_tolerance")),
            finish_mode=wp.get("finish_mode"),
            finish_time_s=wp.get("finish_time_s"),
            current_limit=wp.get("current_limit"),
            pwm_limit=wp.get("pwm_limit"),
        )
    elif move_type == "movej":
        print(f"  frame={frame}")
        jtarget = _resolve_joint_target(indy, target)
        base_type = JOINT_BASE.get(frame)
        if base_type is None:
            raise ValueError(f"Unknown movej frame '{frame}' in waypoint '{name}'")
        print(f"  jtarget (deg): {jtarget}")
        indy = _movej(
            indy,
            jtarget,
            base_type=base_type,
            vel_ratio=vel_ratio,
            acc_ratio=acc_ratio,
            ip=robot_ip,
        )
        _print_state(indy, "  after:")
    elif move_type == "movel":
        print(f"  frame={frame}")
        ttarget = _resolve_task_target(target)
        base_type = TASK_BASE.get(frame)
        if base_type is None:
            raise ValueError(f"Unknown movel frame '{frame}' in waypoint '{name}'")
        print(f"  ttarget (mm, deg): {ttarget}")
        data, _ = _get_control_data(indy, robot_ip)
        print(f"  tcp before (mm, deg): {data.get('p')}")
        step_mm = math.sqrt(sum(float(v) ** 2 for v in ttarget[:3]))
        if step_mm > 120.0:
            print(
                f"  warning: large movel step ({step_mm:.0f} mm); "
                "may hit workspace limit / trigger servo OFF"
            )
        bypass_singular = bool(wp.get("bypass_singular", step_mm > 80.0))
        if bypass_singular:
            print("  bypass_singular=True")
        indy = _movel(
            indy,
            ttarget,
            base_type=base_type,
            vel_ratio=vel_ratio,
            acc_ratio=acc_ratio,
            bypass_singular=bypass_singular,
            ip=robot_ip,
            motion_timeout_s=wp.get("motion_timeout_s"),
        )
        _print_state(indy, "  after:")
    else:
        raise ValueError(
            f"Unknown waypoint type '{move_type}' in '{name}' "
            "(use movej, movel, or gripper)"
        )

    if dwell_s > 0:
        time.sleep(dwell_s)
    return indy


def run_waypoints(
    indy: IndyDCP3,
    waypoint_file: Path,
    *,
    vel_ratio: int,
    acc_ratio: int,
    dwell_s: float,
    gripper_cfg: GripperConfig | None,
    robot_ip: str | None = None,
) -> None:
    data = _load_waypoint_file(waypoint_file)
    defaults = data.get("defaults", {}) or {}
    default_vel_ratio = int(defaults.get("vel_ratio", vel_ratio))
    default_acc_ratio = int(defaults.get("acc_ratio", acc_ratio))
    default_dwell_s = float(defaults.get("dwell_s", dwell_s))
    waypoints = data["waypoints"]

    needs_gripper = _sequence_uses_gripper(waypoints)
    if needs_gripper and gripper_cfg is None:
        print("Gripper waypoints present; running arm only (--no-gripper).")

    print(f"Loaded {len(waypoints)} waypoint(s) from {waypoint_file}")

    gripper_ctx = (
        DynamixelGripperSession(gripper_cfg)
        if gripper_cfg is not None and needs_gripper
        else None
    )

    if gripper_ctx is not None:
        gripper_ctx.__enter__()
    try:
        for wp in waypoints:
            if not isinstance(wp, dict):
                raise ValueError(f"Each waypoint must be a mapping, got {wp!r}")
            indy = _execute_waypoint(
                indy,
                wp,
                default_vel_ratio=default_vel_ratio,
                default_acc_ratio=default_acc_ratio,
                default_dwell_s=default_dwell_s,
                gripper=gripper_ctx,
                robot_ip=robot_ip,
            )
    finally:
        if gripper_ctx is not None:
            gripper_ctx.__exit__(None, None, None)


def _infer_dof_from_waypoints(data: dict[str, Any], default: int = 7) -> int:
    for wp in data.get("waypoints", []):
        if not isinstance(wp, dict):
            continue
        if str(wp.get("type", "")).lower() != "movej":
            continue
        target = wp.get("target")
        if isinstance(target, (list, tuple)) and target:
            return len(target)
    return default


def _parse_joint_names(joint_names_csv: str | None, *, default_dof: int) -> list[str]:
    if joint_names_csv:
        joint_names = [name.strip() for name in joint_names_csv.split(",") if name.strip()]
        if not joint_names:
            raise ValueError("--joint-names must contain at least one joint name")
        return joint_names
    return [f"joint{i}" for i in range(default_dof)]


class Ros2JointTrajectorySession:
    """ROS2 joint trajectory action client for indy_gazebo simulation."""

    def __init__(
        self,
        *,
        action_name: str,
        joint_names: list[str],
        action_timeout_s: float,
        joint_states_topic: str = "/joint_states",
    ):
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 simulation mode requires rclpy, control_msgs, sensor_msgs, "
                "and trajectory_msgs. Source your ROS2/colcon setup first."
            ) from exc

        self.rclpy = rclpy
        self.Duration = Duration
        self.FollowJointTrajectory = FollowJointTrajectory
        self.JointTrajectoryPoint = JointTrajectoryPoint
        self.action_name = action_name
        self.joint_names = joint_names
        self.action_timeout_s = float(action_timeout_s)
        self.current_positions: dict[str, float] = {}
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self.node = rclpy.create_node("indy_waypoint_sim_runner")
        self.action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            action_name,
        )
        self.subscription = self.node.create_subscription(
            JointState,
            joint_states_topic,
            self._on_joint_state,
            10,
        )

    def __enter__(self) -> "Ros2JointTrajectorySession":
        print(f"ROS2 simulation action: {self.action_name}")
        print(f"ROS2 joints: {', '.join(self.joint_names)}")
        if not self.action_client.wait_for_server(timeout_sec=self.action_timeout_s):
            raise RuntimeError(
                f"Action server not available: {self.action_name}. "
                "Check that indy_gazebo spawned joint_trajectory_controller."
            )
        self.wait_for_joint_states(timeout_s=3.0, required=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()

    def _on_joint_state(self, msg) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.current_positions[str(name)] = float(pos)

    def wait_for_joint_states(
        self,
        *,
        timeout_s: float,
        required: bool = True,
    ) -> list[float] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if all(name in self.current_positions for name in self.joint_names):
                return [self.current_positions[name] for name in self.joint_names]
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if required:
            missing = [
                name for name in self.joint_names if name not in self.current_positions
            ]
            raise RuntimeError(
                "No complete /joint_states sample for simulation joints. "
                f"Missing: {missing}"
            )
        return None

    def _duration_msg(self, seconds: float):
        seconds = max(float(seconds), 0.1)
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return self.Duration(sec=sec, nanosec=nanosec)

    def send_joint_goal(self, positions_rad: list[float], *, duration_s: float) -> None:
        if len(positions_rad) != len(self.joint_names):
            raise ValueError(
                f"Target has {len(positions_rad)} joints but --joint-names has "
                f"{len(self.joint_names)}: {self.joint_names}"
            )

        goal_msg = self.FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = list(self.joint_names)
        point = self.JointTrajectoryPoint()
        point.positions = [float(v) for v in positions_rad]
        point.velocities = [0.0] * len(positions_rad)
        point.time_from_start = self._duration_msg(duration_s)
        goal_msg.trajectory.points = [point]

        send_future = self.action_client.send_goal_async(goal_msg)
        self.rclpy.spin_until_future_complete(
            self.node,
            send_future,
            timeout_sec=self.action_timeout_s,
        )
        if not send_future.done():
            raise TimeoutError(f"Timed out sending goal to {self.action_name}")
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            raise RuntimeError(f"Trajectory goal rejected by {self.action_name}")

        result_future = goal_handle.get_result_async()
        self.rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=max(self.action_timeout_s, duration_s + self.action_timeout_s),
        )
        if not result_future.done():
            raise TimeoutError(f"Timed out waiting for {self.action_name} result")
        result = result_future.result().result
        if int(result.error_code) != 0:
            detail = result.error_string or f"error_code={result.error_code}"
            raise RuntimeError(f"Trajectory failed: {detail}")


def _resolve_ros2_joint_target(
    sim: Ros2JointTrajectorySession,
    target: Any,
    *,
    frame: str,
) -> list[float]:
    dof = len(sim.joint_names)
    frame = frame.lower()
    if target in ("home", "HOME"):
        target_rad = [0.0] * dof
    elif isinstance(target, (list, tuple)):
        target_rad = [math.radians(float(v)) for v in target]
    else:
        raise ValueError(f"movej target must be a list or 'home', got {target!r}")

    if len(target_rad) > dof:
        raise ValueError(
            f"movej target has {len(target_rad)} joints but simulation expects {dof}"
        )

    if frame in ("absolute", "abs"):
        if len(target_rad) == dof:
            return target_rad
        current = sim.wait_for_joint_states(timeout_s=3.0)
        return target_rad + current[len(target_rad):]

    if frame in ("relative", "rel"):
        current = sim.wait_for_joint_states(timeout_s=3.0)
        delta = target_rad + [0.0] * (dof - len(target_rad))
        return [q + dq for q, dq in zip(current, delta)]

    raise ValueError(f"ROS2 simulation movej supports absolute/relative, got {frame!r}")


def run_waypoints_ros2(
    waypoint_file: Path,
    *,
    action_name: str,
    joint_names_csv: str | None,
    duration_s: float,
    action_timeout_s: float,
    dwell_s: float,
) -> None:
    data = _load_waypoint_file(waypoint_file)
    defaults = data.get("defaults", {}) or {}
    default_dwell_s = float(defaults.get("dwell_s", dwell_s))
    waypoints = data["waypoints"]
    joint_names = _parse_joint_names(
        joint_names_csv,
        default_dof=_infer_dof_from_waypoints(data),
    )

    print(f"Loaded {len(waypoints)} waypoint(s) from {waypoint_file}")
    print("Simulation mode: gripper waypoints are skipped.")
    with Ros2JointTrajectorySession(
        action_name=action_name,
        joint_names=joint_names,
        action_timeout_s=action_timeout_s,
    ) as sim:
        for wp in waypoints:
            if not isinstance(wp, dict):
                raise ValueError(f"Each waypoint must be a mapping, got {wp!r}")

            name = str(wp.get("name", "unnamed"))
            move_type = str(wp.get("type", "")).lower()
            frame = str(wp.get("frame", "absolute")).lower()
            target = wp.get("target")
            wp_dwell_s = float(wp.get("dwell_s", default_dwell_s))
            wp_duration_s = float(wp.get("duration_s", duration_s))

            print(f"\n=== waypoint: {name} ===")
            if move_type == "gripper":
                print(f"  gripper target: {target!r} (skipped in simulation)")
            elif move_type == "movej":
                target_rad = _resolve_ros2_joint_target(sim, target, frame=frame)
                print(f"  frame={frame}")
                print(f"  jtarget (deg): {[round(math.degrees(v), 4) for v in target_rad]}")
                print(f"  duration_s={wp_duration_s}")
                sim.send_joint_goal(target_rad, duration_s=wp_duration_s)
                current = sim.wait_for_joint_states(timeout_s=1.0, required=False)
                if current is not None:
                    print(
                        "  after joints (deg): "
                        f"{[round(math.degrees(v), 4) for v in current]}"
                    )
            elif move_type == "movel":
                raise NotImplementedError(
                    "ROS2 simulation mode currently supports movej waypoints only. "
                    "For movel, launch MoveIt/IK and convert the Cartesian target to joints."
                )
            else:
                raise ValueError(
                    f"Unknown waypoint type '{move_type}' in '{name}' "
                    "(use movej, movel, or gripper)"
                )

            if wp_dwell_s > 0:
                time.sleep(wp_dwell_s)


def demo_movej_absolute(indy: IndyDCP3, vel_ratio: int, acc_ratio: int) -> None:
    target = _resolve_joint_target(indy, "home")
    print("\n=== movej ABSOLUTE -> home jpos ===")
    print(f"  target (deg): {target}")
    _movej(indy, target, base_type=JointBaseType.ABSOLUTE, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
    _print_state(indy, "  after movej_abs:")


def demo_movej_relative(
    indy: IndyDCP3,
    vel_ratio: int,
    acc_ratio: int,
    joint_rel_deg: float,
) -> None:
    dof = _dof(indy)
    delta = [0.0] * dof
    delta[1] = float(joint_rel_deg)
    target = _pad_joints(delta, dof)
    print("\n=== movej RELATIVE -> offset joint[1] ===")
    print(f"  delta (deg): {target}")
    _movej(indy, target, base_type=JointBaseType.RELATIVE, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
    _print_state(indy, "  after movej_rel:")


def demo_movel_absolute(indy: IndyDCP3, vel_ratio: int, acc_ratio: int) -> None:
    current = copy.deepcopy(indy.get_control_data()["p"])
    target = copy.deepcopy(current)
    target[2] += 20.0
    print("\n=== movel ABSOLUTE -> current TCP with +20 mm Z ===")
    print(f"  from (mm, deg): {current}")
    print(f"  to   (mm, deg): {target}")
    _movel(indy, target, base_type=TaskBaseType.ABSOLUTE, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
    _print_state(indy, "  after movel_abs:")


def demo_movel_relative(
    indy: IndyDCP3,
    vel_ratio: int,
    acc_ratio: int,
    task_rel_mm: float,
) -> None:
    delta = [float(task_rel_mm), 0.0, 0.0, 0.0, 0.0, 0.0]
    print("\n=== movel RELATIVE -> +X in reference frame ===")
    print(f"  delta (mm, deg): {delta}")
    _movel(indy, delta, base_type=TaskBaseType.RELATIVE, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
    _print_state(indy, "  after movel_rel:")


def _connect_and_prepare(ip: str, *, skip_ready: bool = False) -> IndyDCP3:
    print(f"Probing IndyDCP3 @ {ip} ...")
    probe = _probe_controller(ip, verbose=True)
    if not probe["rpc"].get("get_control_data"):
        raise RuntimeError(_format_probe_failure(ip, probe))

    indy = _connect_indy(ip)
    op_state = probe.get("op_state", "?")
    print(f"DCP link OK (op_state={op_state})")
    if not skip_ready:
        indy = _ensure_ready(indy, ip=ip)
    _print_state(indy, "Initial state:")
    return indy


def _handle_motion_errors(indy: IndyDCP3) -> None:
    try:
        if hasattr(indy, "stop_motion"):
            indy.stop_motion()
    except Exception:
        pass


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli() -> None:
    """Neuromeka movej/movel examples and waypoint runner."""


@cli.command("run")
@click.option("--ip", default="192.168.1.10", show_default=True, help="Robot controller IP")
@click.option(
    "--sim",
    is_flag=True,
    help="Use ROS2/Gazebo joint_trajectory_controller instead of IndyDCP3.",
)
@click.option(
    "--ros-action",
    default="/joint_trajectory_controller/follow_joint_trajectory",
    show_default=True,
    help="ROS2 FollowJointTrajectory action for --sim.",
)
@click.option(
    "--joint-names",
    default=None,
    help="Comma-separated simulation joint names. Default is inferred as joint0..jointN.",
)
@click.option(
    "--sim-duration-s",
    default=4.0,
    show_default=True,
    type=float,
    help="Trajectory duration for each movej waypoint in --sim.",
)
@click.option(
    "--ros-timeout-s",
    default=10.0,
    show_default=True,
    type=float,
    help="Timeout for ROS2 action server/goal/result in --sim.",
)
@click.option(
    "--waypoints",
    "waypoint_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_WAYPOINTS,
    show_default=True,
    help="YAML/JSON file with predefined waypoint sequence",
)
@click.option("--vel-ratio", default=20, show_default=True, type=int)
@click.option("--acc-ratio", default=50, show_default=True, type=int)
@click.option("--dwell-s", default=0.3, show_default=True, type=float, help="Pause after each waypoint")
@click.option("--gripper-port", default=None, help="Dynamixel serial port (overrides YAML gripper.port)")
@click.option("--gripper-id", default=None, type=int, help="Dynamixel motor ID (overrides YAML gripper.id)")
@click.option("--no-gripper", is_flag=True, help="Ignore gripper waypoints even if present in YAML")
@click.option(
    "--skip-ready",
    is_flag=True,
    help="Skip waiting for OP_IDLE (debug only; motion may fail if robot not ready)",
)
def run_cmd(
    ip: str,
    sim: bool,
    ros_action: str,
    joint_names: str | None,
    sim_duration_s: float,
    ros_timeout_s: float,
    waypoint_file: Path,
    vel_ratio: int,
    acc_ratio: int,
    dwell_s: float,
    gripper_port: str | None,
    gripper_id: int | None,
    no_gripper: bool,
    skip_ready: bool,
) -> None:
    """Execute predefined waypoints from a YAML/JSON file."""
    data = _load_waypoint_file(waypoint_file)

    if sim:
        try:
            run_waypoints_ros2(
                waypoint_file,
                action_name=ros_action,
                joint_names_csv=joint_names,
                duration_s=sim_duration_s,
                action_timeout_s=ros_timeout_s,
                dwell_s=dwell_s,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            raise SystemExit(130) from None
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print("\nDone.")
        return

    gripper_cfg = None if no_gripper else _parse_gripper_config(
        data, cli_port=gripper_port, cli_id=gripper_id
    )

    indy = _connect_and_prepare(ip, skip_ready=skip_ready)
    try:
        run_waypoints(
            indy,
            waypoint_file,
            vel_ratio=vel_ratio,
            acc_ratio=acc_ratio,
            dwell_s=dwell_s,
            gripper_cfg=gripper_cfg,
            robot_ip=ip,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        _handle_motion_errors(indy)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        _handle_motion_errors(indy)
        raise SystemExit(1) from exc
    print("\nDone.")


@cli.command("status")
@click.option("--ip", default="192.168.1.10", show_default=True, help="Robot controller IP")
def status_cmd(ip: str) -> None:
    """Probe DCP ports and print robot op_state (connectivity check)."""
    print(f"Probing IndyDCP3 @ {ip} ...")
    probe = _probe_controller(ip, verbose=True)
    if not probe["rpc"].get("get_control_data"):
        print(_format_probe_failure(ip, probe), file=sys.stderr)
        raise SystemExit(1)
    op_state = int(probe.get("op_state", -1))
    if op_state in SERVO_OFF_STATES:
        print("Robot DCP responds but servo is OFF. Clear alarm and SERVO ON on teach pendant.")
        raise SystemExit(1)
    if op_state != OP_IDLE:
        print(f"Note: waypoint runner waits for op_state={OP_IDLE} (IDLE) before moving.")


@cli.command("demo")
@click.option("--ip", default="192.168.1.10", show_default=True, help="Robot controller IP")
@click.option("--vel-ratio", default=20, show_default=True, type=int)
@click.option("--acc-ratio", default=50, show_default=True, type=int)
@click.option("--joint-rel-deg", default=5.0, show_default=True, type=float)
@click.option("--task-rel-mm", default=30.0, show_default=True, type=float)
@click.option(
    "--demo",
    "demos",
    multiple=True,
    type=click.Choice(ALL_DEMOS),
    help="Run only selected demo(s). Default: all four.",
)
def demo_cmd(
    ip: str,
    vel_ratio: int,
    acc_ratio: int,
    joint_rel_deg: float,
    task_rel_mm: float,
    demos: tuple[str, ...],
) -> None:
    """Run built-in movej/movel API demos."""
    selected = demos or ALL_DEMOS
    indy = _connect_and_prepare(ip)

    try:
        for name in selected:
            if name == "movej_abs":
                demo_movej_absolute(indy, vel_ratio, acc_ratio)
            elif name == "movej_rel":
                demo_movej_relative(indy, vel_ratio, acc_ratio, joint_rel_deg)
            elif name == "movel_abs":
                demo_movel_absolute(indy, vel_ratio, acc_ratio)
            elif name == "movel_rel":
                demo_movel_relative(indy, vel_ratio, acc_ratio, task_rel_mm)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        _handle_motion_errors(indy)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        _handle_motion_errors(indy)
        raise SystemExit(1) from exc
    print("\nDone.")


def main() -> None:
    # Backward compat: `python ... --ip ...` -> demo mode
    if len(sys.argv) > 1 and sys.argv[1] not in ("run", "demo", "-h", "--help"):
        sys.argv.insert(1, "demo")
    cli()


if __name__ == "__main__":
    main()
