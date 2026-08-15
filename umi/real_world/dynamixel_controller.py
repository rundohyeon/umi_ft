"""Dynamixel position control via Robotis Dynamixel SDK.

Install dependency:
    pip install dynamixel-sdk
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "dynamixel-sdk is required. Install with: pip install dynamixel-sdk"
    ) from exc


PROTOCOL_1_0 = 1.0
PROTOCOL_2_0 = 2.0

# Control table addresses (Protocol 2.0, X-series)
ADDR_TORQUE_ENABLE_P2 = 64
ADDR_OPERATING_MODE_P2 = 11
ADDR_PWM_LIMIT_P2 = 36
ADDR_CURRENT_LIMIT_P2 = 38
ADDR_PROFILE_ACCELERATION_P2 = 108
ADDR_PROFILE_VELOCITY_P2 = 112
ADDR_GOAL_POSITION_P2 = 116
ADDR_PRESENT_POSITION_P2 = 132
ADDR_MOVING_P2 = 122

# Control table addresses (Protocol 1.0, AX/MX series)
ADDR_TORQUE_ENABLE_P1 = 24
ADDR_CW_ANGLE_LIMIT_P1 = 6
ADDR_CCW_ANGLE_LIMIT_P1 = 8
ADDR_GOAL_POSITION_P1 = 30
ADDR_MOVING_P1 = 46
ADDR_PRESENT_POSITION_P1 = 36

OPERATING_MODE_POSITION = 3
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# XL430-W250 has no Current Limit register at addr 38; use PWM Limit instead.
XL430_MODEL_NUMBERS = frozenset({1060})
PWM_LIMIT_MAX = 885


@dataclass
class DynamixelConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    protocol_version: float = PROTOCOL_2_0
    dxl_ids: Sequence[int] = (1,)
    profile_velocity: int = 30
    profile_acceleration: int = 15
    current_limit: float | None = None
    pwm_limit: float | None = None


class DynamixelPositionController:
    """Simple position controller for one or more Dynamixel servos."""

    def __init__(self, config: DynamixelConfig | None = None):
        self.config = config or DynamixelConfig()
        self._port = PortHandler(self.config.port)
        self._packet = PacketHandler(self.config.protocol_version)
        self._connected = False
        self._disable_torque_on_exit = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if not self._port.openPort():
            raise RuntimeError(f"Failed to open port: {self.config.port}")
        if not self._port.setBaudRate(self.config.baudrate):
            raise RuntimeError(
                f"Failed to set baudrate {self.config.baudrate} on {self.config.port}"
            )
        self._connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        if self._connected:
            if disable_torque:
                self.disable_torque()
            self._port.closePort()
            self._connected = False

    def __enter__(self) -> "DynamixelPositionController":
        self.connect()
        self._disable_torque_on_exit = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect(disable_torque=self._disable_torque_on_exit)

    def _check(self, dxl_id: int, result: int, error: int, action: str) -> None:
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"{action} failed for id={dxl_id}: "
                f"{self._packet.getTxRxResult(result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"{action} failed for id={dxl_id}: "
                f"{self._packet.getRxPacketError(error)}"
            )

    def _addr(self, p2_addr: int, p1_addr: int) -> int:
        if self.config.protocol_version >= PROTOCOL_2_0:
            return p2_addr
        return p1_addr

    def enable_torque(self, dxl_ids: Iterable[int] | None = None) -> None:
        self._write_byte(
            dxl_ids or self.config.dxl_ids,
            self._addr(ADDR_TORQUE_ENABLE_P2, ADDR_TORQUE_ENABLE_P1),
            TORQUE_ENABLE,
            action="enable_torque",
        )

    def disable_torque(self, dxl_ids: Iterable[int] | None = None) -> None:
        self._write_byte(
            dxl_ids or self.config.dxl_ids,
            self._addr(ADDR_TORQUE_ENABLE_P2, ADDR_TORQUE_ENABLE_P1),
            TORQUE_DISABLE,
            action="disable_torque",
        )

    def configure_position_mode(self, dxl_ids: Iterable[int] | None = None) -> None:
        ids = list(dxl_ids or self.config.dxl_ids)
        if self.config.protocol_version >= PROTOCOL_2_0:
            needs_mode_write = any(
                self._read_byte(dxl_id, ADDR_OPERATING_MODE_P2, "get_mode")
                != OPERATING_MODE_POSITION
                for dxl_id in ids
            )
            if needs_mode_write:
                # Operating mode is not writable while torque is enabled.
                self.disable_torque(ids)
                self._write_byte(
                    ids, ADDR_OPERATING_MODE_P2, OPERATING_MODE_POSITION, "set_mode"
                )
            self._write_dword(
                ids,
                ADDR_PROFILE_ACCELERATION_P2,
                self.config.profile_acceleration,
                "set_profile_acceleration",
            )
            self._write_dword(
                ids,
                ADDR_PROFILE_VELOCITY_P2,
                self.config.profile_velocity,
                "set_profile_velocity",
            )
            if self.config.pwm_limit is not None or self.config.current_limit is not None:
                try:
                    self.set_grip_force_limit(
                        pwm_limit=self.config.pwm_limit,
                        current_limit=self.config.current_limit,
                        dxl_ids=ids,
                    )
                except RuntimeError as exc:
                    print(f"[WARN] grip force limit not applied: {exc}")
        else:
            # Protocol 1.0 position mode is default; widen joint limits.
            self.disable_torque(ids)
            self._write_word(ids, ADDR_CW_ANGLE_LIMIT_P1, 0, "set_cw_limit")
            self._write_word(ids, ADDR_CCW_ANGLE_LIMIT_P1, 1023, "set_ccw_limit")

    def uses_pwm_limit(self, dxl_id: int) -> bool:
        model = self._try_get_model_number(dxl_id)
        return model in XL430_MODEL_NUMBERS if model is not None else False

    def get_pwm_limit(self, dxl_id: int) -> int | None:
        if self.config.protocol_version < PROTOCOL_2_0:
            return None
        try:
            value, result, error = self._packet.read2ByteTxRx(
                self._port, dxl_id, ADDR_PWM_LIMIT_P2
            )
            self._check(dxl_id, result, error, "get_pwm_limit")
            return int(value)
        except RuntimeError:
            return None

    def get_current_limit(self, dxl_id: int) -> int | None:
        """Read Current Limit (P2 addr 38). Returns None on unsupported motors."""
        if self.config.protocol_version < PROTOCOL_2_0:
            return None
        try:
            value, result, error = self._packet.read2ByteTxRx(
                self._port, dxl_id, ADDR_CURRENT_LIMIT_P2
            )
            self._check(dxl_id, result, error, "get_current_limit")
            return int(value)
        except RuntimeError:
            return None

    def set_pwm_limit(
        self,
        pwm_limit: int | float,
        dxl_ids: Iterable[int] | None = None,
        *,
        reenable_torque: bool = False,
    ) -> None:
        """Set max PWM output (grip force on XL430). Range 0-885."""
        if self.config.protocol_version < PROTOCOL_2_0:
            return
        ids = list(dxl_ids or self.config.dxl_ids)
        value = self._normalize_pwm_limit(pwm_limit)
        self.disable_torque(ids)
        self._write_word(ids, ADDR_PWM_LIMIT_P2, value, "set_pwm_limit")
        if reenable_torque:
            self.enable_torque(ids)

    def set_current_limit(
        self,
        current_limit: int | float,
        dxl_ids: Iterable[int] | None = None,
        *,
        reenable_torque: bool = False,
    ) -> None:
        """Set max motor current (XM/XC series). P2 unit ~2.69 mA at addr 38."""
        if self.config.protocol_version < PROTOCOL_2_0:
            return
        ids = list(dxl_ids or self.config.dxl_ids)
        if ids and self.uses_pwm_limit(ids[0]):
            raise RuntimeError(
                f"Model {self._try_get_model_number(ids[0])} (XL430) has no Current Limit "
                "register at addr 38. Use pwm_limit instead (addr 36, range 0-885)."
            )
        value = self._normalize_current_limit(current_limit)
        self.disable_torque(ids)
        self._write_word(ids, ADDR_CURRENT_LIMIT_P2, value, "set_current_limit")
        if reenable_torque:
            self.enable_torque(ids)

    def set_grip_force_limit(
        self,
        *,
        pwm_limit: int | float | None = None,
        current_limit: int | float | None = None,
        dxl_ids: Iterable[int] | None = None,
        reenable_torque: bool = False,
    ) -> None:
        """Apply grip-force limit using the register appropriate for the motor model."""
        ids = list(dxl_ids or self.config.dxl_ids)
        if not ids:
            return
        if self.uses_pwm_limit(ids[0]):
            if pwm_limit is not None:
                self.set_pwm_limit(pwm_limit, ids, reenable_torque=reenable_torque)
                return
            if current_limit is not None:
                model = self._try_get_model_number(ids[0])
                print(
                    f"[WARN] Model {model} (XL430) ignores current_limit; "
                    "set pwm_limit in YAML or use --pwm-limit."
                )
                return
        elif current_limit is not None:
            self.set_current_limit(current_limit, ids, reenable_torque=reenable_torque)
        elif pwm_limit is not None:
            self.set_pwm_limit(pwm_limit, ids, reenable_torque=reenable_torque)

    def _try_get_model_number(self, dxl_id: int) -> int | None:
        try:
            value, result, error = self._packet.read2ByteTxRx(self._port, dxl_id, 0)
            if result != COMM_SUCCESS or error != 0:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_pwm_limit(pwm_limit: int | float) -> int:
        """Normalize PWM limit. Values <=1.0 are treated as fraction of max (885)."""
        value = float(pwm_limit)
        if value <= 0:
            return 0
        if value <= 1.0:
            return int(round(value * PWM_LIMIT_MAX))
        return int(round(min(value, PWM_LIMIT_MAX)))

    @staticmethod
    def _normalize_current_limit(current_limit: int | float) -> int:
        """Normalize user value to raw register units.

        - If value is < 10, treat as amps (e.g. 0.3 A).
        - Otherwise treat as raw register value.
        """
        value = float(current_limit)
        if value < 0:
            return 0
        if value < 10.0:
            # X-series datasheets commonly use about 2.69 mA per unit.
            return int(round((value * 1000.0) / 2.69))
        return int(round(value))

    def set_goal_position(self, dxl_id: int, position: int) -> None:
        addr = self._addr(ADDR_GOAL_POSITION_P2, ADDR_GOAL_POSITION_P1)
        if self.config.protocol_version >= PROTOCOL_2_0:
            result, error = self._packet.write4ByteTxRx(
                self._port, dxl_id, addr, int(position)
            )
        else:
            result, error = self._packet.write2ByteTxRx(
                self._port, dxl_id, addr, int(position)
            )
        self._check(dxl_id, result, error, "set_goal_position")

    def set_goal_positions(self, positions: dict[int, int]) -> None:
        for dxl_id, position in positions.items():
            self.set_goal_position(dxl_id, position)

    def get_present_position(self, dxl_id: int) -> int:
        addr = self._addr(ADDR_PRESENT_POSITION_P2, ADDR_PRESENT_POSITION_P1)
        if self.config.protocol_version >= PROTOCOL_2_0:
            position, result, error = self._packet.read4ByteTxRx(self._port, dxl_id, addr)
        else:
            position, result, error = self._packet.read2ByteTxRx(self._port, dxl_id, addr)
        self._check(dxl_id, result, error, "get_present_position")
        return int(position)

    def get_present_positions(self, dxl_ids: Iterable[int] | None = None) -> dict[int, int]:
        ids = list(dxl_ids or self.config.dxl_ids)
        return {dxl_id: self.get_present_position(dxl_id) for dxl_id in ids}

    def is_moving(self, dxl_id: int) -> bool:
        addr = self._addr(ADDR_MOVING_P2, ADDR_MOVING_P1)
        moving, result, error = self._packet.read1ByteTxRx(self._port, dxl_id, addr)
        self._check(dxl_id, result, error, "is_moving")
        return bool(moving)

    def wait_until_stopped(
        self,
        dxl_ids: Iterable[int] | None = None,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
        min_run_time_s: float = 0.0,
    ) -> None:
        ids = list(dxl_ids or self.config.dxl_ids)
        start_time = time.time()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ran_long_enough = (time.time() - start_time) >= max(0.0, min_run_time_s)
            if ran_long_enough and not any(self.is_moving(dxl_id) for dxl_id in ids):
                return
            time.sleep(poll_interval_s)
        raise TimeoutError(f"Motors still moving after {timeout_s:.1f}s: {ids}")

    def wait_until_goal(
        self,
        goal: int | dict[int, int],
        dxl_ids: Iterable[int] | None = None,
        tolerance: int = 15,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.05,
    ) -> None:
        ids = list(dxl_ids or self.config.dxl_ids)
        if isinstance(goal, dict):
            goals = {dxl_id: int(goal[dxl_id]) for dxl_id in ids}
        else:
            goals = {dxl_id: int(goal) for dxl_id in ids}

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            positions = self.get_present_positions(ids)
            at_goal = all(
                abs(positions[dxl_id] - goals[dxl_id]) <= tolerance for dxl_id in ids
            )
            moving = any(self.is_moving(dxl_id) for dxl_id in ids)
            if at_goal and not moving:
                return
            time.sleep(poll_interval_s)

        positions = self.get_present_positions(ids)
        details = ", ".join(
            f"id={dxl_id} present={positions[dxl_id]} goal={goals[dxl_id]}"
            for dxl_id in ids
        )
        raise TimeoutError(
            f"Motors did not reach goal within {timeout_s:.1f}s: {details}"
        )

    def move_to(
        self,
        position: int | dict[int, int],
        wait: bool = True,
        timeout_s: float = 30.0,
        tolerance: int = 15,
        finish_mode: str = "goal",
        finish_time_s: float = 0.0,
    ) -> None:
        if isinstance(position, dict):
            self.set_goal_positions(position)
            target_ids = list(position.keys())
            goal = position
        else:
            target_ids = list(self.config.dxl_ids)
            goal = int(position)
            self.set_goal_positions({dxl_id: goal for dxl_id in target_ids})
        if not wait:
            return

        mode = finish_mode.strip().lower()
        if mode == "time":
            time.sleep(max(0.0, finish_time_s))
            return
        if mode == "stopped":
            self.wait_until_stopped(
                target_ids,
                timeout_s=timeout_s or finish_time_s or 30.0,
                min_run_time_s=finish_time_s,
            )
            return
        self.wait_until_goal(
            goal,
            dxl_ids=target_ids,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )

    def _read_byte(self, dxl_id: int, addr: int, action: str) -> int:
        value, result, error = self._packet.read1ByteTxRx(self._port, dxl_id, addr)
        self._check(dxl_id, result, error, action)
        return int(value)

    def _write_byte(self, dxl_ids: Iterable[int], addr: int, value: int, action: str) -> None:
        for dxl_id in dxl_ids:
            result, error = self._packet.write1ByteTxRx(self._port, dxl_id, addr, value)
            self._check(dxl_id, result, error, action)

    def _write_word(self, dxl_ids: Iterable[int], addr: int, value: int, action: str) -> None:
        for dxl_id in dxl_ids:
            result, error = self._packet.write2ByteTxRx(self._port, dxl_id, addr, value)
            self._check(dxl_id, result, error, action)

    def _write_dword(self, dxl_ids: Iterable[int], addr: int, value: int, action: str) -> None:
        for dxl_id in dxl_ids:
            result, error = self._packet.write4ByteTxRx(self._port, dxl_id, addr, value)
            self._check(dxl_id, result, error, action)
