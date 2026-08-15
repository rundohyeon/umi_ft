"""Modbus/TCP helpers for the OnRobot RG2-FT gripper.

The register layout and command semantics match the RG2-FT recorder used for
the 2026-07-24 data collection.  The small client wrapper deliberately hides
pymodbus API differences (``unit``, ``slave`` and ``device_id``) from the
real-time controller.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np


FT_STATUS_ADDRESS = 257
FT_STATUS_COUNT = 26
PROX_OFFSET_ADDRESS = 5

GRIPPER_SPECS = {
    "rg2ft": {"max_force": 400, "max_width": 1000},
    "rg6": {"max_force": 1200, "max_width": 1600},
}

# r2 indices for [fx_l,fy_l,fz_l,tx_l,ty_l,tz_l, fx_r,fy_r,fz_r,tx_r,ty_r,tz_r]
FT_REGISTER_INDICES = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]
FT_FORCE_MASK = np.array([True, True, True, False, False, False] * 2, dtype=bool)
WIDTH_REG_PER_METER = 10000.0  # register is 0.1 mm


def to_int16(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value > 0x7FFF else value


def width_to_meters(grip_width_reg: int | float) -> float:
    return float(grip_width_reg) / WIDTH_REG_PER_METER


def meters_to_width_reg(pos_m: float, max_width: int) -> int:
    reg = int(round(float(pos_m) * WIDTH_REG_PER_METER))
    return max(0, min(int(max_width), reg))


def _device_kwargs(method: Callable, slave_id: int) -> dict[str, int]:
    """Select the unit-id keyword supported by this pymodbus method."""
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        params = {}
    for key in ("device_id", "slave", "unit"):
        if key in params:
            return {key: int(slave_id)}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"device_id": int(slave_id)}
    raise RuntimeError(
        "Unsupported pymodbus API: no device_id/slave/unit keyword found"
    )


def _import_modbus_client():
    try:
        from pymodbus.client import ModbusTcpClient

        return ModbusTcpClient
    except ImportError:
        try:
            from pymodbus.client.sync import ModbusTcpClient

            return ModbusTcpClient
        except ImportError as exc:
            raise ModuleNotFoundError(
                "RG2-FT control requires pymodbus. Install the version listed "
                "in requirements_rg2ft.txt."
            ) from exc


class RG2FTModbusClient:
    """Version-compatible pymodbus client with checked responses."""

    def __init__(
        self,
        hostname: str,
        *,
        port: int = 502,
        slave_id: int = 65,
        timeout: float = 1.0,
        client_factory=None,
    ):
        self.hostname = str(hostname)
        self.port = int(port)
        self.slave_id = int(slave_id)
        self.timeout = float(timeout)
        self._client_factory = client_factory
        self._client = None
        self._read_kwargs = None
        self._write_kwargs = None

    def connect(self) -> None:
        self.close()
        factory = self._client_factory or _import_modbus_client()
        self._client = factory(self.hostname, port=self.port, timeout=self.timeout)
        connected = self._client.connect()
        if connected is False:
            self.close()
            raise ConnectionError(
                f"could not connect to RG2-FT at {self.hostname}:{self.port}"
            )
        self._read_kwargs = _device_kwargs(
            self._client.read_holding_registers, self.slave_id
        )
        self._write_kwargs = _device_kwargs(
            self._client.write_register, self.slave_id
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._read_kwargs = None
        self._write_kwargs = None

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        if self._client is None:
            raise ConnectionError("RG2-FT Modbus client is not connected")
        response = self._client.read_holding_registers(
            address=int(address), count=int(count), **self._read_kwargs
        )
        if response is None or (
            hasattr(response, "isError") and response.isError()
        ):
            raise IOError(f"Modbus read error @{address}: {response}")
        registers = list(getattr(response, "registers", []))
        if len(registers) != int(count):
            raise IOError(
                f"Modbus read @{address} returned {len(registers)} registers; "
                f"expected {count}"
            )
        return registers

    def write_register(self, address: int, value: int) -> None:
        if self._client is None:
            raise ConnectionError("RG2-FT Modbus client is not connected")
        response = self._client.write_register(
            address=int(address),
            value=int(value) & 0xFFFF,
            **self._write_kwargs,
        )
        if response is None or (
            hasattr(response, "isError") and response.isError()
        ):
            raise IOError(f"Modbus write error @{address}: {response}")


def parse_ft_status(registers):
    """Parse status registers into F/T, signed width, busy, grip and g_sta."""
    r2 = list(registers)
    if len(r2) < FT_STATUS_COUNT:
        raise ValueError(
            f"RG2-FT status requires {FT_STATUS_COUNT} registers, got {len(r2)}"
        )
    raw = np.array(
        [to_int16(r2[i]) for i in FT_REGISTER_INDICES], dtype=np.float64
    )
    ft = np.where(FT_FORCE_MASK, raw / 10.0, raw / 100.0)
    width_reg = to_int16(r2[23])
    busy = int(r2[24])
    grip_det = int(r2[25])
    # Same gate used by the working collection recorder / OnRobot ROS driver.
    sta = 0 if (r2[0] == 0 and r2[9] == 0 and r2[17] == 0 and r2[20] == 0) else 1
    return ft, width_reg, busy, grip_det, sta


def read_ft_status_full(client: RG2FTModbusClient):
    registers = client.read_holding_registers(
        address=FT_STATUS_ADDRESS, count=FT_STATUS_COUNT
    )
    return parse_ft_status(registers)


def read_ft_status(client: RG2FTModbusClient, slave_id=None):
    """Backward-compatible four-value status API."""
    ft, width_reg, busy, grip_det, _ = read_ft_status_full(client)
    return ft, width_reg, busy, grip_det


def write_command(
    client: RG2FTModbusClient,
    slave_id=None,
    *,
    out_zero: int,
    r_gfr: int,
    r_gwd: int,
    r_ctr: int,
) -> None:
    client.write_register(address=0, value=out_zero)
    client.write_register(address=2, value=r_gfr)
    client.write_register(address=3, value=r_gwd)
    client.write_register(address=4, value=r_ctr)


def set_proximity_offsets(
    client: RG2FTModbusClient, slave_id=None, offsets=(230, 170)
) -> None:
    client.write_register(address=PROX_OFFSET_ADDRESS, value=offsets[0])
    client.write_register(address=PROX_OFFSET_ADDRESS + 1, value=offsets[1])


# Compatibility name for older code that imported this symbol from the module.
ModbusTcpClient = RG2FTModbusClient
