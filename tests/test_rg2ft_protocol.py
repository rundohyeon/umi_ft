import unittest

import numpy as np

from umi.real_world.rg2ft_protocol import (
    FT_STATUS_COUNT,
    RG2FTModbusClient,
    meters_to_width_reg,
    parse_ft_status,
    read_ft_status_full,
    to_int16,
    width_to_meters,
    write_command,
)


class _Response:
    def __init__(self, registers=None, error=False):
        self.registers = [] if registers is None else registers
        self._error = error

    def isError(self):
        return self._error


class _Pymodbus314Client:
    def __init__(self, host, *, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.registers = [0] * FT_STATUS_COUNT
        self.read_device_ids = []
        self.writes = []
        self.closed = False

    def connect(self):
        return True

    def close(self):
        self.closed = True

    def read_holding_registers(self, address, *, count=1, device_id=1):
        self.read_device_ids.append(device_id)
        return _Response(self.registers[:count])

    def write_register(self, address, value, *, device_id=1):
        self.writes.append((address, value, device_id))
        return _Response()


class RG2FTProtocolTest(unittest.TestCase):
    def test_units_and_clamping(self):
        self.assertEqual(meters_to_width_reg(-1.0, 1000), 0)
        self.assertEqual(meters_to_width_reg(0.05, 1000), 500)
        self.assertEqual(meters_to_width_reg(0.1, 1000), 1000)
        self.assertEqual(meters_to_width_reg(0.2, 1000), 1000)
        self.assertAlmostEqual(width_to_meters(500), 0.05)

    def test_signed_registers_and_status(self):
        regs = [0] * FT_STATUS_COUNT
        regs[2] = 0xFFF6  # -10 -> -1 N
        regs[3] = 20      # +20 -> +2 N
        regs[5] = 25      # +25 -> +0.25 Nm
        regs[23] = 0xFFFA  # -6 -> -0.6 mm at fully closed
        regs[24] = 1
        regs[25] = 2
        ft, width_reg, busy, grip_det, sta = parse_ft_status(regs)
        self.assertEqual(to_int16(0xFFFA), -6)
        np.testing.assert_allclose(ft[:4], [-1.0, 2.0, 0.0, 0.25])
        self.assertEqual(width_reg, -6)
        self.assertEqual((busy, grip_det, sta), (1, 2, 0))

        regs[17] = 1
        self.assertEqual(parse_ft_status(regs)[-1], 1)

    def test_pymodbus_314_device_id_and_commands(self):
        client = RG2FTModbusClient(
            "192.168.2.1",
            slave_id=65,
            client_factory=_Pymodbus314Client,
        )
        client.connect()
        ft, width_reg, busy, grip_det, sta = read_ft_status_full(client)
        self.assertEqual(client._client.read_device_ids, [65])
        self.assertEqual(width_reg, 0)
        self.assertEqual(ft.shape, (12,))
        self.assertEqual((busy, grip_det, sta), (0, 0, 0))

        write_command(
            client,
            out_zero=0,
            r_gfr=200,
            r_gwd=500,
            r_ctr=1,
        )
        self.assertEqual(
            client._client.writes,
            [(0, 0, 65), (2, 200, 65), (3, 500, 65), (4, 1, 65)],
        )
        client.close()


if __name__ == "__main__":
    unittest.main()
