import time
import unittest
from unittest import mock
import sys
import types

import numpy as np

# The minimal CI shell used for this hardware-free test does not include the
# optional atomics package.  Controller.run() below uses fake queues/ring
# buffers, so the shared-memory atomic implementation is never instantiated.
try:
    import atomics  # noqa: F401
except ModuleNotFoundError:
    atomics_stub = types.ModuleType("atomics")
    atomics_stub.atomicview = None
    atomics_stub.MemoryOrder = object
    atomics_stub.UINT = object
    sys.modules["atomics"] = atomics_stub

import umi.real_world.rg2ft_controller as controller_module
from umi.real_world.rg2ft_controller import (
    Command,
    RG2FTController,
    _pop_due_direct_waypoint,
    _schedule_direct_waypoint,
)


class _ReadyEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class _ErrorQueue:
    def put_nowait(self, value):
        raise AssertionError(f"unexpected startup error: {value}")


class _RingBuffer:
    def __init__(self):
        self.states = []

    def put(self, state):
        self.states.append(state)


class _InputQueue:
    def __init__(self, batches):
        self.batches = list(batches)

    def get_all(self):
        if not self.batches:
            raise controller_module.Empty()
        return self.batches.pop(0)


class _FakeModbusClient:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.registers = [0] * 26
        self.registers[23] = 500
        self.writes = []
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def read_holding_registers(self, address, count):
        return self.registers[:count]

    def write_register(self, address, value):
        self.writes.append((address, value))


def _controller_for_run(input_batches):
    # run() itself does not depend on multiprocessing.Process internals, so a
    # lightweight instance keeps this test deterministic and hardware-free.
    controller = object.__new__(RG2FTController)
    controller.gripper_type = "rg2ft"
    controller.hostname = "fake-rg2"
    controller.port = 502
    controller.slave_id = 65
    controller.frequency = 100.0
    controller.home_to_open = False
    controller.move_max_speed = 0.2
    controller.force_n = 20.0
    controller.open_tolerance = 0.005
    controller.reconnect_interval = 0.01
    controller.receive_latency = 0.01
    controller.verbose = False
    controller.input_queue = _InputQueue(input_batches)
    controller.ring_buffer = _RingBuffer()
    controller.ready_event = _ReadyEvent()
    controller.startup_error_queue = _ErrorQueue()
    return controller


class RG2FTControllerSafetyTest(unittest.TestCase):
    def test_startup_then_shutdown_sends_no_motion_registers(self):
        shutdown = {
            "cmd": np.array([Command.SHUTDOWN.value]),
            "target_pos": np.array([0.0]),
            "target_time": np.array([0.0]),
        }
        controller = _controller_for_run([shutdown])
        with mock.patch.object(
            controller_module, "RG2FTModbusClient", _FakeModbusClient
        ), mock.patch.object(controller_module, "precise_wait", lambda **_: None):
            controller.run()

        self.assertTrue(controller.ready_event.was_set)
        self.assertEqual(_FakeModbusClient.last_instance.writes, [])
        self.assertGreaterEqual(len(controller.ring_buffer.states), 1)
        self.assertAlmostEqual(
            controller.ring_buffer.states[0]["gripper_position"], 0.05
        )

    def test_scheduled_width_enables_register_command(self):
        schedule = {
            "cmd": np.array([Command.SCHEDULE_WAYPOINT.value]),
            "target_pos": np.array([0.04]),
            "target_time": np.array([time.time() - 0.01]),
        }
        shutdown = {
            "cmd": np.array([Command.SHUTDOWN.value]),
            "target_pos": np.array([0.0]),
            "target_time": np.array([0.0]),
        }
        controller = _controller_for_run([schedule, shutdown])
        with mock.patch.object(
            controller_module, "RG2FTModbusClient", _FakeModbusClient
        ), mock.patch.object(controller_module, "precise_wait", lambda **_: None):
            controller.run()

        addresses = [address for address, _ in _FakeModbusClient.last_instance.writes]
        self.assertEqual(addresses, [0, 2, 3, 4])
        self.assertEqual(_FakeModbusClient.last_instance.writes[1], (2, 200))
        self.assertEqual(_FakeModbusClient.last_instance.writes[2], (3, 400))

    def test_future_width_is_not_sent_before_its_deadline(self):
        schedule = {
            "cmd": np.array([Command.SCHEDULE_WAYPOINT.value]),
            "target_pos": np.array([0.04]),
            "target_time": np.array([time.time() + 10.0]),
        }
        shutdown = {
            "cmd": np.array([Command.SHUTDOWN.value]),
            "target_pos": np.array([0.0]),
            "target_time": np.array([0.0]),
        }
        controller = _controller_for_run([schedule, shutdown])
        with mock.patch.object(
            controller_module, "RG2FTModbusClient", _FakeModbusClient
        ), mock.patch.object(controller_module, "precise_wait", lambda **_: None):
            controller.run()

        self.assertEqual(_FakeModbusClient.last_instance.writes, [])

    def test_direct_waypoint_queue_preserves_order_and_replaces_future_plan(self):
        pending = []
        pending = _schedule_direct_waypoint(
            pending, target_time=10.0, target_pos=0.02
        )
        pending = _schedule_direct_waypoint(
            pending, target_time=20.0, target_pos=0.04
        )
        pending, due = _pop_due_direct_waypoint(pending, now=9.0)
        self.assertIsNone(due)

        pending, due = _pop_due_direct_waypoint(pending, now=10.0)
        self.assertEqual(due, 0.02)
        self.assertEqual(pending, [(20.0, 0.04)])

        pending = _schedule_direct_waypoint(
            pending, target_time=15.0, target_pos=0.03
        )
        self.assertEqual(pending, [(15.0, 0.03)])


if __name__ == "__main__":
    unittest.main()
