import unittest
import time

from scripts.example_neuromeka_movej_movel_rg2 import (
    RG2GripperConfig,
    RG2GripperSession,
    _parse_rg2_config,
)


class _FakeDirectRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_width = 0.05
        self.last_sample_time = 0.0
        self.calls = []

    def start(self, wait=True, timeout=5.0):
        self.last_sample_time = time.time()

    def stop(self):
        self.calls.append(("stop",))

    def open_gripper(self, force_n=None):
        self.calls.append(("open", force_n))

    def close_gripper(self, force_n=None):
        self.calls.append(("close", force_n))

    def set_width(self, width_m, force_n=None):
        self.calls.append(("width", width_m, force_n))


class RG2WaypointRunnerConfigTest(unittest.TestCase):
    def test_open_close_and_numeric_targets_are_metres(self):
        session = RG2GripperSession(RG2GripperConfig())
        self.assertEqual(session.resolve_target("open"), 0.1)
        self.assertEqual(session.resolve_target("close"), 0.0)
        self.assertEqual(session.resolve_target(0.042), 0.042)
        with self.assertRaises(ValueError):
            session.resolve_target(0.101)

    def test_rg2_yaml_config_and_cli_override(self):
        cfg = _parse_rg2_config(
            {
                "waypoints": [{"type": "gripper", "target": "open"}],
                "gripper": {"host": "192.168.2.1", "force_n": 20.0},
            },
            cli_host="10.0.0.5",
            cli_port=1502,
            cli_slave_id=66,
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.host, "10.0.0.5")
        self.assertEqual(cfg.port, 1502)
        self.assertEqual(cfg.slave_id, 66)

    def test_dynamixel_yaml_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Dynamixel"):
            _parse_rg2_config(
                {
                    "waypoints": [{"type": "gripper", "target": "open"}],
                    "gripper": {"baudrate": 57600, "open_position": 1700},
                },
                cli_host=None,
                cli_port=None,
                cli_slave_id=None,
            )

    def test_waypoints_send_final_width_directly_without_interpolation(self):
        session = RG2GripperSession(
            RG2GripperConfig(force_n=20.0),
            recorder_factory=_FakeDirectRecorder,
        )
        with session:
            recorder = session._recorder
            session.move_to("close", wait=False)
            session.move_to(0.042, wait=False)
            session.move_to("open", wait=False)

        self.assertEqual(
            recorder.calls,
            [
                ("close", 20.0),
                ("width", 0.042, 20.0),
                ("open", 20.0),
                ("stop",),
            ],
        )

    def test_custom_open_width_sends_that_width_instead_of_physical_max(self):
        session = RG2GripperSession(
            RG2GripperConfig(open_width_m=0.03, force_n=20.0),
            recorder_factory=_FakeDirectRecorder,
        )
        with session:
            recorder = session._recorder
            session.move_to("open", wait=False)

        self.assertEqual(
            recorder.calls,
            [
                ("width", 0.03, 20.0),
                ("stop",),
            ],
        )


if __name__ == "__main__":
    unittest.main()
