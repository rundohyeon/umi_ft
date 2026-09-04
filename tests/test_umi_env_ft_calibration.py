import numpy as np

from umi.real_world.umi_env import UmiEnv


class _StaticGripper:
    def __init__(self, left, right):
        self._left = np.asarray(left, dtype=np.float64)
        self._right = np.asarray(right, dtype=np.float64)

    def get_state(self):
        return {
            "gripper_ft_left": self._left,
            "gripper_ft_right": self._right,
            "gripper_timestamp": 1.0,
        }


def test_latest_ft_feedback_uses_the_same_tare_then_residual_order_as_policy():
    software_tare = np.asarray(
        [5.8, 2.6, 146.7, 0.36, 0.11, 0.07, 1.6, -5.3, -137.2, 0.43, 0.27, -0.12],
        dtype=np.float64,
    )
    residual = np.asarray(
        [0.02, -0.01, 0.03, 0.001, 0.0, -0.002, -0.03, 0.02, -0.04, 0.002, 0.0, 0.001],
        dtype=np.float64,
    )
    signal = np.asarray(
        [0.08, 0.01, -0.04, 0.002, 0.0, 0.001, -0.05, 0.03, 0.12, -0.001, 0.0, 0.002],
        dtype=np.float64,
    )
    raw = software_tare + residual + signal

    env = object.__new__(UmiEnv)
    env.gripper = _StaticGripper(raw[:6], raw[6:])
    env.rg2ft_ft_offset = software_tare
    env.ft_startup_bias_12d = residual
    result = env.get_latest_ft_state()

    np.testing.assert_allclose(result["left"], signal[:6])
    np.testing.assert_allclose(result["right"], signal[6:])
    np.testing.assert_allclose(
        np.r_[result["left_after_software_tare"], result["right_after_software_tare"]],
        residual + signal,
    )
