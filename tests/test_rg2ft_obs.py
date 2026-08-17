import unittest

import numpy as np

from umi.real_world.rg2ft_obs import causal_ft_history, prepare_rg2ft_policy_obs


class RG2FTPolicyObsTest(unittest.TestCase):
    def test_causal_history_never_selects_future_ft(self):
        timestamps = np.array([1.01, 1.02, 1.03, 1.04])
        ft = np.arange(48, dtype=np.float32).reshape(4, 12)
        result = causal_ft_history(
            timestamps,
            ft,
            anchor_timestamp=1.035,
            num_steps=4,
            frequency=100.0,
        )
        self.assertLessEqual(result["robot0_ft_left_timestamps"].max(), 1.035)
        np.testing.assert_array_equal(result["robot0_ft_left"][-1], ft[2, :6])
        np.testing.assert_array_equal(result["robot0_ft_right"][-1], ft[2, 6:])

    def test_anchor_before_first_ft_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "precedes the first causal"):
            causal_ft_history(
                np.array([1.01, 1.02]),
                np.zeros((2, 12), dtype=np.float32),
                anchor_timestamp=1.0,
                num_steps=4,
                frequency=100.0,
            )

    def test_history_before_first_target_repeats_first_causal_ft(self):
        timestamps = np.array([1.01, 1.02])
        ft = np.arange(24, dtype=np.float32).reshape(2, 12)
        result = causal_ft_history(
            timestamps,
            ft,
            anchor_timestamp=1.015,
            num_steps=4,
            frequency=100.0,
        )
        np.testing.assert_array_equal(
            result["robot0_ft_left"], np.repeat(ft[:1, :6], 4, axis=0)
        )

    def test_combined_ft_is_split_left_then_right(self):
        ft = np.arange(24, dtype=np.float32).reshape(2, 12)
        obs = {"robot0_ft": ft, "robot0_gripper_width": np.zeros((2, 1))}
        shape_meta = {
            "obs": {
                "robot0_ft_left": {"shape": [6]},
                "robot0_ft_right": {"shape": [6]},
            }
        }

        result = prepare_rg2ft_policy_obs(obs, shape_meta)

        np.testing.assert_array_equal(result["robot0_ft_left"], ft[:, :6])
        np.testing.assert_array_equal(result["robot0_ft_right"], ft[:, 6:])
        np.testing.assert_array_equal(result["robot0_ft"], ft)
        self.assertIsNot(result, obs)

    def test_non_rg2_shape_meta_leaves_observation_keys_unchanged(self):
        obs = {"robot0_ft": np.zeros((2, 12), dtype=np.float32)}
        result = prepare_rg2ft_policy_obs(
            obs, {"obs": {"robot0_ft": {"shape": [12]}}}
        )
        self.assertEqual(set(result), {"robot0_ft"})

    def test_invalid_combined_ft_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "12 channels"):
            prepare_rg2ft_policy_obs(
                {"robot0_ft": np.zeros((2, 6), dtype=np.float32)},
                {"obs": {"robot0_ft_left": {"shape": [6]}}},
            )


if __name__ == "__main__":
    unittest.main()
