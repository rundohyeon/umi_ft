import unittest

import numpy as np

from umi.real_world.rg2ft_obs import prepare_rg2ft_policy_obs


class RG2FTPolicyObsTest(unittest.TestCase):
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
