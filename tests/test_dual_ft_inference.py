import unittest

import numpy as np
from omegaconf import OmegaConf

from eval_real_indy_rg2 import inspect_dual_ft_checkpoint_payload
from umi.real_world.rg2ft_obs import causal_ft_history_from_streams


def _payload(*, include_right=True, include_normalizer=True):
    obs = {
        "camera0_rgb": {"shape": [3, 224, 224], "horizon": 2},
        "robot0_ft_left": {"shape": [6], "horizon": 32},
    }
    if include_right:
        obs["robot0_ft_right"] = {"shape": [6], "horizon": 32}
    state = {
        "obs_encoder.left_ft_encoder.network.0.conv.weight": 0,
        "obs_encoder.right_ft_encoder.network.0.conv.weight": 0,
        "normalizer.params_dict.action.offset": 0,
    }
    if include_normalizer:
        state.update(
            {
                "normalizer.params_dict.robot0_ft_left.offset": 0,
                "normalizer.params_dict.robot0_ft_right.offset": 0,
            }
        )
    return {
        "cfg": OmegaConf.create(
            {
                "task": {
                    "shape_meta": {
                        "obs": obs,
                        "action": {"shape": [10], "horizon": 16},
                    }
                },
                "policy": {"obs_encoder": {"fusion_dim": 768, "low_dim_output": 32}},
            }
        ),
        "state_dicts": {"model": state},
    }


class DualFTInferenceContractTest(unittest.TestCase):
    def test_checkpoint_contract_requires_dual_encoders_and_normalizers(self):
        contract = inspect_dual_ft_checkpoint_payload(_payload())
        self.assertEqual(contract["condition_dim"], 800)
        self.assertEqual(contract["action_horizon"], 16)
        self.assertEqual(contract["action_dim"], 10)
        self.assertEqual(contract["normalizer_owner"], "policy.predict_action")

    def test_rgb_only_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RGB-only/non-dual"):
            inspect_dual_ft_checkpoint_payload(_payload(include_right=False))

    def test_missing_ft_normalizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalizer"):
            inspect_dual_ft_checkpoint_payload(_payload(include_normalizer=False))

    def test_independent_streams_have_independent_causal_indices(self):
        left_t = np.array([1.00, 1.02, 1.04])
        right_t = np.array([1.01, 1.03, 1.05])
        left = np.arange(18, dtype=np.float32).reshape(3, 6)
        right = np.arange(100, 118, dtype=np.float32).reshape(3, 6)
        result = causal_ft_history_from_streams(
            left_t,
            left,
            right_t,
            right,
            anchor_timestamp=1.045,
            num_steps=3,
            frequency=100.0,
        )
        np.testing.assert_array_equal(result["robot0_ft_left"][-1], left[2])
        np.testing.assert_array_equal(result["robot0_ft_right"][-1], right[1])
        self.assertLessEqual(result["robot0_ft_left_timestamps"].max(), 1.045)
        self.assertLessEqual(result["robot0_ft_right_timestamps"].max(), 1.045)

    def test_stale_stream_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            causal_ft_history_from_streams(
                np.array([1.0]), np.zeros((1, 6), dtype=np.float32),
                np.array([1.0]), np.zeros((1, 6), dtype=np.float32),
                anchor_timestamp=1.1,
                num_steps=32,
                frequency=100.0,
                max_age=0.012,
            )


if __name__ == "__main__":
    unittest.main()
