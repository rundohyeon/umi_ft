import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.dual_ft_contract import (
    inspect_dual_ft_checkpoint_payload,
)
from umi.real_world.rg2ft_obs import causal_ft_history_from_streams


def _payload(*, include_right=True, include_normalizer=True):
    obs = {
        "camera0_rgb": {
            "shape": [3, 224, 224], "horizon": 2, "type": "rgb",
            "ignore_by_policy": False, "down_sample_steps": 3,
        },
        "robot0_eef_pos": {
            "shape": [3], "horizon": 2, "type": "low_dim",
            "ignore_by_policy": False, "down_sample_steps": 3,
        },
        "robot0_eef_rot_axis_angle": {
            "shape": [6], "horizon": 2, "type": "low_dim",
            "ignore_by_policy": False, "down_sample_steps": 3,
            "rotation_rep": "rotation_6d",
        },
        "robot0_ft_left": {
            "shape": [6], "horizon": 32, "type": "low_dim",
            "ignore_by_policy": False, "down_sample_steps": 1,
        },
    }
    if include_right:
        obs["robot0_ft_right"] = {
            "shape": [6], "horizon": 32, "type": "low_dim",
            "ignore_by_policy": False, "down_sample_steps": 1,
        }
    state = {
        "obs_encoder.architecture_contract_version": torch.tensor(2),
        "obs_encoder.left_ft_encoder.network.0.conv.weight": torch.empty(16, 6, 2),
        "obs_encoder.right_ft_encoder.network.0.conv.weight": torch.empty(16, 6, 2),
        "obs_encoder.left_ft_encoder.temporal_contract_version": torch.tensor(1),
        "obs_encoder.right_ft_encoder.temporal_contract_version": torch.tensor(1),
        "obs_encoder.position_embedding": torch.empty(4, 768),
        "obs_encoder.fusion_projection.weight": torch.empty(768, 3072),
        "obs_encoder.fusion_projection.bias": torch.empty(768),
        "normalizer.params_dict.action.scale": torch.ones(11),
        "normalizer.params_dict.action.offset": torch.zeros(11),
    }
    if include_normalizer:
        state.update(
            {
                "normalizer.params_dict.robot0_ft_left.scale": torch.ones(6),
                "normalizer.params_dict.robot0_ft_left.offset": torch.zeros(6),
                "normalizer.params_dict.robot0_ft_right.scale": torch.ones(6),
                "normalizer.params_dict.robot0_ft_right.offset": torch.zeros(6),
            }
        )
    return {
        "cfg": OmegaConf.create(
            {
                "task": {
                    "model_contract": {
                        "version": (
                            "dual_ft_786_action11_base_sidecar_bias_only_width_feedback_v6"
                        ),
                        "condition_dim": 786,
                        "pose_quaternion_order": "xyzw",
                        "offline_pose_source_representation": "axis_angle",
                        "vision_pretrained": True,
                        "ft_temporal_contract": "full_32_samples_no_padding_v1",
                        "action_schema_version": "pose9_width1_grasp_force1_v1",
                        "action_channels": [
                            "x", "y", "z",
                            "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
                            "gripper_width_m", "grasp_force_N",
                        ],
                        "grasp_force_source": "derived_from_sidecar_wrench_12d",
                        "ft_input_schema_version": "native_dual_wrench12_bias_only_v1",
                        "ft_history_padding": "repeat_first",
                        "ft_coordinate_transform": "none",
                        "ft_channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                        "ft_force_unit": "N",
                        "ft_torque_unit": "Nm",
                        "ft_left_frame": "left_native_sensor",
                        "ft_right_frame": "right_native_sensor",
                        "ft_input_key": "wrench_12d",
                        "ft_bias_metadata_key": "wrench_episode_bias_12d",
                        "ft_bias_key": "wrench_episode_bias_12d",
                        "ft_bias_removal": "precomputed_in_sidecar",
                        "deployment_bias_requirement": "startup_static_calibration",
                        "grasp_force_semantics_version": "signed_native_fz_measurement_v1",
                        "grasp_force_formula": "0.5*((right_Fz-right_bias_Fz)-(left_Fz-left_bias_Fz))",
                        "grasp_force_alignment": "linear_interpolation_to_rgb_timestamp",
                        "grasp_force_role": "gripper_width_feedback_reference_not_direct_command_v1",
                        "grasp_force_feedback_control_law": "bounded_proportional_width_correction_v1",
                        "action_pose_semantics": "anchor_relative_xyz_rotation6d",
                        "gripper_width_semantics": "absolute_measured_width_m",
                    },
                    "shape_meta": {
                        "obs": obs,
                        "action": {
                            "shape": [11], "horizon": 16,
                            "rotation_rep": "rotation_6d",
                            "down_sample_steps": 3,
                        },
                    },
                    "pose_repr": {
                        "obs_pose_repr": "relative",
                        "action_pose_repr": "relative",
                    },
                    "pose_quaternion_order": "xyzw",
                    "dataset_path": "session_260827/dataset.zarr.zip",
                    "force_sidecar_path": (
                        "session_260827/dataset_force_sidecar.zarr"
                    ),
                    "dataset": {
                        "dataset_path": "session_260827/dataset.zarr.zip",
                        "force_sidecar_path": (
                            "session_260827/dataset_force_sidecar.zarr"
                        ),
                        "pose_quaternion_order": "xyzw",
                        "data_keys": {
                            "rgb": "camera0_rgb",
                            "pose_position": "robot0_eef_pos",
                            "pose_rotation_axis_angle": (
                                "robot0_eef_rot_axis_angle"
                            ),
                            "gripper": "robot0_gripper_width",
                            "rgb_episode_ends": "episode_ends",
                        },
                        "ft": {
                            "wrench_key": "wrench_12d",
                            "timestamp_key": "wrench_timestamp_s",
                            "episode_ends_key": "wrench_episode_ends",
                            "rgb_timestamp_key": "rgb_timestamp_s",
                            "rgb_episode_ends_key": "rgb_episode_ends",
                            "rgb_to_wrench_end_idx_key": "rgb_to_wrench_end_idx",
                            "rgb_wrench_age_key": "rgb_wrench_age_s",
                            "rgb_wrench_valid_key": "rgb_wrench_valid",
                            "padding": "repeat_first",
                            "channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                            "force_unit": "N",
                            "torque_unit": "Nm",
                            "left_frame": "left_native_sensor",
                            "right_frame": "right_native_sensor",
                            "axis_permutation": [0, 1, 2, 3, 4, 5],
                            "axis_sign": [1, 1, 1, 1, 1, 1],
                            "bias_key": "wrench_episode_bias_12d",
                            "bias_removal": "precomputed_in_sidecar",
                            "deployment_bias_requirement": "startup_static_calibration",
                            "num_steps": 32,
                            "stride": 1,
                            "history_seconds": 0.31,
                        },
                    },
                    "ft": {
                        "wrench_key": "wrench_12d",
                        "timestamp_key": "wrench_timestamp_s",
                        "episode_ends_key": "wrench_episode_ends",
                        "rgb_timestamp_key": "rgb_timestamp_s",
                        "rgb_episode_ends_key": "rgb_episode_ends",
                        "rgb_to_wrench_end_idx_key": "rgb_to_wrench_end_idx",
                        "rgb_wrench_age_key": "rgb_wrench_age_s",
                        "rgb_wrench_valid_key": "rgb_wrench_valid",
                        "padding": "repeat_first",
                        "channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                        "force_unit": "N",
                        "torque_unit": "Nm",
                        "left_frame": "left_native_sensor",
                        "right_frame": "right_native_sensor",
                        "axis_permutation": [0, 1, 2, 3, 4, 5],
                        "axis_sign": [1, 1, 1, 1, 1, 1],
                        "bias_key": "wrench_episode_bias_12d",
                        "bias_removal": "precomputed_in_sidecar",
                        "deployment_bias_requirement": "startup_static_calibration",
                        "num_steps": 32,
                        "stride": 1,
                        "history_seconds": 0.31,
                    },
                    "ft_obs_horizon": 32,
                    "ft_obs_stride": 1,
                    "ft_history_seconds": 0.31,
                    "ignore_proprioception": False,
                    "grasp_force_feedback": {
                        "mode": "gripper_width_correction",
                        "kp_m_per_n": 1e-4,
                        "force_deadband_n": 0.5,
                        "max_width_correction_m": 1e-3,
                        "target_force_min_n": 0.0,
                        "target_force_max_n": 12.0,
                        "width_min_m": 0.0,
                        "width_max_m": 0.1,
                        "requires_startup_bias": True,
                        "direct_force_command": False,
                    },
                },
                "policy": {
                    "diffusion_step_embed_dim": 32,
                    "obs_encoder": {
                        "pretrained": True,
                        "fusion_dim": 768,
                        "fusion_heads": 8,
                        "fusion_layers": 1,
                        "fusion_feedforward_dim": 2048,
                        "fusion_dropout": 0.0,
                        "fusion_position_encoding": "learnable",
                        "left_ft_key": "robot0_ft_left",
                        "right_ft_key": "robot0_ft_right",
                        "ft_channel_dims": [16, 32, 64, 128],
                        "share_ft_encoder": False,
                        "vision_feature_dim": 768,
                    },
                },
                "training": {"use_ema": False},
            }
        ),
        "state_dicts": {"model": state},
    }


class DualFTInferenceContractTest(unittest.TestCase):
    def test_checkpoint_contract_requires_dual_encoders_and_normalizers(self):
        contract = inspect_dual_ft_checkpoint_payload(_payload())
        self.assertEqual(contract["condition_dim"], 786)
        self.assertEqual(contract["action_horizon"], 16)
        self.assertEqual(contract["action_dim"], 11)
        self.assertEqual(contract["normalizer_owner"], "policy.predict_action")

    def test_rgb_only_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RGB-only/non-dual"):
            inspect_dual_ft_checkpoint_payload(_payload(include_right=False))

    def test_missing_ft_normalizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalizer"):
            inspect_dual_ft_checkpoint_payload(_payload(include_normalizer=False))

    def test_wrong_temporal_marker_is_rejected(self):
        payload = _payload()
        payload["state_dicts"]["model"][
            "obs_encoder.left_ft_encoder.temporal_contract_version"
        ] = torch.tensor(0)
        with self.assertRaisesRegex(ValueError, "must equal 1"):
            inspect_dual_ft_checkpoint_payload(payload)

    def test_pre_xyzw_fix_checkpoint_is_rejected(self):
        payload = _payload()
        payload["cfg"].task.model_contract.version = (
            "dual_ft_786_action11_bias_only_v3"
        )
        with self.assertRaisesRegex(ValueError, "model_contract.version"):
            inspect_dual_ft_checkpoint_payload(payload)

    def test_non_pretrained_vision_checkpoint_is_rejected(self):
        payload = _payload()
        payload["cfg"].policy.obs_encoder.pretrained = False
        with self.assertRaisesRegex(ValueError, "pretrained"):
            inspect_dual_ft_checkpoint_payload(payload)

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
