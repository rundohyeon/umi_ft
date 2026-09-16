"""Hardware-free checkpoint contract validation for the dual-F/T policy."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.valve_context_contract import (
    VALVE_CONTEXT_V1_SCHEMA,
    VALVE_CONTEXT_V2_SCHEMA,
    valve_context_spec,
)


EXPECTED_ACTION_CHANNELS = [
    "x",
    "y",
    "z",
    "r6d_0",
    "r6d_1",
    "r6d_2",
    "r6d_3",
    "r6d_4",
    "r6d_5",
    "gripper_width_m",
    "grasp_force_N",
]

_VALVE_CONTEXT_KEY = "valve_context"


def inspect_dual_ft_checkpoint_payload(payload: dict) -> dict:
    """Fail closed unless ``payload`` implements the corrected 786/11 contract.

    The function only examines resolved configuration and CPU state dictionaries.
    It does not instantiate a policy, dataset, camera, or robot interface.
    """

    if not isinstance(payload, dict) or "cfg" not in payload:
        raise ValueError("checkpoint payload is missing resolved cfg")
    cfg = payload["cfg"]
    obs_meta = OmegaConf.select(cfg, "task.shape_meta.obs", default=None)
    action_meta = OmegaConf.select(cfg, "task.shape_meta.action", default=None)
    if obs_meta is None or action_meta is None:
        raise ValueError("checkpoint cfg is missing task.shape_meta")

    is_valve_context_policy = _VALVE_CONTEXT_KEY in obs_meta
    if is_valve_context_policy:
        context_schema = str(
            OmegaConf.select(
                cfg,
                "task.model_contract.valve_context_schema",
                default=VALVE_CONTEXT_V1_SCHEMA,
            )
        )
        context_spec = valve_context_spec(context_schema)
        context_dim = int(context_spec["dim"])
        context_phase_names = list(context_spec["phase_names"])
        context_reason_names = list(context_spec["reason_names"])
        context_num_experts = int(context_spec["num_phase_experts"])
    else:
        context_schema = None
        context_spec = None
        context_dim = 0
        context_phase_names = []
        context_reason_names = []
        context_num_experts = 0
    architecture_contract_version = (
        (4 if context_schema == VALVE_CONTEXT_V2_SCHEMA else 3)
        if is_valve_context_policy
        else 2
    )
    expected_obs_shapes = {
        "camera0_rgb": (2, (3, 224, 224)),
        "robot0_eef_pos": (2, (3,)),
        "robot0_eef_rot_axis_angle": (2, (6,)),
        "robot0_ft_left": (32, (6,)),
        "robot0_ft_right": (32, (6,)),
    }
    if is_valve_context_policy:
        expected_obs_shapes[_VALVE_CONTEXT_KEY] = (1, (context_dim,))
    required_modalities = {"camera0_rgb", "robot0_ft_left", "robot0_ft_right"}
    missing_modalities = sorted(required_modalities - set(obs_meta.keys()))
    if missing_modalities:
        raise ValueError(
            "RGB-only/non-dual checkpoint rejected; missing required obs keys: "
            + ", ".join(missing_modalities)
        )
    if set(obs_meta.keys()) != set(expected_obs_shapes):
        raise ValueError(
            "checkpoint observation keys do not match the 18-D proprioception "
            f"contract: expected={sorted(expected_obs_shapes)} "
            f"got={sorted(obs_meta.keys())}"
        )
    for key, (expected_horizon, expected_shape) in expected_obs_shapes.items():
        meta = obs_meta[key]
        actual_horizon = int(meta.get("horizon", -1))
        actual_shape = tuple(meta.get("shape", ()))
        if (actual_horizon, actual_shape) != (expected_horizon, expected_shape):
            raise ValueError(
                f"{key} must be [{expected_horizon},"
                f"{','.join(map(str, expected_shape))}], got "
                f"horizon={actual_horizon} shape={actual_shape}"
            )
    expected_obs_attributes = {
        "camera0_rgb": ("rgb", False, 3),
        "robot0_eef_pos": ("low_dim", False, 3),
        "robot0_eef_rot_axis_angle": ("low_dim", False, 3),
        "robot0_ft_left": ("low_dim", False, 1),
        "robot0_ft_right": ("low_dim", False, 1),
    }
    if is_valve_context_policy:
        # Context is consumed by ValveStageConditioner, not flattened by the
        # inherited Timm low-dimensional encoder.
        expected_obs_attributes[_VALVE_CONTEXT_KEY] = ("low_dim", True, 1)
    for key, (expected_type, expected_ignored, expected_stride) in (
        expected_obs_attributes.items()
    ):
        meta = obs_meta[key]
        actual = (
            str(meta.get("type", "")),
            bool(meta.get("ignore_by_policy", False)),
            int(meta.get("down_sample_steps", -1)),
        )
        expected = (expected_type, expected_ignored, expected_stride)
        if actual != expected:
            raise ValueError(
                f"checkpoint observation metadata for {key} must be "
                f"type/ignored/stride={expected}, got {actual}"
            )
    if str(obs_meta["robot0_eef_rot_axis_angle"].get("rotation_rep", "")) != (
        "rotation_6d"
    ):
        raise ValueError("robot0_eef_rot_axis_angle must use rotation_6d")

    if (
        tuple(action_meta.get("shape", ())) != (11,)
        or int(action_meta.get("horizon", -1)) != 16
    ):
        raise ValueError(
            "checkpoint action contract must be [16,11] "
            "(pose9 + width1 + grasp_force1), got "
            f"horizon={action_meta.get('horizon')} shape={action_meta.get('shape')}"
        )
    if (
        str(action_meta.get("rotation_rep", "")) != "rotation_6d"
        or int(action_meta.get("down_sample_steps", -1)) != 3
    ):
        raise ValueError(
            "checkpoint action metadata must use rotation_6d and down_sample_steps=3"
        )
    diffusion_step_embed_dim = int(
        OmegaConf.select(cfg, "policy.diffusion_step_embed_dim", default=-1)
    )
    if diffusion_step_embed_dim != 32:
        raise ValueError(
            "checkpoint diffusion timestep embedding must be 32-D, got "
            f"{diffusion_step_embed_dim}"
        )

    contract_requirements = {
        "task.model_contract.version": (
            (
                "dual_ft_context_v2_4state_786_action11_v1"
                if context_schema == VALVE_CONTEXT_V2_SCHEMA
                else "dual_ft_context_786_action11_v1"
            )
            if is_valve_context_policy
            else "dual_ft_786_action11_base_sidecar_bias_only_width_feedback_v6"
        ),
        "task.model_contract.condition_dim": 786,
        "task.model_contract.pose_quaternion_order": "xyzw",
        "task.model_contract.offline_pose_source_representation": "axis_angle",
        "task.model_contract.vision_pretrained": True,
        "task.model_contract.ft_temporal_contract": (
            "full_32_samples_no_padding_v1"
        ),
        "task.model_contract.action_schema_version": (
            "pose9_width1_grasp_force1_v1"
        ),
        "task.model_contract.grasp_force_source": (
            "derived_from_sidecar_wrench_12d"
        ),
        "task.model_contract.ft_input_schema_version": (
            "native_dual_wrench12_bias_only_v1"
        ),
        "task.model_contract.ft_history_padding": "repeat_first",
        "task.model_contract.ft_coordinate_transform": "none",
        "task.model_contract.ft_force_unit": "N",
        "task.model_contract.ft_torque_unit": "Nm",
        "task.model_contract.ft_left_frame": "left_native_sensor",
        "task.model_contract.ft_right_frame": "right_native_sensor",
        "task.model_contract.ft_input_key": "wrench_12d",
        "task.model_contract.ft_bias_metadata_key": "wrench_episode_bias_12d",
        "task.model_contract.ft_bias_key": "wrench_episode_bias_12d",
        "task.model_contract.ft_bias_removal": "precomputed_in_sidecar",
        "task.model_contract.deployment_bias_requirement": (
            "startup_static_calibration"
        ),
        "task.model_contract.grasp_force_semantics_version": (
            "signed_native_fz_measurement_v1"
        ),
        "task.model_contract.grasp_force_formula": (
            "0.5*((right_Fz-right_bias_Fz)-(left_Fz-left_bias_Fz))"
        ),
        "task.model_contract.grasp_force_alignment": (
            "linear_interpolation_to_rgb_timestamp"
        ),
        "task.model_contract.grasp_force_role": (
            "gripper_width_feedback_reference_not_direct_command_v1"
        ),
        "task.model_contract.grasp_force_feedback_control_law": (
            "bounded_proportional_width_correction_v1"
        ),
        "task.model_contract.action_pose_semantics": (
            "anchor_relative_xyz_rotation6d"
        ),
        "task.model_contract.gripper_width_semantics": (
            "absolute_measured_width_m"
        ),
        "task.dataset.data_keys.rgb": "camera0_rgb",
        "task.dataset.data_keys.pose_position": "robot0_eef_pos",
        "task.dataset.data_keys.pose_rotation_axis_angle": (
            "robot0_eef_rot_axis_angle"
        ),
        "task.dataset.data_keys.gripper": "robot0_gripper_width",
        "task.dataset.data_keys.rgb_episode_ends": "episode_ends",
        "task.dataset.pose_quaternion_order": "xyzw",
        "task.pose_quaternion_order": "xyzw",
        "task.grasp_force_feedback.mode": "gripper_width_correction",
        "task.grasp_force_feedback.kp_m_per_n": 1.0e-4,
        "task.grasp_force_feedback.force_deadband_n": 0.5,
        "task.grasp_force_feedback.max_width_correction_m": 1.0e-3,
        "task.grasp_force_feedback.target_force_min_n": 0.0,
        "task.grasp_force_feedback.target_force_max_n": 12.0,
        "task.grasp_force_feedback.width_min_m": 0.0,
        "task.grasp_force_feedback.width_max_m": 0.1,
        "task.grasp_force_feedback.requires_startup_bias": True,
        "task.grasp_force_feedback.direct_force_command": False,
        "policy.obs_encoder.pretrained": True,
        "task.dataset.ft.wrench_key": "wrench_12d",
        "task.dataset.ft.timestamp_key": "wrench_timestamp_s",
        "task.dataset.ft.episode_ends_key": "wrench_episode_ends",
        "task.dataset.ft.rgb_timestamp_key": "rgb_timestamp_s",
        "task.dataset.ft.rgb_episode_ends_key": "rgb_episode_ends",
        "task.dataset.ft.rgb_to_wrench_end_idx_key": "rgb_to_wrench_end_idx",
        "task.dataset.ft.rgb_wrench_age_key": "rgb_wrench_age_s",
        "task.dataset.ft.rgb_wrench_valid_key": "rgb_wrench_valid",
        "task.dataset.ft.padding": "repeat_first",
        "task.dataset.ft.force_unit": "N",
        "task.dataset.ft.torque_unit": "Nm",
        "task.dataset.ft.left_frame": "left_native_sensor",
        "task.dataset.ft.right_frame": "right_native_sensor",
        "task.dataset.ft.bias_key": "wrench_episode_bias_12d",
        "task.dataset.ft.bias_removal": "precomputed_in_sidecar",
        "task.dataset.ft.deployment_bias_requirement": (
            "startup_static_calibration"
        ),
        "task.dataset.ft.num_steps": 32,
        "task.dataset.ft.stride": 1,
        "task.dataset.ft.history_seconds": 0.31,
        "task.ft.wrench_key": "wrench_12d",
        "task.ft.timestamp_key": "wrench_timestamp_s",
        "task.ft.episode_ends_key": "wrench_episode_ends",
        "task.ft.rgb_timestamp_key": "rgb_timestamp_s",
        "task.ft.rgb_episode_ends_key": "rgb_episode_ends",
        "task.ft.rgb_to_wrench_end_idx_key": "rgb_to_wrench_end_idx",
        "task.ft.rgb_wrench_age_key": "rgb_wrench_age_s",
        "task.ft.rgb_wrench_valid_key": "rgb_wrench_valid",
        "task.ft.padding": "repeat_first",
        "task.ft.force_unit": "N",
        "task.ft.torque_unit": "Nm",
        "task.ft.left_frame": "left_native_sensor",
        "task.ft.right_frame": "right_native_sensor",
        "task.ft.bias_key": "wrench_episode_bias_12d",
        "task.ft.bias_removal": "precomputed_in_sidecar",
        "task.ft.deployment_bias_requirement": "startup_static_calibration",
        "task.ft.num_steps": 32,
        "task.ft.stride": 1,
        "task.ft.history_seconds": 0.31,
        "task.ft_obs_horizon": 32,
        "task.ft_obs_stride": 1,
        "task.ft_history_seconds": 0.31,
        "task.ignore_proprioception": False,
        "task.pose_repr.obs_pose_repr": "relative",
        "task.pose_repr.action_pose_repr": "relative",
        "policy.obs_encoder.left_ft_key": "robot0_ft_left",
        "policy.obs_encoder.right_ft_key": "robot0_ft_right",
        "policy.obs_encoder.share_ft_encoder": False,
        "policy.obs_encoder.vision_feature_dim": 768,
    }
    if is_valve_context_policy:
        is_v2_context = context_schema == VALVE_CONTEXT_V2_SCHEMA
        contract_requirements.update(
            {
                "task.model_contract.valve_context_schema": context_schema,
                "task.model_contract.valve_context_dim": context_dim,
                "task.model_contract.valve_context_key": _VALVE_CONTEXT_KEY,
                "task.model_contract.valve_context_classifier_version": (
                    "observer_v2_4state" if is_v2_context else "v4"
                ),
                "task.model_contract.valve_context_classifier_frozen": True,
                "task.model_contract.valve_context_routing": (
                    "soft_phase_moe_residual_v2"
                    if is_v2_context
                    else "soft_phase_moe_residual_v1"
                ),
                "task.model_contract.valve_context_input_wrench_key": (
                    "wrench_12d" if is_v2_context else "wrench_12d_capture"
                ),
                "policy.obs_encoder.valve_context_key": _VALVE_CONTEXT_KEY,
                "policy.obs_encoder.valve_context_dim": context_dim,
                "policy.obs_encoder.valve_context_hidden_dim": 128,
                "policy.obs_encoder.valve_expert_bottleneck_dim": 128,
            }
        )
        if is_v2_context:
            contract_requirements[
                "policy.obs_encoder.valve_context_num_phase_experts"
            ] = context_num_experts
    for path, expected in contract_requirements.items():
        actual = OmegaConf.select(cfg, path, default=None)
        if actual != expected:
            raise ValueError(
                f"checkpoint contract marker {path} must be {expected!r}, "
                f"got {actual!r}"
            )
    base_dataset_path = str(
        OmegaConf.select(cfg, "task.dataset.dataset_path", default="")
    )
    force_sidecar_path = str(
        OmegaConf.select(cfg, "task.dataset.force_sidecar_path", default="")
    )
    if not base_dataset_path or not force_sidecar_path:
        raise ValueError(
            "checkpoint must serialize both base dataset_path and "
            "force_sidecar_path"
        )
    if base_dataset_path == force_sidecar_path:
        raise ValueError("base dataset and force sidecar paths must differ")
    list_requirements = {
        "task.model_contract.ft_channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "task.dataset.ft.channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "task.dataset.ft.axis_permutation": [0, 1, 2, 3, 4, 5],
        "task.dataset.ft.axis_sign": [1, 1, 1, 1, 1, 1],
        "task.ft.channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "task.ft.axis_permutation": [0, 1, 2, 3, 4, 5],
        "task.ft.axis_sign": [1, 1, 1, 1, 1, 1],
        "policy.obs_encoder.ft_channel_dims": [16, 32, 64, 128],
    }
    for path, expected in list_requirements.items():
        actual = list(OmegaConf.select(cfg, path, default=[]))
        if actual != expected:
            raise ValueError(
                f"checkpoint contract marker {path} must be {expected!r}, "
                f"got {actual!r}"
            )
    action_channels = list(
        OmegaConf.select(cfg, "task.model_contract.action_channels", default=[])
    )
    if action_channels != EXPECTED_ACTION_CHANNELS:
        raise ValueError(
            "checkpoint action channel schema mismatch: "
            f"expected={EXPECTED_ACTION_CHANNELS} got={action_channels}"
        )
    if is_valve_context_policy:
        expected_context_key = (
            "valve_context_5d"
            if context_schema == VALVE_CONTEXT_V2_SCHEMA
            else _VALVE_CONTEXT_KEY
        )
        for path, expected in {
            "task.model_contract.valve_context_phase_names": context_phase_names,
            "task.model_contract.valve_context_error_reason_names": context_reason_names,
            "task.valve_context.schema": context_schema,
            "task.valve_context.context_key": expected_context_key,
            "task.valve_context.timestamp_key": (
                "valve_context_timestamp_s"
            ),
            "task.valve_context.episode_ends_key": "rgb_episode_ends",
            "task.valve_context.input_wrench_key": (
                "wrench_12d"
                if context_schema == VALVE_CONTEXT_V2_SCHEMA
                else "wrench_12d_capture"
            ),
            "task.valve_context.classifier_frozen": True,
        }.items():
            actual = OmegaConf.select(cfg, path, default=None)
            if isinstance(expected, list):
                actual = list(actual) if actual is not None else []
            if actual != expected:
                raise ValueError(
                    f"checkpoint context marker {path} must be {expected!r}, got {actual!r}"
                )
        classifier_sha256 = str(
            OmegaConf.select(cfg, "task.valve_context.checkpoint_sha256", default="")
        ).lower()
        if len(classifier_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in classifier_sha256
        ):
            raise ValueError(
                "checkpoint task.valve_context.checkpoint_sha256 must be a 64-char hex digest"
            )

    fusion_requirements = {
        "policy.obs_encoder.fusion_dim": 768,
        "policy.obs_encoder.fusion_heads": 8,
        "policy.obs_encoder.fusion_layers": 1,
        "policy.obs_encoder.fusion_feedforward_dim": 2048,
        "policy.obs_encoder.fusion_dropout": 0.0,
    }
    for path, expected in fusion_requirements.items():
        actual = OmegaConf.select(cfg, path, default=None)
        if actual is None or float(actual) != float(expected):
            raise ValueError(
                f"checkpoint official-fusion contract {path} must be "
                f"{expected}, got {actual}"
            )
    if (
        str(
            OmegaConf.select(
                cfg,
                "policy.obs_encoder.fusion_position_encoding",
                default="",
            )
        )
        != "learnable"
    ):
        raise ValueError("checkpoint must use learnable fusion position encoding")

    state_dicts = payload.get("state_dicts", {})
    if not isinstance(state_dicts, dict) or not state_dicts:
        raise ValueError("checkpoint is missing model state dictionaries")
    policy_states = {
        name: state_dicts.get(name)
        for name in ("model", "ema_model")
        if isinstance(state_dicts.get(name), dict) and state_dicts[name]
    }
    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    required_policy_state = "ema_model" if use_ema else "model"
    if required_policy_state not in policy_states:
        raise ValueError(
            f"checkpoint training.use_ema={use_ema} requires non-empty "
            f"state_dicts[{required_policy_state!r}]"
        )

    required_state_fragments = (
        "obs_encoder.architecture_contract_version",
        "obs_encoder.left_ft_encoder.network.",
        "obs_encoder.right_ft_encoder.network.",
        "obs_encoder.left_ft_encoder.temporal_contract_version",
        "obs_encoder.right_ft_encoder.temporal_contract_version",
        "obs_encoder.position_embedding",
        "obs_encoder.fusion_projection.",
        "normalizer.params_dict.robot0_ft_left.",
        "normalizer.params_dict.robot0_ft_right.",
        "normalizer.params_dict.action.",
    )
    if is_valve_context_policy:
        required_state_fragments = required_state_fragments + (
            "obs_encoder.valve_stage_conditioner.stage_encoder.",
            "obs_encoder.valve_stage_conditioner.experts.",
            "normalizer.params_dict.valve_context.",
        )
    deprecated_fusion_state = (
        "obs_encoder.cls_token",
        "obs_encoder.token_embedding",
        "obs_encoder.fusion_norm.",
    )
    exact_state_contract = {
        "obs_encoder.architecture_contract_version": (
            (),
            architecture_contract_version,
        ),
        "obs_encoder.left_ft_encoder.temporal_contract_version": ((), 1),
        "obs_encoder.right_ft_encoder.temporal_contract_version": ((), 1),
        "obs_encoder.position_embedding": ((4, 768), None),
        "obs_encoder.fusion_projection.weight": ((768, 3072), None),
        "obs_encoder.fusion_projection.bias": ((768,), None),
        "normalizer.params_dict.robot0_ft_left.scale": ((6,), None),
        "normalizer.params_dict.robot0_ft_left.offset": ((6,), None),
        "normalizer.params_dict.robot0_ft_right.scale": ((6,), None),
        "normalizer.params_dict.robot0_ft_right.offset": ((6,), None),
        "normalizer.params_dict.action.scale": ((11,), None),
        "normalizer.params_dict.action.offset": ((11,), None),
    }
    if is_valve_context_policy:
        exact_state_contract.update(
            {
                "obs_encoder.valve_stage_conditioner.stage_encoder.0.weight": (
                    (128, context_dim), None
                ),
                "obs_encoder.valve_stage_conditioner.stage_encoder.0.bias": (
                    (128,), None
                ),
                "obs_encoder.valve_stage_conditioner.stage_encoder.2.weight": (
                    (768, 128), None
                ),
                "obs_encoder.valve_stage_conditioner.stage_encoder.2.bias": (
                    (768,), None
                ),
                "normalizer.params_dict.valve_context.scale": ((1,), None),
                "normalizer.params_dict.valve_context.offset": ((1,), None),
            }
        )
    for state_name, state in policy_states.items():
        absent = [
            fragment
            for fragment in required_state_fragments
            if not any(str(key).startswith(fragment) for key in state)
        ]
        if absent:
            raise ValueError(
                f"checkpoint {state_name} has no restorable dual-F/T "
                "normalizer/encoder state: " + ", ".join(absent)
            )
        present_deprecated = [
            fragment
            for fragment in deprecated_fusion_state
            if any(str(key).startswith(fragment) for key in state)
        ]
        if present_deprecated:
            raise ValueError(
                f"checkpoint {state_name} uses the deprecated CLS-token "
                "fusion path: " + ", ".join(present_deprecated)
            )
        for key, (expected_shape, expected_scalar) in exact_state_contract.items():
            if key not in state:
                raise ValueError(f"checkpoint {state_name} is missing state {key}")
            value = state[key]
            actual_shape = tuple(getattr(value, "shape", ()))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"checkpoint {state_name} state {key} must have shape "
                    f"{expected_shape}, got {actual_shape}"
                )
            if expected_scalar is not None:
                scalar = int(torch.as_tensor(value).detach().cpu().item())
                if scalar != expected_scalar:
                    raise ValueError(
                        f"checkpoint {state_name} state {key} must equal "
                        f"{expected_scalar}, got {scalar}"
                    )
        if is_valve_context_policy:
            for expert_idx in range(context_num_experts):
                expert_shapes = {
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.0.weight": (768,),
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.0.bias": (768,),
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.1.weight": (128, 768),
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.1.bias": (128,),
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.3.weight": (768, 128),
                    f"obs_encoder.valve_stage_conditioner.experts.{expert_idx}.network.3.bias": (768,),
                }
                for key, expected_shape in expert_shapes.items():
                    if key not in state or tuple(getattr(state[key], "shape", ())) != expected_shape:
                        actual_shape = tuple(getattr(state.get(key), "shape", ()))
                        raise ValueError(
                            f"checkpoint {state_name} context expert state {key} "
                            f"must have shape {expected_shape}, got {actual_shape}"
                        )
            context_scale = torch.as_tensor(
                state["normalizer.params_dict.valve_context.scale"]
            ).detach().cpu()
            context_offset = torch.as_tensor(
                state["normalizer.params_dict.valve_context.offset"]
            ).detach().cpu()
            if not bool(torch.allclose(context_scale, torch.ones_like(context_scale))):
                raise ValueError(
                    f"checkpoint {state_name} valve_context normalizer must be identity scale"
                )
            if not bool(torch.allclose(context_offset, torch.zeros_like(context_offset))):
                raise ValueError(
                    f"checkpoint {state_name} valve_context normalizer must be identity offset"
                )
        for key, value in state.items():
            if not str(key).startswith("normalizer.params_dict."):
                continue
            tensor = torch.as_tensor(value).detach().cpu()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"checkpoint {state_name} normalizer state {key} "
                    "contains NaN or Inf"
                )

    low_dim_output = sum(
        int(np.prod(meta.get("shape", ()))) * int(meta.get("horizon", 1))
        for key, meta in obs_meta.items()
        if key not in ("robot0_ft_left", "robot0_ft_right")
        and str(meta.get("type", "")) != "rgb"
        and not key.endswith("_rgb")
        and not bool(meta.get("ignore_by_policy", False))
    )
    serialized_low_dim_output = OmegaConf.select(
        cfg, "policy.obs_encoder.low_dim_output", default=None
    )
    if serialized_low_dim_output is not None:
        low_dim_output = int(serialized_low_dim_output)
    condition_dim = int(
        OmegaConf.select(cfg, "policy.obs_encoder.fusion_dim", default=0)
    ) + low_dim_output
    if condition_dim != 786:
        raise ValueError(
            f"checkpoint observation condition must be 786-D, got {condition_dim}"
        )

    return {
        "cfg": cfg,
        "condition_dim": condition_dim,
        "action_horizon": int(action_meta.horizon),
        "action_dim": int(action_meta.shape[0]),
        "ft_horizon": int(obs_meta.robot0_ft_left.horizon),
        "ft_dim": int(obs_meta.robot0_ft_left.shape[0]),
        "normalizer_owner": "policy.predict_action",
        "validated_state_names": sorted(policy_states),
        "valve_context_enabled": is_valve_context_policy,
        "valve_context_key": _VALVE_CONTEXT_KEY if is_valve_context_policy else None,
        "valve_context_dim": context_dim,
        "valve_context_schema": context_schema,
        "valve_context_phase_names": context_phase_names,
        "valve_context_num_phase_experts": context_num_experts,
        "architecture_contract_version": architecture_contract_version,
    }
