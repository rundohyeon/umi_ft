import numpy as np
import pytest

from umi.real_world.dual_ft_policy_safety import (
    FTSafetyConfig,
    PolicyMotionSafetyConfig,
    PolicySafetyError,
    validate_ft_load,
    validate_policy_waypoints,
)


def _motion_cfg():
    return PolicyMotionSafetyConfig(
        max_position_delta_m=0.02,
        max_rotation_delta_rad=0.2,
        max_gripper_delta_m=0.02,
        min_tcp_z_m=-1.0,
        exclusion_sphere_radius_m=0.01,
        exclusion_sphere_center_m=(10.0, 10.0, 10.0),
    )


def test_policy_waypoint_guard_checks_every_consecutive_delta():
    current = np.zeros(6)
    valid = np.asarray([[0.01, 0, 0, 0, 0, 0.1, 0.05], [0.02, 0, 0, 0, 0, 0.2, 0.06]])
    np.testing.assert_array_equal(
        validate_policy_waypoints(valid, current, 0.04, _motion_cfg()), valid
    )

    invalid = valid.copy()
    invalid[1, 0] = 0.04
    with pytest.raises(PolicySafetyError, match="position delta"):
        validate_policy_waypoints(invalid, current, 0.04, _motion_cfg())


def test_ft_guard_is_fail_closed_for_force_torque_and_grasp_load():
    cfg = FTSafetyConfig(10.0, 1.0, 5.0, 0.05)
    validate_ft_load(
        np.zeros(6), np.zeros(6), 0.0, cfg, latest_sample_age_s=0.01
    )
    left = np.zeros(6)
    left[0] = 11.0
    with pytest.raises(PolicySafetyError, match="F/T overload"):
        validate_ft_load(left, np.zeros(6), 0.0, cfg)
    with pytest.raises(PolicySafetyError, match="grasp overload"):
        validate_ft_load(np.zeros(6), np.zeros(6), 6.0, cfg)
    with pytest.raises(PolicySafetyError, match="stale"):
        validate_ft_load(
            np.zeros(6), np.zeros(6), 0.0, cfg, latest_sample_age_s=0.06
        )


def test_motion_guard_checks_segment_through_exclusion_sphere():
    cfg = PolicyMotionSafetyConfig(
        max_position_delta_m=1.0,
        max_rotation_delta_rad=1.0,
        max_gripper_delta_m=0.1,
        min_tcp_z_m=-1.0,
        exclusion_sphere_radius_m=0.1,
        exclusion_sphere_center_m=(0.0, 0.0, 0.0),
    )
    current = np.asarray([-0.2, 0.0, 0.2, 0, 0, 0], dtype=np.float64)
    target = np.asarray([[0.2, 0.0, 0.2, 0, 0, 0, 0.05]])
    validate_policy_waypoints(target, current, 0.05, cfg)
    target[0, 2] = 0.0
    current[2] = 0.0
    with pytest.raises(PolicySafetyError, match="segment"):
        validate_policy_waypoints(target, current, 0.05, cfg)
