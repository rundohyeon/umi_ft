import numpy as np
import pytest

from umi.real_world.grasp_force_width_feedback import (
    GraspForceWidthFeedbackConfig,
    GraspForceWidthFeedbackController,
    signed_grasp_force_from_native_wrenches,
)


def _controller():
    return GraspForceWidthFeedbackController(
        GraspForceWidthFeedbackConfig()
    )


def test_feedback_opens_when_measured_force_is_above_reference():
    result = _controller().correct(
        policy_width_m=np.asarray([0.05]),
        predicted_force_n=np.asarray([6.0]),
        measured_force_n=8.0,
    )
    np.testing.assert_allclose(result["force_error_n"], [2.0])
    np.testing.assert_allclose(result["width_correction_m"], [0.0002])
    np.testing.assert_allclose(result["corrected_width_m"], [0.0502])


def test_feedback_closes_when_measured_force_is_below_reference():
    result = _controller().correct(
        policy_width_m=np.asarray([0.05]),
        predicted_force_n=np.asarray([6.0]),
        measured_force_n=4.0,
    )
    np.testing.assert_allclose(result["width_correction_m"], [-0.0002])
    np.testing.assert_allclose(result["corrected_width_m"], [0.0498])


def test_feedback_deadband_and_bounds_are_enforced():
    deadband = _controller().correct(
        policy_width_m=np.asarray([0.05]),
        predicted_force_n=np.asarray([6.0]),
        measured_force_n=6.5,
    )
    np.testing.assert_array_equal(deadband["width_correction_m"], [0.0])

    bounded = _controller().correct(
        policy_width_m=np.asarray([0.0998, 0.0002]),
        predicted_force_n=np.asarray([-100.0, 100.0]),
        measured_force_n=20.0,
    )
    np.testing.assert_allclose(bounded["predicted_force_n"], [0.0, 12.0])
    np.testing.assert_allclose(bounded["width_correction_m"], [0.001, 0.0008])
    np.testing.assert_allclose(bounded["corrected_width_m"], [0.1, 0.001])


def test_native_wrench_measurement_requires_explicit_startup_bias():
    left = np.asarray([0, 0, 3, 0, 0, 0], dtype=np.float64)
    right = np.asarray([0, 0, 13, 0, 0, 0], dtype=np.float64)
    bias = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    assert signed_grasp_force_from_native_wrenches(left, right, bias) == 5.0

    result = _controller().correct_from_native_wrenches(
        policy_width_m=np.asarray([0.05]),
        predicted_force_n=np.asarray([6.0]),
        left_wrench=left,
        right_wrench=right,
        startup_bias_12d=bias,
    )
    assert result["measured_force_n"] == 5.0
    np.testing.assert_allclose(result["corrected_width_m"], [0.0499])

    with pytest.raises(ValueError, match=r"shape \[12\]"):
        signed_grasp_force_from_native_wrenches(left, right, np.zeros(6))


def test_feedback_rejects_nonfinite_or_mismatched_trajectories():
    with pytest.raises(ValueError, match="same shape"):
        _controller().correct(np.zeros(2), np.zeros(1), 0.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        _controller().correct(np.asarray([np.nan]), np.zeros(1), 0.0)
    with pytest.raises(ValueError, match="finite"):
        _controller().correct(np.zeros(1), np.zeros(1), np.inf)
