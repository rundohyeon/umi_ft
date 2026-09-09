import numpy as np
import scipy.spatial.transform as st

from eval_real_indy_rg2 import _apply_policy_motion_momentum


def _rotation_distance(rotvec_a, rotvec_b) -> float:
    return float(
        (
            st.Rotation.from_rotvec(rotvec_a)
            * st.Rotation.from_rotvec(rotvec_b).inv()
        ).magnitude()
    )


def test_motion_momentum_blends_tcp_delta_at_requested_one_to_two_ratio():
    current_tcp6 = np.zeros(6, dtype=np.float64)
    # This is the rotation delta that previously exceeded the 0.1 rad safety
    # gate. With no prior movement, a 1:2 previous:policy blend is 2/3 of it.
    target = np.asarray([[0.03, -0.02, 0.01, 0.0, 0.0, 0.11349, 0.055]])
    output, next_target, next_delta = _apply_policy_motion_momentum(
        target,
        current_tcp6=current_tcp6,
        previous_sent_target_tcp6=None,
        previous_sent_delta_tcp6=None,
        previous_weight=1.0 / 3.0,
    )
    np.testing.assert_allclose(output[0, :3], target[0, :3] * (2.0 / 3.0))
    assert np.isclose(
        _rotation_distance(output[0, 3:6], current_tcp6[3:6]),
        0.11349 * (2.0 / 3.0),
        atol=1e-9,
    )
    np.testing.assert_allclose(next_target, output[-1, :6])
    np.testing.assert_allclose(next_delta, output[-1, :6])
    # F/T width remains untouched by TCP momentum filtering.
    assert output[0, 6] == target[0, 6]


def test_motion_momentum_state_persists_without_contact_or_watchdog_reset():
    current_tcp6 = np.zeros(6, dtype=np.float64)
    first = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.055]])
    _, last_target, last_delta = _apply_policy_motion_momentum(
        first,
        current_tcp6=current_tcp6,
        previous_sent_target_tcp6=None,
        previous_sent_delta_tcp6=None,
        previous_weight=1.0 / 3.0,
    )
    # A policy target that equals the last transmitted target has a zero new
    # delta, but retained momentum continues it by one third of the prior one.
    second = np.concatenate([last_target, [0.055]])[None]
    output, _, next_delta = _apply_policy_motion_momentum(
        second,
        current_tcp6=current_tcp6,
        previous_sent_target_tcp6=last_target,
        previous_sent_delta_tcp6=last_delta,
        previous_weight=1.0 / 3.0,
    )
    assert np.isclose(
        _rotation_distance(output[0, 3:6], last_target[3:6]),
        0.09 * (2.0 / 3.0) * (1.0 / 3.0),
        atol=1e-9,
    )
    assert np.isclose(np.linalg.norm(next_delta[3:6]), 0.09 * 2.0 / 9.0)
