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


def test_motion_momentum_blends_tcp_and_gripper_at_one_to_two_ratio():
    current_pose_width7 = np.asarray([0, 0, 0, 0, 0, 0, 0.04], dtype=np.float64)
    # This is the rotation delta that previously exceeded the 0.1 rad safety
    # gate. With no prior movement, a 1:2 previous:policy blend is 2/3 of it.
    target = np.asarray([[0.03, -0.02, 0.01, 0.0, 0.0, 0.11349, 0.055]])
    output = _apply_policy_motion_momentum(
        target,
        current_pose_width7=current_pose_width7,
        previous_weight=1.0 / 3.0,
    )
    np.testing.assert_allclose(output[0, :3], target[0, :3] * (2.0 / 3.0))
    assert np.isclose(
        _rotation_distance(output[0, 3:6], current_pose_width7[3:6]),
        0.11349 * (2.0 / 3.0),
        atol=1e-9,
    )
    assert output[0, 6] == np.float64(0.05)


def test_tcp_momentum_is_retained_only_inside_current_horizon():
    current_pose_width7 = np.asarray([0, 0, 0, 0, 0, 0, 0.05], dtype=np.float64)
    targets = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.056],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.056],
        ]
    )
    output = _apply_policy_motion_momentum(
        targets,
        current_pose_width7=current_pose_width7,
        previous_weight=1.0 / 3.0,
    )
    assert np.isclose(_rotation_distance(output[0, 3:6], [0, 0, 0]), 0.06)
    assert np.isclose(_rotation_distance(output[1, 3:6], [0, 0, 0]), 0.10)
    np.testing.assert_allclose(output[:, 6], [0.054, 0.054])


def test_gripper_smoothing_reproduces_ep8_without_velocity_overshoot():
    current = np.asarray([0, 0, 0, 0, 0, 0, 0.03772], dtype=np.float64)
    # A large retained width velocity previously pushed waypoint 1 above both
    # its policy target and the 15 mm safety gate.
    targets = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0.0499016866],
            [0, 0, 0, 0, 0, 0, 0.0522849448],
        ],
        dtype=np.float64,
    )

    output = _apply_policy_motion_momentum(
        targets,
        current_pose_width7=current,
        previous_weight=1.0 / 3.0,
    )

    for waypoint_idx in range(len(targets)):
        assert output[waypoint_idx, 6] <= max(
            current[6], targets[waypoint_idx, 6]
        )
        assert output[waypoint_idx, 6] >= min(
            current[6], targets[waypoint_idx, 6]
        )
    assert output[1, 6] - current[6] < 0.015


def test_gripper_smoothing_reproduces_ep17_without_unexecuted_horizon_carryover():
    current = np.asarray([0, 0, 0, 0, 0, 0, 0.03061], dtype=np.float64)
    targets = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0.04570326954126358],
            [0, 0, 0, 0, 0, 0, 0.04677128791809082],
            [0, 0, 0, 0, 0, 0, 0.04869627580046654],
            [0, 0, 0, 0, 0, 0, 0.05002998560667038],
        ],
        dtype=np.float64,
    )

    output = _apply_policy_motion_momentum(
        targets,
        current_pose_width7=current,
        previous_weight=1.0 / 3.0,
    )

    expected_widths = current[6] / 3.0 + targets[:, 6] * (2.0 / 3.0)
    np.testing.assert_allclose(output[:, 6], expected_widths)
    assert np.max(np.abs(output[:, 6] - current[6])) < 0.015
    # The old implementation produced 0.04854 m for waypoint 0 by anchoring
    # to the previous horizon's unexecuted final target.
    assert output[0, 6] < 0.04854


def test_tcp_smoothing_reproduces_ep22_without_unexecuted_horizon_carryover():
    current = np.asarray(
        [
            0.03909474721608728,
            -0.828103271699786,
            0.3315350642005303,
            1.3972295989339623,
            1.642585083153258,
            -0.6577903516154574,
            0.05578422249597271,
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [
                0.03947850240006977,
                -0.8284051081959055,
                0.33120951462872344,
                1.3984820909712214,
                1.6415867025642858,
                -0.657007761318131,
                0.05210695415735245,
            ],
            [
                0.040539952732241556,
                -0.8279229609744304,
                0.33178333195268994,
                1.395938107239689,
                1.637497550464306,
                -0.6559718407083481,
                0.05675865337252617,
            ],
            [
                0.0406788591473732,
                -0.8273130810761824,
                0.33272223884815266,
                1.3950725707307348,
                1.634617751321805,
                -0.6567070338433723,
                0.05797920376062393,
            ],
            [
                0.04299621507478691,
                -0.8267696887896174,
                0.33245630171783014,
                1.3926727434478872,
                1.636146606753337,
                -0.656994078501694,
                0.05937926098704338,
            ],
        ],
        dtype=np.float64,
    )

    output = _apply_policy_motion_momentum(
        targets,
        current_pose_width7=current,
        previous_weight=0.5,
    )

    rotation_deltas = [
        _rotation_distance(pose[3:6], current[3:6]) for pose in output
    ]
    np.testing.assert_allclose(
        rotation_deltas,
        [0.0007162119, 0.0030768203, 0.0068321163, 0.0091737316],
        atol=1e-8,
    )
    assert max(rotation_deltas) < 0.1
    # The old cross-cycle carryover produced 0.10658060 rad at waypoint 0.
    assert rotation_deltas[0] < 0.001
