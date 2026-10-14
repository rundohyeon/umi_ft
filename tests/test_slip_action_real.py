import numpy as np

from eval_real_slip_action import (
    DEFAULT_ACTION_CHECKPOINT,
    DEFAULT_CONTEXT_CHECKPOINT,
    ContextProbabilityMomentum,
    apply_policy_motion_momentum,
    assess_result_age,
    assemble_action_observation,
    build_parser,
    compose_teleop_rotvec,
    future_action_rows,
    rgb_uint8,
    scale_pose_targets,
    select_context_update_anchors,
    select_context_updates_for_runtime,
    teleop_delta_for_key,
)


def test_default_checkpoints_use_installed_contextft4_bundle():
    assert DEFAULT_ACTION_CHECKPOINT.as_posix().endswith(
        "/data/contextFT-4/epoch=0008-val_loss=0.02815.ckpt"
    )
    assert DEFAULT_CONTEXT_CHECKPOINT.as_posix().endswith(
        "/data/contextFT-4/best.pt"
    )


def test_live_observation_preserves_paired_rgb_force_contract():
    frames = np.zeros((70, 224, 224, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(70, dtype=np.uint8)
    left = np.arange(100 * 6, dtype=np.float32).reshape(100, 6)
    right = left + 1000
    rgb = np.array([3, 23, 43, 63])
    force = np.stack([np.arange(end - 40, end + 1) for end in (40, 55, 70, 90)])
    obs = assemble_action_observation(
        {
            "camera0_rgb": frames,
            "robot0_ft_left": left,
            "robot0_ft_right": right,
        },
        rgb,
        force,
        [0.1, 0.2, 0.3, 0.4],
    )

    assert obs["camera0_rgb"].shape == (4, 3, 224, 224)
    assert obs["robot0_ft_left"].shape == (4, 41, 6)
    assert obs["robot0_ft_right"].shape == (4, 41, 6)
    assert obs["valve_context"].shape == (1, 5)
    np.testing.assert_allclose(obs["camera0_rgb"][:, 0, 0, 0], rgb / 255)
    np.testing.assert_array_equal(obs["robot0_ft_left"][:, -1], left[force[:, -1]])
    np.testing.assert_array_equal(obs["robot0_ft_right"][:, 0], right[force[:, 0]])
    np.testing.assert_allclose(obs["valve_context"], [[0.1, 0.2, 0.3, 0.4, 1]])


def test_future_action_rows_never_commands_current_row_zero():
    action = np.arange(16 * 11, dtype=np.float32).reshape(16, 11)
    pose_width, force = future_action_rows(action, 2)

    np.testing.assert_array_equal(pose_width, action[1:3, :10])
    np.testing.assert_array_equal(force, action[1:3, 10])


def test_action_scale_uses_current_tcp_as_fixed_anchor():
    current = np.array([0.5, -0.2, 0.1, 0.1, -0.2, 0.3])
    targets = np.array(
        [
            [0.51, -0.18, 0.07, 0.2, -0.1, 0.4, 0.04],
            [0.54, -0.22, 0.11, 0.3, -0.3, 0.5, 0.05],
        ]
    )
    scaled = scale_pose_targets(targets, current, 0.25, freeze_rotation=True)

    np.testing.assert_allclose(
        scaled[:, :3], current[:3] + 0.25 * (targets[:, :3] - current[:3])
    )
    np.testing.assert_allclose(scaled[:, 3:6], np.tile(current[3:6], (2, 1)))
    np.testing.assert_allclose(scaled[:, 6], targets[:, 6])


def test_motion_momentum_uses_established_one_to_two_ratio():
    current = np.array([0, 0, 0, 0, 0, 0, 0.04], dtype=np.float64)
    target = np.array([[0.03, -0.02, 0.01, 0, 0, 0.12, 0.055]])

    blended = apply_policy_motion_momentum(
        target,
        current_pose_width7=current,
        previous_weight=1.0 / 3.0,
    )

    np.testing.assert_allclose(blended[0, :3], target[0, :3] * (2.0 / 3.0))
    np.testing.assert_allclose(blended[0, 5], 0.08, atol=1e-12)
    np.testing.assert_allclose(blended[0, 6], 0.05, atol=1e-12)


def test_context_probability_momentum_blends_and_resets_without_hard_labels():
    momentum = ContextProbabilityMomentum(previous_weight=0.5)
    first = momentum.apply(
        {"probabilities": [0.8, 0.1, 0.05, 0.05], "class_id": 0, "class_name": "approach"}
    )
    second = momentum.apply(
        {"probabilities": [0.1, 0.7, 0.1, 0.1], "class_id": 1, "class_name": "error1"}
    )

    np.testing.assert_allclose(first["probabilities"], [0.8, 0.1, 0.05, 0.05])
    np.testing.assert_allclose(second["model_probabilities"], [0.1, 0.7, 0.1, 0.1])
    np.testing.assert_allclose(second["probabilities"], [0.45, 0.4, 0.075, 0.075])
    assert second["class_name"] == "approach"

    momentum.reset()
    after_reset = momentum.apply(
        {"probabilities": [0.1, 0.7, 0.1, 0.1], "class_id": 1, "class_name": "error1"}
    )
    np.testing.assert_allclose(after_reset["probabilities"], [0.1, 0.7, 0.1, 0.1])
    assert after_reset["class_name"] == "error1"


def test_established_deploy_option_names_map_to_new_runner():
    args = build_parser().parse_args(
        [
            "--steps_per_inference",
            "4",
            "--show_policy_image",
            "--policy_input_audit",
            "--print_policy_output",
            "--zero_ft_on_start",
            "--ft_max_age_sec",
            "0.020",
            "--latest_ft_max_age_sec",
            "0.050",
            "--wait_for_start_key",
            "--execute",
            "--motion_momentum_previous_weight",
            "0.333333",
            "--context_momentum_previous_weight",
            "0.5",
        ]
    )

    assert args.steps_per_inference == 4
    assert args.show_policy_image
    assert args.policy_input_audit
    assert args.print_policy_output
    assert args.zero_ft_on_start
    assert args.ft_max_age_sec == 0.020
    assert args.latest_ft_max_age_sec == 0.050
    assert args.wait_for_start_key
    assert args.execute
    assert np.isclose(args.motion_momentum_previous_weight, 0.333333)
    assert np.isclose(args.context_momentum_previous_weight, 0.5)


def test_result_age_separates_expected_camera_latency_from_processing():
    observed = assess_result_age(
        0.354,
        camera_latency_s=0.125,
        max_total_age_s=0.400,
        max_processing_age_s=0.250,
    )
    assert observed["is_fresh"]
    assert np.isclose(observed["processing_age_s"], 0.229)

    total_too_old = assess_result_age(
        0.401,
        camera_latency_s=0.200,
        max_total_age_s=0.400,
        max_processing_age_s=0.250,
    )
    assert not total_too_old["is_fresh"]
    assert total_too_old["stale_reasons"] == ["total_age"]

    processing_too_old = assess_result_age(
        0.390,
        camera_latency_s=0.125,
        max_total_age_s=0.400,
        max_processing_age_s=0.250,
    )
    assert not processing_too_old["is_fresh"]
    assert processing_too_old["stale_reasons"] == ["processing_age"]


def test_context_cadence_backfills_frames_missed_during_action_inference():
    rgb_times = np.arange(0.0, 1.251, 0.025)
    anchors = select_context_update_anchors(
        rgb_times,
        0.9,
        period_s=0.1001,
    )
    np.testing.assert_allclose(anchors, [1.0, 1.1, 1.2], atol=1e-12)

    first = select_context_update_anchors(
        rgb_times,
        None,
        period_s=0.1001,
    )
    np.testing.assert_allclose(first, [1.25], atol=1e-12)

    # Repeatedly entering the action loop at ~3-4 Hz must not let small camera
    # timestamp errors accumulate relative to the classifier's 0.1001 s grid.
    rgb_times = np.arange(0.0, 4.0, 0.025)
    last = None
    next_target = None
    selected = []
    for latest in (0.90, 1.15, 1.40, 1.68, 1.92, 2.20, 2.49, 2.78, 3.07):
        available = rgb_times[rgb_times <= latest]
        update = select_context_update_anchors(
            available,
            last,
            period_s=0.1001,
            next_target_timestamp=next_target,
        )
        selected.extend(update.tolist())
        last = float(update[-1])
        if next_target is None:
            next_target = last + 0.1001
        else:
            next_target += len(update) * 0.1001
    selected = np.asarray(selected)
    ideal = selected[0] + np.arange(len(selected)) * 0.1001
    assert np.max(np.abs(selected - ideal)) <= 0.013
    assert np.max(np.abs(np.diff(selected) - 0.1001)) <= 0.025


def test_context_cadence_miss_resyncs_to_latest_frame_instead_of_stalling():
    class Runtime:
        class Model:
            history_period_s = 0.1001

        model = Model()
        last_timestamp = 0.9
        next_update_target_timestamp = 1.0

        def __init__(self):
            self.resets = 0

        def reset_history(self):
            self.resets += 1
            self.last_timestamp = None
            self.next_update_target_timestamp = None

    runtime = Runtime()
    anchors, resynced = select_context_updates_for_runtime(
        np.array([0.80, 0.90, 1.03]), runtime
    )

    np.testing.assert_allclose(anchors, [1.03])
    assert resynced
    assert runtime.resets == 1


def test_rgb_uint8_preserves_rgb_channel_order():
    chw = np.zeros((3, 2, 2), dtype=np.float32)
    chw[0] = 1.0
    image = rgb_uint8(chw)

    assert image.shape == (2, 2, 3)
    np.testing.assert_array_equal(image[0, 0], [255, 0, 0])


def test_teleop_keys_match_established_indy_controls():
    dpos, drot, grip = teleop_delta_for_key(
        ord("q"),
        position_step_m=0.005,
        rotation_step_rad=0.05,
        gripper_step_m=0.01,
    )
    np.testing.assert_allclose(dpos, [0, 0, -0.005])
    np.testing.assert_allclose(drot, [0, 0, 0])
    assert grip == 0

    dpos, drot, grip = teleop_delta_for_key(
        ord("l"),
        position_step_m=0.005,
        rotation_step_rad=0.05,
        gripper_step_m=0.01,
    )
    np.testing.assert_allclose(dpos, [0, 0, 0])
    np.testing.assert_allclose(drot, [0.05, 0, 0])
    assert grip == 0

    _, _, grip = teleop_delta_for_key(
        ord("z"),
        position_step_m=0.005,
        rotation_step_rad=0.05,
        gripper_step_m=0.01,
    )
    assert grip == -0.01


def test_teleop_rotation_composes_increment_with_current_rotvec():
    result = compose_teleop_rotvec(
        np.zeros(3),
        [0, 0, 0.05],
        euler_sequence="xyz",
        extrinsic=False,
    )
    np.testing.assert_allclose(result, [0, 0, 0.05], atol=1e-12)
