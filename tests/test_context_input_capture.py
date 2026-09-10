import csv
from types import SimpleNamespace

import cv2
import numpy as np

from eval_real_indy_rg2 import _PolicyInputCapture, _ValveContextInputCapture
from diffusion_policy.common.valve_context_contract import VALVE_CONTEXT_V2_SCHEMA


def _record(timestamp_s=10.02):
    return SimpleNamespace(
        timestamp_s=timestamp_s,
        phase_name="turning",
        error_reason_name="none",
        warmed_up=True,
        values=np.asarray([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
    )


def test_context_capture_preserves_exact_rgb_and_causal_corrected_wrenches(tmp_path):
    capture = _ValveContextInputCapture(
        tmp_path / "context_inputs", episode_start_timestamp_s=10.0
    )
    timestamps = np.asarray([9.99, 10.0, 10.01, 10.02], dtype=np.float64)
    left = np.arange(24, dtype=np.float32).reshape(4, 6) * 0.1
    right = -left
    latest = capture.append_wrenches(
        timestamps,
        left,
        right,
        anchor_timestamp_s=10.02,
    )
    assert latest == 10.02
    # Replaying overlapping stream data must not create duplicate wrench rows.
    assert capture.append_wrenches(
        timestamps, left, right, anchor_timestamp_s=10.02
    ) == 10.02
    assert capture.wrench_count == 3

    rgb = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    capture.append_frame(
        timestamp_s=10.02,
        rgb=rgb,
        position_m=np.asarray([0.1, -0.2, 0.3]),
        rotation_axis_angle_rad=np.asarray([0.01, 0.02, 0.03]),
        gripper_width_m=0.055,
        latest_wrench_timestamp_s=latest,
        context_record=_record(),
    )
    # Startup clamping can select the same first RGB frame repeatedly. The
    # archive must preserve those exact temporal references and tensors.
    capture.append_classifier_window(
        classifier_timestamp_s=10.02,
        model_inputs={
            "rgb_timestamp_s": np.asarray([10.02, 10.02]),
            "lowdim": np.asarray(
                [[0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.055]] * 2,
                dtype=np.float32,
            ),
            "wrench_history_physical": np.arange(
                2 * 3 * 12, dtype=np.float32
            ).reshape(2, 3, 12),
            "wrench_history_model_scaled": np.arange(
                2 * 3 * 12, dtype=np.float32
            ).reshape(2, 3, 12)
            / np.asarray([10, 10, 10, 0.1, 0.1, 0.1] * 2, dtype=np.float32),
            "wrench_mask": np.asarray([[0, 1, 1], [1, 1, 1]], dtype=np.float32),
        },
    )
    capture.close()

    root = tmp_path / "context_inputs"
    with open(root / "context_wrenches.csv", newline="") as file:
        wrench_rows = list(csv.DictReader(file))
    assert [float(row["timestamp_s"]) for row in wrench_rows] == [10.0, 10.01, 10.02]
    assert float(wrench_rows[-1]["left_fz_N"]) == float(left[-1, 2])
    assert float(wrench_rows[-1]["right_tz_Nm"]) == float(right[-1, 5])

    with open(root / "context_frames.csv", newline="") as file:
        frame_rows = list(csv.DictReader(file))
    assert len(frame_rows) == 1
    assert frame_rows[0]["image_file"] == "images/frame_00000000.png"
    assert float(frame_rows[0]["latest_wrench_timestamp_s"]) == 10.02
    assert frame_rows[0]["phase"] == "turning"

    restored_bgr = cv2.imread(str(root / frame_rows[0]["image_file"]), cv2.IMREAD_COLOR)
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(restored_rgb, rgb)

    assert not (root / "context_imu.csv").exists()
    with open(root / "classifier_windows" / "index.csv", newline="") as file:
        window_rows = list(csv.DictReader(file))
    assert window_rows == [
        {
            "window_idx": "0",
            "classifier_timestamp_s": "10.02",
            "npz_file": "classifier_windows/window_00000000.npz",
            "temporal_steps": "2",
            "force_history_samples": "3",
        }
    ]
    with np.load(root / window_rows[0]["npz_file"]) as archive:
        np.testing.assert_array_equal(
            archive["rgb_image_file"],
            np.asarray(["images/frame_00000000.png", "images/frame_00000000.png"]),
        )
        assert archive["lowdim"].shape == (2, 10)
        assert archive["wrench_history_physical"].shape == (2, 3, 12)
        assert archive["wrench_history_model_scaled"].shape == (2, 3, 12)
        np.testing.assert_array_equal(
            archive["wrench_mask"],
            np.asarray([[0, 1, 1], [1, 1, 1]], dtype=np.float32),
        )


def test_policy_input_capture_preserves_exact_predict_action_arrays(tmp_path):
    shape_meta = {
        "obs": {
            "camera0_rgb": {"type": "rgb", "shape": [3, 224, 224]},
            "robot0_eef_pos": {"type": "low_dim", "shape": [3]},
            "robot0_eef_rot_axis_angle": {"type": "low_dim", "shape": [6]},
            "robot0_ft_left": {"type": "low_dim", "shape": [6]},
            "robot0_ft_right": {"type": "low_dim", "shape": [6]},
            "valve_context": {"type": "low_dim", "shape": [10]},
        }
    }
    capture = _PolicyInputCapture(
        tmp_path / "policy_inputs",
        shape_meta=shape_meta,
        episode_start_timestamp_s=10.0,
    )
    rgb = np.zeros((2, 3, 224, 224), dtype=np.float32)
    rgb[0, 0] = 0.25
    rgb[1, 1] = 0.75
    obs_dict_np = {
        "camera0_rgb": rgb,
        "robot0_eef_pos": np.asarray([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.arange(12, dtype=np.float32).reshape(2, 6),
        "robot0_ft_left": np.arange(192, dtype=np.float32).reshape(32, 6),
        "robot0_ft_right": -np.arange(192, dtype=np.float32).reshape(32, 6),
        "valve_context": np.asarray([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float32),
    }
    source_obs = {
        "robot0_ft_left_timestamps": np.linspace(9.7, 10.01, 32),
        "robot0_ft_right_timestamps": np.linspace(9.7, 10.01, 32),
    }
    capture.append(
        policy_iter_idx=4,
        policy_anchor_timestamp_s=10.02,
        obs_dict_np=obs_dict_np,
        source_obs=source_obs,
    )
    capture.close()

    root = tmp_path / "policy_inputs"
    with np.load(root / "samples" / "sample_000000.npz") as archive:
        assert int(archive["policy_iter_idx"]) == 4
        assert float(archive["policy_anchor_timestamp_s"]) == 10.02
        for key, expected in obs_dict_np.items():
            np.testing.assert_array_equal(archive[key], expected)
    with open(root / "index.csv", newline="") as file:
        index_rows = list(csv.DictReader(file))
    assert len(index_rows) == 1
    assert index_rows[0]["camera0_rgb_t0_file"] == (
        "images/sample_000000_camera0_rgb_t0.png"
    )
    restored = cv2.imread(
        str(root / index_rows[0]["camera0_rgb_t1_file"]), cv2.IMREAD_COLOR
    )
    restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(
        restored,
        np.moveaxis((rgb[1] * 255).astype(np.uint8), 0, -1),
    )
    with open(root / "ft_history.csv", newline="") as file:
        ft_rows = list(csv.DictReader(file))
    assert len(ft_rows) == 64
    assert ft_rows[0]["finger"] == "left"
    assert ft_rows[-1]["finger"] == "right"
    assert not (root / "imu.csv").exists()


def test_context_capture_supports_four_state_v2_columns(tmp_path):
    root = tmp_path / "context_v2_inputs"
    capture = _ValveContextInputCapture(
        root,
        episode_start_timestamp_s=10.0,
        context_schema=VALVE_CONTEXT_V2_SCHEMA,
    )
    record = SimpleNamespace(
        timestamp_s=10.02,
        phase_name="recovery",
        error_reason_name="n/a",
        warmed_up=True,
        schema=VALVE_CONTEXT_V2_SCHEMA,
        values=np.asarray([0.05, 0.1, 0.8, 0.05, 1.0], dtype=np.float32),
    )
    capture.append_frame(
        timestamp_s=10.02,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        position_m=np.zeros(3),
        rotation_axis_angle_rad=np.zeros(3),
        gripper_width_m=0.05,
        latest_wrench_timestamp_s=None,
        context_record=record,
    )
    capture.close()

    with open(root / "context_frames.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["phase"] == "recovery"
    assert rows[0]["phase_recovery"] == str(float(record.values[2]))
    assert rows[0]["context_valid"] == "1.0"
    assert "reason_none" not in rows[0]
