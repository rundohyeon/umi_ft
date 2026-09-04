import csv
from types import SimpleNamespace

import cv2
import numpy as np

from eval_real_indy_rg2 import _ValveContextInputCapture


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

    with open(root / "context_imu.csv", newline="") as file:
        imu_rows = list(csv.DictReader(file))
    assert imu_rows == [
        {
            "frame_idx": "0",
            "rgb_timestamp_s": "10.02",
            "imu_available": "0",
            "imu_source": "not_available_over_hdmi_elgato",
            "imu_timestamp_s": "",
            "accel_x_m_s2": "",
            "accel_y_m_s2": "",
            "accel_z_m_s2": "",
            "gyro_x_rad_s": "",
            "gyro_y_rad_s": "",
            "gyro_z_rad_s": "",
        }
    ]
