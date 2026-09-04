from types import SimpleNamespace
import time

import numpy as np

from eval_real_indy_rg2 import _ValveContextWorker


class _FakeContextRuntime:
    def __init__(self):
        self.last_prediction_timestamp = -np.inf
        self.calls = []

    def reset(self, *, episode_start_timestamp_s=None):
        self.last_prediction_timestamp = -np.inf
        self.episode_start_timestamp_s = episode_start_timestamp_s

    def predict(self, **kwargs):
        timestamp_s = float(kwargs["timestamp_s"])
        self.calls.append(kwargs)
        self.last_prediction_timestamp = timestamp_s
        return SimpleNamespace(
            timestamp_s=timestamp_s,
            values=np.asarray([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            phase_name="turning",
            error_reason_name="none",
            warmed_up=False,
        )


def test_context_worker_replays_all_unseen_camera_frames_and_matches_policy_anchor():
    timestamps = np.asarray([9.99, 10.0, 10.016, 10.032], dtype=np.float64)
    rgb = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    rgb[1, ..., 0] = 10
    rgb[2, ..., 0] = 20
    rgb[3, ..., 0] = 30
    stream = {
        "rgb_timestamp_s": timestamps,
        "camera_step_idx": np.asarray([100, 101, 102, 103], dtype=np.int64),
        "camera0_rgb": rgb,
        "robot0_eef_pos": np.zeros((4, 3), dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros((4, 3), dtype=np.float32),
        "robot0_gripper_width": np.full((4, 1), 0.055, dtype=np.float32),
        "ft_timestamp_s": np.asarray([9.99, 10.005, 10.015, 10.025], dtype=np.float64),
        "robot0_ft_left": np.zeros((4, 6), dtype=np.float32),
        "robot0_ft_right": np.zeros((4, 6), dtype=np.float32),
    }
    runtime = _FakeContextRuntime()
    newest_read_count = 0

    def stream_provider(*, history_frames):
        """Simulate a 3-frame camera jump between two 60 Hz worker polls."""
        nonlocal newest_read_count
        if history_frames > 1:
            return stream
        frame_idx = 0 if newest_read_count == 0 else 3
        newest_read_count += 1
        return {
            key: value[frame_idx:frame_idx + 1]
            for key, value in stream.items()
            if key not in {"ft_timestamp_s", "robot0_ft_left", "robot0_ft_right"}
        } | {
            "ft_timestamp_s": stream["ft_timestamp_s"],
            "robot0_ft_left": stream["robot0_ft_left"],
            "robot0_ft_right": stream["robot0_ft_right"],
        }

    worker = _ValveContextWorker(runtime, stream_provider, poll_hz=100.0)
    worker.start(episode_start_timestamp_s=10.0)
    try:
        record = worker.record_for_policy_anchor(
            timestamp_s=10.032,
            rgb=rgb[3],
            timeout_s=1.0,
        )
        assert record.timestamp_s == 10.032
        assert [call["timestamp_s"] for call in runtime.calls] == [10.0, 10.016, 10.032]
        # Each RGB uses no newer-than-anchor force samples.
        for call in runtime.calls:
            assert np.all(call["ft_timestamps"] <= call["timestamp_s"])
        summary = worker.summary()
        assert summary["processed_frames"] == 3
        assert np.isclose(summary["max_frame_gap_s"], 0.016)
        assert summary["recovery_polls"] == 0
        assert summary["anchor_recovery_polls"] == 1
    finally:
        worker.stop()


def test_context_worker_does_not_recover_for_normal_lower_camera_fps_step_jumps():
    """A 40 Hz source requested as 60 Hz often advances step_idx by two."""
    timestamps = np.asarray([10.000, 10.025, 10.050, 10.075], dtype=np.float64)
    rgb = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    for idx in range(4):
        rgb[idx, ..., 0] = 10 * idx
    stream = {
        "rgb_timestamp_s": timestamps,
        "camera_step_idx": np.asarray([100, 102, 103, 105], dtype=np.int64),
        "camera0_rgb": rgb,
        "robot0_eef_pos": np.zeros((4, 3), dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros((4, 3), dtype=np.float32),
        "robot0_gripper_width": np.full((4, 1), 0.055, dtype=np.float32),
        "ft_timestamp_s": np.asarray([9.99, 10.02, 10.04, 10.06], dtype=np.float64),
        "robot0_ft_left": np.zeros((4, 6), dtype=np.float32),
        "robot0_ft_right": np.zeros((4, 6), dtype=np.float32),
    }
    next_frame_idx = 0

    def stream_provider(*, history_frames):
        nonlocal next_frame_idx
        if history_frames > 1:
            raise AssertionError("normal camera cadence must not trigger recovery")
        frame_idx = min(next_frame_idx, len(timestamps) - 1)
        next_frame_idx += 1
        return {
            key: value[frame_idx:frame_idx + 1]
            for key, value in stream.items()
            if key not in {"ft_timestamp_s", "robot0_ft_left", "robot0_ft_right"}
        } | {
            "ft_timestamp_s": stream["ft_timestamp_s"],
            "robot0_ft_left": stream["robot0_ft_left"],
            "robot0_ft_right": stream["robot0_ft_right"],
        }

    runtime = _FakeContextRuntime()
    worker = _ValveContextWorker(runtime, stream_provider, poll_hz=200.0)
    worker.start(episode_start_timestamp_s=10.0)
    try:
        deadline = time.monotonic() + 1.0
        while len(runtime.calls) < len(timestamps) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert [call["timestamp_s"] for call in runtime.calls] == timestamps.tolist()
        record = worker.record_for_policy_anchor(
            timestamp_s=10.075,
            rgb=rgb[-1],
            timeout_s=1.0,
        )
        assert record.timestamp_s == 10.075
        assert [call["timestamp_s"] for call in runtime.calls] == timestamps.tolist()
        assert worker.summary()["recovery_polls"] == 0
        assert worker.summary()["anchor_recovery_polls"] == 0
    finally:
        worker.stop()
