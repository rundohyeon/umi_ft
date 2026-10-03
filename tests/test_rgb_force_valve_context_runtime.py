from pathlib import Path

import numpy as np
import pytest

import umi.real_world.valve_state_context as runtime_module
from diffusion_policy.common.valve_context_contract import VALVE_CONTEXT_V2_SCHEMA
from umi.real_world.rgb_force_context import ObservationUnavailable


class _FakeRGBForceRuntime:
    def __init__(self, checkpoint, device):
        self.metadata = {
            "schema": "context_rgb_force_4state_ft_features_v2",
            "phase_names": ["approach", "turning", "recovery", "error"],
        }
        self.device = device
        self.fail_reason = None
        self.predict_calls = 0

    def prepare(
        self,
        rgb_times,
        rgb_frames,
        ft_times,
        wrench_12d,
        *,
        episode_start,
        rgb_target_span_s,
    ):
        assert rgb_target_span_s == 3 / 60
        if self.fail_reason is not None:
            raise ObservationUnavailable(self.fail_reason)
        if len(rgb_times) < 4:
            raise ObservationUnavailable("warming_up_rgb")
        if len(ft_times) < 3:
            raise ObservationUnavailable("warming_up_ft")
        rgb_indices = np.asarray([-4, -1])
        ft_indices = np.asarray([-3, -2, -1])
        obs = {
            "camera0_rgb": np.asarray(rgb_frames)[rgb_indices],
            "robot0_ft_left": np.asarray(wrench_12d)[ft_indices, :6],
            "robot0_ft_right": np.asarray(wrench_12d)[ft_indices, 6:],
        }
        timing = {
            "rgb_timestamps": np.asarray(rgb_times)[rgb_indices].tolist(),
            "ft_timestamps": np.asarray(ft_times)[ft_indices].tolist(),
        }
        return obs, timing

    def predict(self, obs):
        self.predict_calls += 1
        assert obs["camera0_rgb"].shape == (2, 224, 224, 3)
        return {"probabilities": [0.1, 0.2, 0.6, 0.1]}


def _predict(runtime, timestamp_s, ft_times, left, right, fill):
    return runtime.predict(
        timestamp_s=timestamp_s,
        rgb=np.full((224, 224, 3), fill, dtype=np.uint8),
        position_m=np.full(3, np.nan),
        rotation_axis_angle_rad=np.full(3, np.nan),
        gripper_width_m=float("nan"),
        ft_timestamps=ft_times[ft_times <= timestamp_s],
        ft_left=left[ft_times <= timestamp_s],
        ft_right=right[ft_times <= timestamp_s],
    )


def test_rgb_force_runtime_maps_same_four_state_output_and_validity(
    monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "best_context_dkim.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        runtime_module, "RGBForceContextRuntime", _FakeRGBForceRuntime
    )
    runtime = runtime_module.RGBForceValveContextRuntime(
        checkpoint, device="cpu"
    )
    runtime.reset(episode_start_timestamp_s=0.5)

    ft_times = np.arange(0.5, 1.061, 0.01, dtype=np.float64)
    left = np.zeros((len(ft_times), 6), dtype=np.float32)
    right = np.zeros_like(left)
    right[:, 2] = np.arange(len(ft_times), dtype=np.float32)
    timestamps = [1.0, 1.016, 1.033, 1.05]

    for index, timestamp_s in enumerate(timestamps[:-1]):
        record = _predict(
            runtime, timestamp_s, ft_times, left, right, fill=index
        )
        np.testing.assert_allclose(
            record.values, [0.25, 0.25, 0.25, 0.25, 0.0]
        )
        assert not record.warmed_up

    record = _predict(
        runtime, timestamps[-1], ft_times, left, right, fill=3
    )
    assert record.schema == VALVE_CONTEXT_V2_SCHEMA
    assert record.phase_name == "recovery"
    assert record.warmed_up
    np.testing.assert_allclose(record.values, [0.1, 0.2, 0.6, 0.1, 1.0])

    snapshot = runtime.get_last_classifier_model_inputs()
    assert str(snapshot["observer_input_schema"]) == (
        "context_rgb_force_4state_ft_features_v2"
    )
    assert snapshot["camera0_rgb"].shape == (2, 224, 224, 3)
    assert snapshot["robot0_ft_left"].shape == (3, 6)
    assert snapshot["robot0_ft_right"].shape == (3, 6)
    assert snapshot["rgb_timestamp_s"].tolist() == [1.0, 1.05]
    assert snapshot["ft_timestamp_s"][-1] <= timestamps[-1]
    assert runtime.checkpoint_metadata["integration_rgb_target_span_s"] == 0.05


def test_rgb_force_runtime_rejects_future_force(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "best_context_dkim.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        runtime_module, "RGBForceContextRuntime", _FakeRGBForceRuntime
    )
    runtime = runtime_module.RGBForceValveContextRuntime(
        checkpoint, device="cpu"
    )
    try:
        runtime.predict(
            timestamp_s=1.0,
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            position_m=np.zeros(3),
            rotation_axis_angle_rad=np.zeros(3),
            gripper_width_m=0.05,
            ft_timestamps=np.asarray([1.01]),
            ft_left=np.zeros((1, 6), dtype=np.float32),
            ft_right=np.zeros((1, 6), dtype=np.float32),
        )
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("future F/T was accepted")


def test_rgb_force_runtime_bounds_cadence_rewarm_and_recovers(
    monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "best_context_dkim.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        runtime_module, "RGBForceContextRuntime", _FakeRGBForceRuntime
    )
    runtime = runtime_module.RGBForceValveContextRuntime(
        checkpoint, device="cpu"
    )
    ft_times = np.arange(0.5, 1.5, 0.01, dtype=np.float64)
    left = np.zeros((len(ft_times), 6), dtype=np.float32)
    right = np.zeros_like(left)
    for index, timestamp_s in enumerate([1.0, 1.016, 1.033, 1.05]):
        record = _predict(
            runtime, timestamp_s, ft_times, left, right, fill=index
        )
    assert record.warmed_up

    runtime._runtime.fail_reason = "rgb_cadence_mismatch"
    record = _predict(runtime, 1.08, ft_times, left, right, fill=10)
    assert not record.warmed_up
    np.testing.assert_allclose(
        record.values, [0.25, 0.25, 0.25, 0.25, 0.0]
    )
    assert runtime.last_unavailable_reason == "rgb_cadence_mismatch"

    runtime._runtime.fail_reason = None
    assert _predict(runtime, 1.11, ft_times, left, right, fill=11).warmed_up

    runtime._runtime.fail_reason = "rgb_cadence_mismatch"
    for index, timestamp_s in enumerate([1.14, 1.17, 1.20]):
        assert not _predict(
            runtime, timestamp_s, ft_times, left, right, fill=12 + index
        ).warmed_up
    try:
        _predict(runtime, 1.23, ft_times, left, right, fill=15)
    except ObservationUnavailable as exc:
        assert "cadence" in str(exc)
    else:
        raise AssertionError("persistent cadence failure did not stop the observer")

    runtime._runtime.fail_reason = None
    runtime.reset(episode_start_timestamp_s=0.5)
    for index, timestamp_s in enumerate([1.26, 1.276, 1.293, 1.31]):
        record = _predict(
            runtime, timestamp_s, ft_times, left, right, fill=20 + index
        )
    assert record.warmed_up

    runtime._runtime.fail_reason = "stale_ft_at_rgb_anchor"
    try:
        _predict(runtime, 1.34, ft_times, left, right, fill=24)
    except ObservationUnavailable as exc:
        assert "stale_ft" in str(exc)
    else:
        raise AssertionError("non-cadence timing fault was converted to rewarm")


def test_policy_anchor_path_runs_classifier_once_and_excludes_future_history(
    monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "best_context_dkim.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        runtime_module, "RGBForceContextRuntime", _FakeRGBForceRuntime
    )
    runtime = runtime_module.RGBForceValveContextRuntime(
        checkpoint, device="cpu"
    )
    runtime.reset(episode_start_timestamp_s=0.5)

    rgb_timestamps = np.asarray(
        [1.0, 1.016, 1.033, 1.05, 1.08], dtype=np.float64
    )
    rgb_frames = np.stack(
        [
            np.full((224, 224, 3), fill, dtype=np.uint8)
            for fill in range(len(rgb_timestamps))
        ]
    )
    ft_timestamps = np.arange(0.5, 1.101, 0.01, dtype=np.float64)
    ft_left = np.zeros((len(ft_timestamps), 6), dtype=np.float32)
    ft_right = np.zeros_like(ft_left)

    record = runtime.predict_policy_anchor(
        timestamp_s=1.05,
        rgb=rgb_frames[3],
        rgb_timestamps=rgb_timestamps,
        rgb_frames=rgb_frames,
        ft_timestamps=ft_timestamps,
        ft_left=ft_left,
        ft_right=ft_right,
    )

    assert record.warmed_up
    assert runtime._runtime.predict_calls == 1
    snapshot = runtime.get_last_classifier_model_inputs()
    assert snapshot["rgb_timestamp_s"][-1] == pytest.approx(1.05)
    assert snapshot["ft_timestamp_s"][-1] <= 1.05
    np.testing.assert_array_equal(snapshot["camera0_rgb"][-1], rgb_frames[3])

    runtime.predict_policy_anchor(
        timestamp_s=1.08,
        rgb=rgb_frames[4],
        rgb_timestamps=rgb_timestamps,
        rgb_frames=rgb_frames,
        ft_timestamps=ft_timestamps,
        ft_left=ft_left,
        ft_right=ft_right,
    )
    assert runtime._runtime.predict_calls == 2


def test_policy_anchor_path_rejects_same_timestamp_with_different_pixels(
    monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "best_context_dkim.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        runtime_module, "RGBForceContextRuntime", _FakeRGBForceRuntime
    )
    runtime = runtime_module.RGBForceValveContextRuntime(
        checkpoint, device="cpu"
    )
    runtime.reset(episode_start_timestamp_s=0.5)
    rgb_timestamps = np.asarray(
        [1.0, 1.016, 1.033, 1.05], dtype=np.float64
    )
    rgb_frames = np.zeros(
        (len(rgb_timestamps), 224, 224, 3), dtype=np.uint8
    )
    ft_timestamps = np.arange(0.5, 1.061, 0.01, dtype=np.float64)
    ft = np.zeros((len(ft_timestamps), 6), dtype=np.float32)

    with pytest.raises(ValueError, match="pixels differ"):
        runtime.predict_policy_anchor(
            timestamp_s=1.05,
            rgb=np.ones((224, 224, 3), dtype=np.uint8),
            rgb_timestamps=rgb_timestamps,
            rgb_frames=rgb_frames,
            ft_timestamps=ft_timestamps,
            ft_left=ft,
            ft_right=ft,
        )
    assert runtime._runtime.predict_calls == 0
