from pathlib import Path

import numpy as np
import torch

import umi.real_world.valve_state_context as runtime_module
from diffusion_policy.model.vision.valve_context_observer_v2 import (
    OBSERVER_PHASE_NAMES,
    OBSERVER_SCHEMA,
)


class _FakeObserver:
    def __init__(self):
        self.last_obs = None

    def predict_context(self, obs):
        self.last_obs = obs
        batch = obs["camera0_rgb"].shape[0]
        probability = torch.tensor(
            [[0.1, 0.2, 0.6, 0.1]], dtype=torch.float32
        ).repeat(batch, 1)
        return {
            "context_prob": probability,
            "context_logits": torch.log(probability),
            "context_pred": probability.argmax(dim=-1),
        }


def test_v2_runtime_builds_relative_pose_and_causal_mask(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "best_context.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    fake = _FakeObserver()
    monkeypatch.setattr(
        runtime_module,
        "load_frozen_context_observer",
        lambda path, device: (
            fake,
            OBSERVER_PHASE_NAMES,
            {"schema": OBSERVER_SCHEMA, "epoch": 2, "validation": {}},
        ),
    )
    runtime = runtime_module.ValveStateContextRuntimeV2(checkpoint, device="cpu")
    force_timestamps = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    left = np.zeros((101, 6), dtype=np.float32)
    right = np.zeros((101, 6), dtype=np.float32)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)

    record = None
    for index in range(4):
        timestamp = 1.0 + 0.01 * index
        usable = force_timestamps <= timestamp
        record = runtime.predict(
            timestamp_s=timestamp,
            rgb=rgb,
            position_m=np.asarray([0.01 * index, 0.0, 0.0]),
            rotation_axis_angle_rad=np.zeros(3),
            gripper_width_m=0.05,
            ft_timestamps=force_timestamps[usable],
            ft_left=left[usable],
            ft_right=right[usable],
        )

    assert record is not None and record.warmed_up
    assert record.phase_name == "recovery"
    np.testing.assert_allclose(record.values, [0.1, 0.2, 0.6, 0.1, 1.0])
    snapshot = runtime.get_last_classifier_model_inputs()
    np.testing.assert_allclose(snapshot["lowdim"][:, :3], [[-0.03, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(snapshot["wrench_mask"], 1.0)
    np.testing.assert_allclose(snapshot["rgb_timestamp_s"], [1.0, 1.03])
    assert fake.last_obs["robot0_ft_history"].shape == (1, 2, 50, 12)


def test_v2_runtime_rejects_future_force(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "best_context.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    fake = _FakeObserver()
    monkeypatch.setattr(
        runtime_module,
        "load_frozen_context_observer",
        lambda path, device: (
            fake,
            OBSERVER_PHASE_NAMES,
            {"schema": OBSERVER_SCHEMA, "epoch": 2, "validation": {}},
        ),
    )
    runtime = runtime_module.ValveStateContextRuntimeV2(checkpoint, device="cpu")
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
        assert "newer" in str(exc)
    else:
        raise AssertionError("future F/T was accepted")
