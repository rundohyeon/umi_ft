import pytest
import numpy as np

from eval_real_indy_rg2 import (
    _exec_actions_with_fresh_ft_guard,
    _register_context_recovery_skip,
    _runtime_cycle_counts,
)
from umi.real_world.dual_ft_policy_safety import FTSafetyConfig, PolicySafetyError


class _GuardedActionEnv:
    def __init__(self, timestamp):
        self.events = []
        self.timestamp = timestamp

    def get_latest_ft_state(self):
        self.events.append("read_ft")
        zeros = np.zeros(6)
        return {
            "left_raw": zeros,
            "right_raw": zeros,
            "left": zeros,
            "right": zeros,
            "timestamp": self.timestamp,
        }

    def exec_actions(self, *, actions, timestamps, compensate_latency):
        self.events.append("exec_actions")
        self.submitted = (actions, timestamps, compensate_latency)


def test_exec_actions_reads_and_validates_ft_at_command_boundary(monkeypatch):
    env = _GuardedActionEnv(timestamp=100.0)
    monkeypatch.setattr(
        "umi.real_world.dual_ft_policy_safety.time.time",
        lambda: 100.01,
    )
    actions = np.zeros((1, 7))
    timestamps = np.asarray([101.0])

    measured_force_n, sample_age_s = _exec_actions_with_fresh_ft_guard(
        env,
        actions,
        timestamps,
        np.zeros(12),
        FTSafetyConfig(max_latest_sample_age_s=0.05),
    )

    assert env.events == ["read_ft", "exec_actions"]
    assert env.submitted[0] is actions
    assert env.submitted[1] is timestamps
    assert env.submitted[2] is False
    assert measured_force_n == pytest.approx(0.0)
    assert sample_age_s == pytest.approx(0.01)


def test_exec_actions_does_not_submit_when_fresh_ft_guard_rejects(monkeypatch):
    env = _GuardedActionEnv(timestamp=100.0)
    monkeypatch.setattr(
        "umi.real_world.dual_ft_policy_safety.time.time",
        lambda: 100.06,
    )

    with pytest.raises(PolicySafetyError, match="stale"):
        _exec_actions_with_fresh_ft_guard(
            env,
            np.zeros((1, 7)),
            np.asarray([101.0]),
            np.zeros(12),
            FTSafetyConfig(max_latest_sample_age_s=0.05),
        )

    assert env.events == ["read_ft"]


def test_runtime_cycle_counts_separate_safety_rejection_from_dropped_observation():
    counts = _runtime_cycle_counts(
        {
            "attempted_cycles": 10,
            "completed_cycles": 9,
            "valid_observations": 10,
            "safety_rejections": 1,
            "context_recovery_skips": 0,
        }
    )

    assert counts == {
        "cycles": 10,
        "completed_cycles": 9,
        "valid_observation_cycles": 10,
        "dropped_cycles": 0,
        "safety_rejected_cycles": 1,
        "context_recovery_skipped_cycles": 0,
    }


def test_runtime_cycle_counts_report_observation_failure_without_negative_drop():
    counts = _runtime_cycle_counts(
        {
            "attempted_cycles": 10,
            "completed_cycles": 9,
            "valid_observations": 9,
            "safety_rejections": 0,
        }
    )

    assert counts["dropped_cycles"] == 1


def test_runtime_cycle_counts_include_context_recovery_skip():
    counts = _runtime_cycle_counts(
        {
            "attempted_cycles": 10,
            "completed_cycles": 8,
            "valid_observations": 10,
            "safety_rejections": 1,
            "context_recovery_skips": 1,
        }
    )

    assert counts["context_recovery_skipped_cycles"] == 1


def test_context_recovery_skip_holds_motion_and_is_bounded():
    class Env:
        def __init__(self):
            self.hold_calls = 0

        def hold_robot(self):
            self.hold_calls += 1

    env = Env()
    metrics = {"context_recovery_skips": 0}
    consecutive = 0
    for _ in range(3):
        consecutive = _register_context_recovery_skip(
            env,
            metrics,
            plan_only=False,
            consecutive_skips=consecutive,
            reason="test anchor timeout",
        )

    assert consecutive == 3
    assert metrics["context_recovery_skips"] == 3
    assert env.hold_calls == 3
    with pytest.raises(PolicySafetyError, match="exceeded 3"):
        _register_context_recovery_skip(
            env,
            metrics,
            plan_only=False,
            consecutive_skips=consecutive,
            reason="persistent timeout",
        )
    assert metrics["context_recovery_skips"] == 3
    assert env.hold_calls == 4


def test_context_recovery_skip_plan_only_does_not_hold_robot():
    class Env:
        def hold_robot(self):
            raise AssertionError("plan-only recovery must not command the robot")

    metrics = {"context_recovery_skips": 0}
    assert _register_context_recovery_skip(
        Env(),
        metrics,
        plan_only=True,
        consecutive_skips=0,
        reason="test warmup",
    ) == 1
    assert metrics["context_recovery_skips"] == 1


def test_runtime_cycle_counts_reject_invalid_counter_ordering():
    with pytest.raises(ValueError, match="completed <= valid_observations <= attempted"):
        _runtime_cycle_counts(
            {
                "attempted_cycles": 9,
                "completed_cycles": 9,
                "valid_observations": 10,
                "safety_rejections": 0,
            }
        )
