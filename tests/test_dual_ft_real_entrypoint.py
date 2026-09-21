import hashlib
import sys
import types

from click.testing import CliRunner

import eval_real_indy_rg2_dual_ft as entrypoint


def _fake_contract(classifier_sha256):
    return {
        "checkpoint": "/unused/latest.ckpt",
        "architecture_contract_version": 5,
        "condition": [1, 786],
        "action": [1, 16, 11],
        "left_ft": [1, 32, 6],
        "right_ft": [1, 32, 6],
        "ft_feature_mode": "raw_history_plus_delta_history",
        "ft_delta_ema_alpha": 0.25,
        "fusion_tokens": [
            "rgb_old",
            "rgb_current",
            "left_raw_history",
            "right_raw_history",
            "left_delta_history",
            "right_delta_history",
        ],
        "num_fusion_tokens": 6,
        "training_dataset": "three_dataset/dataset.zarr.zip",
        "valve_context_enabled": True,
        "valve_context_schema": "umi_valve_context_sidecar_v2_4state",
        "valve_context_dim": 5,
        "valve_classifier_checkpoint": "answer/best_context.pt",
        "valve_classifier_sha256": classifier_sha256,
        "n_action_steps": 2,
        "action_frequency_hz": 19.98,
        "replanning_interval_ms": 100.1,
        "normalizer_owner": "policy.predict_action",
        "grasp_force_feedback": {"direct_force_command": False},
    }


def test_dry_run_auto_resolves_assets_and_enables_commissioning_audits(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    classifier = tmp_path / "answer" / "best_context.pt"
    classifier.parent.mkdir()
    classifier.write_bytes(b"frozen observer")
    classifier_sha256 = hashlib.sha256(classifier.read_bytes()).hexdigest()
    dataset = tmp_path / "three_dataset" / "dataset.zarr.zip"
    dataset.parent.mkdir()
    dataset.write_bytes(b"dataset")
    robot_config = tmp_path / "robot.yaml"
    robot_config.write_text("robots: []\n")

    monkeypatch.setattr(entrypoint, "_ROOT", tmp_path)
    monkeypatch.setattr(
        entrypoint,
        "inspect_checkpoint_contract",
        lambda _path: _fake_contract(classifier_sha256),
    )

    captured = {}

    class FakeLiveCommand:
        @staticmethod
        def main(*, args, prog_name, standalone_mode):
            captured["args"] = args
            captured["prog_name"] = prog_name
            captured["standalone_mode"] = standalone_mode

    fake_live_module = types.ModuleType("eval_real_indy_rg2")
    fake_live_module.main = FakeLiveCommand
    monkeypatch.setitem(sys.modules, "eval_real_indy_rg2", fake_live_module)

    result = CliRunner().invoke(
        entrypoint.main,
        [
            "--checkpoint",
            str(checkpoint),
            "--robot-config",
            str(robot_config),
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
            "--max-cycles",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args[args.index("--match_dataset") + 1] == str(dataset)
    assert args[args.index("--valve_classifier_checkpoint") + 1] == str(classifier)
    assert args[args.index("--ft_max_age_sec") + 1] == "0.02"
    assert "--plan_only" in args
    assert "--print_motion_debug" in args
    assert "--policy_input_audit" in args
    assert "--coord_transform_audit" in args
    assert captured["standalone_mode"] is True


def test_inspection_rejects_wrong_context_observer(tmp_path, monkeypatch):
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    classifier = tmp_path / "answer" / "best_context.pt"
    classifier.parent.mkdir()
    classifier.write_bytes(b"wrong observer")
    dataset = tmp_path / "three_dataset" / "dataset.zarr.zip"
    dataset.parent.mkdir()
    dataset.write_bytes(b"dataset")

    monkeypatch.setattr(entrypoint, "_ROOT", tmp_path)
    monkeypatch.setattr(
        entrypoint,
        "inspect_checkpoint_contract",
        lambda _path: _fake_contract("0" * 64),
    )

    result = CliRunner().invoke(
        entrypoint.main,
        ["--checkpoint", str(checkpoint), "--inspect-checkpoint"],
    )

    assert result.exit_code != 0
    assert "SHA-256 mismatch" in result.output
