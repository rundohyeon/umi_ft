"""Production entry point for the dual-finger F/T Indy policy.

Checkpoint inspection remains hardware-free. Live execution delegates to the
full UMI Indy loop, including teleop initialization, first-frame overlap,
startup F/T bias calibration, causal histories, force-to-width feedback,
deadline-aware scheduling, and fail-closed motion/F/T guards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import click
import dill
import torch
from omegaconf import OmegaConf


_ROOT = Path(__file__).resolve().parent
_DEFAULT_ROBOT_CONFIG = _ROOT / "example" / "eval_robots_config_indy_rg2.yaml"
_DEFAULT_LOG_DIR = _ROOT / "data" / "eval_dual_ft"


def _resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.suffix != ".ckpt":
        checkpoint = checkpoint / "checkpoints" / "latest.ckpt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise click.ClickException(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve checkpoint-serialized project-relative assets portably."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_ROOT / candidate).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint_contract(path: str) -> dict:
    """Load and validate metadata/state without loading any robot adapter."""
    from diffusion_policy.common.dual_ft_contract import (
        inspect_dual_ft_checkpoint_payload,
    )

    checkpoint = _resolve_checkpoint(path)
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    contract = inspect_dual_ft_checkpoint_payload(payload)
    cfg = contract["cfg"]
    feedback = OmegaConf.select(cfg, "task.grasp_force_feedback", default=None)
    if feedback is None:
        raise click.ClickException(
            "checkpoint has no grasp-force width-feedback configuration"
        )
    ft_feature_mode = str(contract["ft_feature_mode"])
    ft_delta_alpha = OmegaConf.select(
        cfg, "policy.obs_encoder.ft_delta_ema_alpha", default=None
    )
    valve_context_enabled = bool(contract.get("valve_context_enabled", False))
    valve_classifier_path = None
    valve_classifier_sha256 = None
    if valve_context_enabled:
        valve_classifier_path = OmegaConf.select(
            cfg, "task.valve_context.checkpoint_path", default=None
        )
        valve_classifier_sha256 = OmegaConf.select(
            cfg, "task.valve_context.checkpoint_sha256", default=None
        )
    return {
        "checkpoint": str(checkpoint),
        "architecture_contract_version": int(
            contract["architecture_contract_version"]
        ),
        "condition": [1, contract["condition_dim"]],
        "action": [1, contract["action_horizon"], contract["action_dim"]],
        "left_ft": [1, contract["ft_horizon"], contract["ft_dim"]],
        "right_ft": [1, contract["ft_horizon"], contract["ft_dim"]],
        "ft_feature_mode": ft_feature_mode,
        "ft_delta_ema_alpha": (
            float(ft_delta_alpha) if ft_delta_alpha is not None else None
        ),
        "fusion_tokens": list(
            OmegaConf.select(
                cfg, "task.model_contract.ft_fusion_token_order", default=[]
            )
        ),
        "num_fusion_tokens": int(contract["num_fusion_tokens"]),
        "training_dataset": str(cfg.task.dataset.dataset_path),
        "valve_context_enabled": valve_context_enabled,
        "valve_context_schema": contract.get("valve_context_schema"),
        "valve_context_dim": contract.get("valve_context_dim"),
        "valve_classifier_checkpoint": (
            str(valve_classifier_path) if valve_classifier_path is not None else None
        ),
        "valve_classifier_sha256": (
            str(valve_classifier_sha256).lower()
            if valve_classifier_sha256 is not None
            else None
        ),
        "n_action_steps": int(
            OmegaConf.select(cfg, "execution.n_action_steps", default=2)
        ),
        "action_frequency_hz": float(
            OmegaConf.select(cfg, "execution.action_frequency", default=0.0)
        ),
        "replanning_interval_ms": float(
            OmegaConf.select(cfg, "execution.replanning_interval_ms", default=0.0)
        ),
        "normalizer_owner": contract["normalizer_owner"],
        "grasp_force_feedback": OmegaConf.to_container(
            feedback, resolve=True
        ),
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--checkpoint", required=True, type=click.Path(path_type=Path))
@click.option(
    "--robot-config",
    default=_DEFAULT_ROBOT_CONFIG,
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option(
    "--log-dir",
    default=_DEFAULT_LOG_DIR,
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option(
    "--match-dataset",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Training replay used for initial-pose selection and exact first-frame "
        "overlap. Defaults to task.dataset.dataset_path serialized in the checkpoint."
    ),
)
@click.option(
    "--show-policy-overlap/--no-show-policy-overlap",
    default=True,
    show_default=True,
    help="Show the exact 224x224 live policy input beside the selected training first frame.",
)
@click.option("--n-action-steps", default="2", type=click.Choice(["1", "2", "4", "8"]))
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--valve-classifier-checkpoint",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Frozen context observer override. By default, use the checkpoint-serialized "
        "path and verify its SHA-256 before connecting to hardware."
    ),
)
@click.option(
    "--ft-max-age-sec",
    default=0.020,
    type=click.FloatRange(min=0.0, min_open=True),
    show_default=True,
    help="Maximum permitted age of the latest causal F/T sample at the RGB anchor.",
)
@click.option(
    "--commissioning-audit/--no-commissioning-audit",
    default=True,
    show_default=True,
    help=(
        "During dry-run, print motion, policy-input, and coordinate-transform audits."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run inference/safety checks but suppress waypoint submission (controllers still connect).",
)
@click.option("--max-cycles", default=None, type=click.IntRange(min=1))
@click.option("--inspect-checkpoint", "inspect_checkpoint_flag", is_flag=True, default=False)
@click.option(
    "--reference-arg",
    multiple=True,
    help="Additional one-token argument forwarded to eval_real_indy_rg2.py.",
)
def main(
    checkpoint: Path,
    robot_config: Path,
    log_dir: Path,
    match_dataset: Path,
    show_policy_overlap: bool,
    n_action_steps: str,
    device: str,
    valve_classifier_checkpoint: Path | None,
    ft_max_age_sec: float,
    commissioning_audit: bool,
    dry_run: bool,
    max_cycles: int | None,
    inspect_checkpoint_flag: bool,
    reference_arg: tuple[str, ...],
):
    """Inspect the checkpoint or start the guarded live deployment loop."""
    contract = inspect_checkpoint_contract(str(checkpoint))
    for key, value in contract.items():
        click.echo(f"{key}: {value}")

    resolved_classifier = None
    if contract["valve_context_enabled"]:
        classifier_arg = (
            valve_classifier_checkpoint
            if valve_classifier_checkpoint is not None
            else contract["valve_classifier_checkpoint"]
        )
        if classifier_arg is None:
            raise click.ClickException(
                "context policy has no frozen classifier path; pass "
                "--valve-classifier-checkpoint"
            )
        resolved_classifier = _resolve_project_path(classifier_arg)
        if not resolved_classifier.is_file():
            raise click.ClickException(
                f"frozen valve classifier does not exist: {resolved_classifier}"
            )
        actual_sha256 = _sha256_file(resolved_classifier)
        expected_sha256 = contract["valve_classifier_sha256"]
        if actual_sha256.lower() != expected_sha256:
            raise click.ClickException(
                "frozen valve classifier SHA-256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        click.echo(f"resolved_valve_classifier: {resolved_classifier}")
        click.echo(f"valve_classifier_sha256_verified: {actual_sha256}")

    if match_dataset is None:
        match_dataset = _resolve_project_path(contract["training_dataset"])
        click.echo(f"auto_match_dataset: {match_dataset}")
    else:
        match_dataset = match_dataset.expanduser().resolve()
    click.echo(f"match_dataset_exists: {match_dataset.exists()}")
    if not inspect_checkpoint_flag and not match_dataset.exists():
        raise click.ClickException(
            f"first-frame match dataset does not exist: {match_dataset}"
        )
    if inspect_checkpoint_flag:
        return

    from eval_real_indy_rg2 import main as live_main

    checkpoint_path = _resolve_checkpoint(str(checkpoint))
    log_dir = log_dir.expanduser().resolve()
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    live_args = [
        "--input", str(checkpoint_path),
        "--output", str(log_dir),
        "--robot_config", str(robot_config.expanduser().resolve()),
        "--steps_per_inference", str(n_action_steps),
        "--device", str(device),
        "--ft_max_age_sec", str(ft_max_age_sec),
    ]
    live_args.extend(["--match_dataset", str(match_dataset)])
    if resolved_classifier is not None:
        live_args.extend(
            ["--valve_classifier_checkpoint", str(resolved_classifier)]
        )
    if show_policy_overlap:
        live_args.append("--show_policy_image")
    if dry_run:
        live_args.append("--plan_only")
        if commissioning_audit:
            live_args.extend(
                [
                    "--print_motion_debug",
                    "--policy_input_audit",
                    "--coord_transform_audit",
                ]
            )
    if max_cycles is not None:
        live_args.extend(["--max_policy_iters", str(max_cycles)])
    live_args.extend(reference_arg)
    live_main.main(
        args=live_args,
        prog_name="eval_real_indy_rg2_dual_ft.py",
        standalone_mode=True,
    )


if __name__ == "__main__":
    main()
