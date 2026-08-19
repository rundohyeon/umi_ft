"""Safe entry point for the 0814 dual-finger F/T Indy policy.

This file deliberately delegates robot/camera/action execution to
``eval_real_indy_rg2.py``.  That keeps the established Indy TCP-frame and
timestamped-command implementation single-sourced while exposing a compact,
checkpoint-first dual-F/T CLI.  ``--dry-run`` is implemented with the
reference evaluator's ``--plan_only`` path, so it starts every live sensor and
the policy but never calls ``UmiEnv.exec_actions``.
"""

from __future__ import annotations

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


def inspect_checkpoint_contract(path: str) -> dict:
    """Load and validate metadata/state without loading any robot adapter."""
    from eval_real_indy_rg2 import inspect_dual_ft_checkpoint_payload

    checkpoint = _resolve_checkpoint(path)
    payload = torch.load(
        open(checkpoint, "rb"), map_location="cpu", pickle_module=dill
    )
    contract = inspect_dual_ft_checkpoint_payload(payload)
    cfg = contract["cfg"]
    return {
        "checkpoint": str(checkpoint),
        "condition": [1, contract["condition_dim"]],
        "action": [1, contract["action_horizon"], contract["action_dim"]],
        "left_ft": [1, contract["ft_horizon"], contract["ft_dim"]],
        "right_ft": [1, contract["ft_horizon"], contract["ft_dim"]],
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
@click.option("--n-action-steps", default="2", type=click.Choice(["1", "2", "4", "8"]))
@click.option("--device", default="auto", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
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
    n_action_steps: str,
    device: str,
    dry_run: bool,
    max_cycles: int | None,
    inspect_checkpoint_flag: bool,
    reference_arg: tuple[str, ...],
):
    """Inspect or run the FT-conditioned position policy (never auto-actuate)."""
    contract = inspect_checkpoint_contract(str(checkpoint))
    for key, value in contract.items():
        click.echo(f"{key}: {value}")
    if inspect_checkpoint_flag:
        return

    if not robot_config.is_file():
        raise click.ClickException(f"robot config does not exist: {robot_config}")
    # UmiEnv requires the parent output directory to exist.  This creates only
    # the requested evaluation log location, not a hardware side effect.
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "--input", contract["checkpoint"],
        "--output", str(log_dir),
        "--robot_config", str(robot_config),
        "--steps_per_inference", str(n_action_steps),
        "--device", str(device),
    ]
    if dry_run:
        command.extend(["--plan_only", "--auto_start_policy"])
        if max_cycles is not None:
            command.extend(["--max_policy_iters", str(max_cycles)])
        click.echo(
            "DRY RUN: live camera/robot/RG2-FT observations and model inference "
            "will run; robot commands are disabled."
        )
    elif max_cycles is not None:
        command.extend(["--max_policy_iters", str(max_cycles)])
    command.extend(reference_arg)

    # Import only after inspection so --inspect-checkpoint has no robot/camera
    # dependency.  The reference command owns its tried-and-tested motion path.
    from eval_real_indy_rg2 import main as reference_main

    try:
        reference_main.main(args=command, standalone_mode=False)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
