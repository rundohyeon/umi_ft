"""Production entry point for the dual-finger F/T Indy policy.

Checkpoint inspection remains hardware-free. Live execution delegates to the
full UMI Indy loop, including teleop initialization, first-frame overlap,
startup F/T bias calibration, causal histories, force-to-width feedback,
deadline-aware scheduling, and fail-closed motion/F/T guards.
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
_DEFAULT_MATCH_DATASET = _ROOT / "session_260827" / "dataset.zarr.zip"


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
    default=_DEFAULT_MATCH_DATASET,
    type=click.Path(path_type=Path),
    show_default=True,
    help="Training replay used for initial-pose selection and exact first-frame overlap.",
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
    dry_run: bool,
    max_cycles: int | None,
    inspect_checkpoint_flag: bool,
    reference_arg: tuple[str, ...],
):
    """Inspect the checkpoint or start the guarded live deployment loop."""
    contract = inspect_checkpoint_contract(str(checkpoint))
    for key, value in contract.items():
        click.echo(f"{key}: {value}")
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
    ]
    if match_dataset is not None:
        live_args.extend(["--match_dataset", str(match_dataset.expanduser().resolve())])
    if show_policy_overlap:
        live_args.append("--show_policy_image")
    if dry_run:
        live_args.append("--plan_only")
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
