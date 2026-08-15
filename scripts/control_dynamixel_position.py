#!/usr/bin/env python3
"""CLI for Dynamixel position control.

Examples:
    python control_dynamixel_position.py --port /dev/ttyUSB0 --id 1 --read
    python control_dynamixel_position.py --port /dev/ttyUSB0 --id 1 --goal 1700 --keep-torque
    python control_dynamixel_position.py run --port /dev/ttyUSB0 --id 1 --sweep 500 1700
"""

from __future__ import annotations

import os
import sys
import time

import click

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from umi.real_world.dynamixel_controller import (  # noqa: E402
    DynamixelConfig,
    DynamixelPositionController,
    PROTOCOL_1_0,
    PROTOCOL_2_0,
)

SCRIPT_VERSION = "2026-06-17"


def _parse_ids(ids: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in ids.split(",") if x.strip())


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--port", default="/dev/ttyUSB0", show_default=True, help="Serial port")
@click.option("--baudrate", default=57600, show_default=True, type=int)
@click.option(
    "--protocol",
    type=click.Choice(["1.0", "2.0"], case_sensitive=False),
    default="2.0",
    show_default=True,
)
@click.option("--id", "dxl_ids", default="1", show_default=True, help="Comma-separated motor IDs")
@click.option("--profile-velocity", default=30, show_default=True, type=int,
              help="Motion speed (lower = slower; typical range 10-200)")
@click.option("--profile-acceleration", default=15, show_default=True, type=int,
              help="Motion acceleration (lower = gentler ramp)")
@click.option("--read", "read_only", is_flag=True, help="Read present position and exit")
@click.option("--goal", type=int, default=None, help="Goal position in encoder ticks")
@click.option(
    "--sweep",
    nargs=2,
    type=int,
    default=None,
    metavar="MIN MAX",
    help="Move back and forth between two positions",
)
@click.option("--cycles", default=3, show_default=True, type=int, help="Sweep repeat count")
@click.option("--hold", default=0.5, show_default=True, type=float, help="Hold time at each sweep point")
@click.option("--timeout", default=30.0, show_default=True, type=float, help="Move timeout in seconds")
@click.option("--tolerance", default=15, show_default=True, type=int, help="Goal reach tolerance in ticks")
@click.option(
    "--current-limit",
    default=None,
    type=float,
    help="Max current for XM/XC motors. Ignored on XL430 (model 1060).",
)
@click.option(
    "--pwm-limit",
    default=None,
    type=float,
    help="Max PWM for XL430 grip force: raw 0-885 or fraction <=1 (e.g. 0.45).",
)
@click.option(
    "--finish-mode",
    type=click.Choice(["goal", "stopped", "time"], case_sensitive=False),
    default="goal",
    show_default=True,
    help="How to decide the move is done",
)
@click.option(
    "--finish-time",
    default=1.0,
    show_default=True,
    type=float,
    help="Hold time for finish_mode=time, or stopped timeout",
)
@click.option("--no-wait", is_flag=True, help="Do not wait until motion completes")
@click.option(
    "--keep-torque",
    is_flag=True,
    help="Leave torque enabled when the script exits (avoids partial moves on rerun)",
)
def main(
    port: str,
    baudrate: int,
    protocol: str,
    dxl_ids: str,
    profile_velocity: int,
    profile_acceleration: int,
    read_only: bool,
    goal: int | None,
    sweep: tuple[int, int] | None,
    cycles: int,
    hold: float,
    timeout: float,
    tolerance: int,
    current_limit: float | None,
    pwm_limit: float | None,
    finish_mode: str,
    finish_time: float,
    no_wait: bool,
    keep_torque: bool,
) -> None:
    ids = _parse_ids(dxl_ids)
    protocol_version = PROTOCOL_2_0 if protocol == "2.0" else PROTOCOL_1_0

    config = DynamixelConfig(
        port=port,
        baudrate=baudrate,
        protocol_version=protocol_version,
        dxl_ids=ids,
        profile_velocity=profile_velocity,
        profile_acceleration=profile_acceleration,
        current_limit=current_limit,
        pwm_limit=pwm_limit,
    )

    with DynamixelPositionController(config) as controller:
        controller._disable_torque_on_exit = not keep_torque
        controller.configure_position_mode()
        controller.enable_torque()

        positions = controller.get_present_positions()
        print("Present positions:")
        for dxl_id, pos in positions.items():
            pwm = controller.get_pwm_limit(dxl_id)
            limit = controller.get_current_limit(dxl_id)
            extras = []
            if pwm is not None:
                extras.append(f"pwm_limit={pwm}")
            if limit is not None:
                extras.append(f"current_limit={limit}")
            model = controller._try_get_model_number(dxl_id)
            suffix = f", {', '.join(extras)}" if extras else ""
            model_s = f", model={model}" if model is not None else ""
            print(f"  id={dxl_id}: pos={pos}{model_s}{suffix}")

        if read_only:
            return

        if goal is None and sweep is None:
            raise click.UsageError("Specify --read, --goal, or --sweep")

        move_kwargs = {
            "wait": not no_wait,
            "timeout_s": timeout,
            "tolerance": tolerance,
            "finish_mode": finish_mode,
            "finish_time_s": finish_time,
        }

        if goal is not None:
            print(f"Moving to goal={goal}")
            controller.move_to(goal, **move_kwargs)
            final = controller.get_present_positions()
            for dxl_id, pos in final.items():
                print(f"  id={dxl_id}: {pos}")
            return

        assert sweep is not None
        low, high = sweep
        print(f"Sweeping between {low} and {high} for {cycles} cycles")
        for i in range(cycles):
            for target in (low, high):
                print(f"cycle {i + 1}: goal={target}")
                controller.move_to(target, **move_kwargs)
                time.sleep(hold)


if __name__ == "__main__":
    # Backward compat: `python ... run --goal 1700`
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv.pop(1)
    main()
