#!/usr/bin/env python3
"""Neuromeka Indy waypoint runner using an OnRobot RG2-FT gripper.

The Indy movej/movel implementation is shared with
``example_neuromeka_movej_movel.py``.  Only the gripper session and its CLI/YAML
configuration are RG2-FT specific.

Example:
    python3 scripts/example_neuromeka_movej_movel_rg2.py run \
        --ip 192.168.1.10 \
        --waypoints scripts/waypoints/rulebase_indy_rg2.yaml
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Do not import through ``from scripts ...``.  ROS 2 installs an unrelated
# top-level package named ``scripts`` under /opt/ros, which can win module
# resolution inside the container because this repository's scripts directory
# is not a Python package.  Loading the sibling file directly also makes it
# explicit that the RG2 runner reuses this repository's Indy motion code.
_BASE_RUNNER_PATH = Path(__file__).resolve().with_name(
    "example_neuromeka_movej_movel.py"
)
_BASE_MODULE_NAME = "_indy_umi_neuromeka_waypoint_base"
_BASE_SPEC = importlib.util.spec_from_file_location(
    _BASE_MODULE_NAME, _BASE_RUNNER_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Could not load Indy waypoint runner: {_BASE_RUNNER_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)

from umi.real_world.rg2ft_protocol import (  # noqa: E402
    GRIPPER_SPECS,
    width_to_meters,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WAYPOINTS = SCRIPT_DIR / "waypoints" / "rulebase_indy_rg2.yaml"


@dataclass
class RG2GripperConfig:
    host: str = "192.168.2.1"
    port: int = 502
    slave_id: int = 65
    gripper_type: str = "rg2ft"
    frequency: float = 100.0
    force_n: float = 20.0
    open_width_m: float = 0.1
    close_width_m: float = 0.0
    move_timeout_s: float = 8.0
    move_tolerance_m: float = 0.005
    finish_mode: str = "goal"
    finish_time_s: float = 1.0
    stopped_stable_s: float = 0.35
    stopped_tolerance_m: float = 0.0005
    launch_timeout_s: float = 5.0


class RG2GripperSession:
    """Synchronous waypoint interface using direct RG2-FT width targets.

    Starting the session is read-only.  A motion command is sent only when a
    gripper waypoint is executed.  Unlike the realtime eval controller, this
    session never interpolates the width into 100 Hz intermediate targets.
    RG2-FT receives one final target repeatedly: closed/intermediate widths are
    held, while a fully-open target is released after it is reached.
    """

    def __init__(self, cfg: RG2GripperConfig, *, recorder_factory=None):
        self.cfg = cfg
        self._recorder_factory = recorder_factory
        self._recorder = None

    @property
    def max_width_m(self) -> float:
        spec = GRIPPER_SPECS[self.cfg.gripper_type]
        return width_to_meters(spec["max_width"])

    def __enter__(self) -> "RG2GripperSession":
        factory = self._recorder_factory
        if factory is None:
            # This is the direct-target implementation already verified during
            # RG2-FT data collection. Import lazily so --help stays lightweight.
            from umi.real_world.rg2ft_recorder import RG2FTRecorder

            factory = RG2FTRecorder
        try:
            self._recorder = factory(
                hostname=self.cfg.host,
                port=self.cfg.port,
                slave_id=self.cfg.slave_id,
                gripper_type=self.cfg.gripper_type,
                frequency=self.cfg.frequency,
                default_force=self.cfg.force_n,
                verbose=True,
            )
            self._recorder.start(wait=True, timeout=self.cfg.launch_timeout_s)
            present = self._wait_for_sample(timeout_s=self.cfg.launch_timeout_s)
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

        print(
            f"RG2-FT connected at {self.cfg.host}:{self.cfg.port} "
            f"slave={self.cfg.slave_id} "
            f"(present={present:.4f}m, "
            f"open={self.cfg.open_width_m:.4f}m, "
            f"close={self.cfg.close_width_m:.4f}m, "
            f"force={self.cfg.force_n:.1f}N, control=direct-target)"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as stop_exc:
                if exc is None:
                    print(f"RG2-FT shutdown warning: {stop_exc}", file=sys.stderr)

    def _wait_for_sample(self, *, timeout_s: float) -> float:
        if self._recorder is None:
            raise RuntimeError("RG2-FT session is not connected")
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            sample_time = float(self._recorder.last_sample_time)
            max_age_s = max(1.0, 5.0 / self.cfg.frequency)
            if sample_time > 0 and time.time() - sample_time <= max_age_s:
                return float(self._recorder.last_width)
            thread = getattr(self._recorder, "_thread", None)
            if thread is not None and not thread.is_alive():
                raise RuntimeError("RG2-FT direct-target thread stopped unexpectedly")
            time.sleep(0.02)
        raise TimeoutError(
            f"RG2-FT status is unavailable or stale for {timeout_s:.1f}s"
        )

    def _position_m(self) -> float:
        return self._wait_for_sample(timeout_s=1.0)

    def resolve_target(self, target: Any) -> float:
        if isinstance(target, str):
            key = target.strip().lower()
            if key == "open":
                value = self.cfg.open_width_m
            elif key == "close":
                value = self.cfg.close_width_m
            else:
                try:
                    value = float(key)
                except ValueError as exc:
                    raise ValueError(
                        f"Unknown RG2 target {target!r}. Use open, close, or width in metres."
                    ) from exc
        else:
            value = float(target)
        if not 0.0 <= value <= self.max_width_m:
            raise ValueError(
                f"RG2 target width must be 0..{self.max_width_m:.4f}m, got {value}"
            )
        return value

    def move_to(
        self,
        target: Any,
        *,
        wait: bool = True,
        optional: bool = False,
        timeout_s: float | None = None,
        tolerance: float | None = None,
        finish_mode: str | None = None,
        finish_time_s: float | None = None,
        current_limit: float | None = None,
        pwm_limit: float | None = None,
    ) -> float:
        if self._recorder is None:
            raise RuntimeError("RG2-FT session is not connected")
        if current_limit is not None or pwm_limit is not None:
            print(
                "  gripper: current_limit/pwm_limit are Dynamixel-only; "
                f"RG2-FT uses force_n={self.cfg.force_n:.1f}N from the gripper config"
            )

        goal = self.resolve_target(target)
        timeout_s = self.cfg.move_timeout_s if timeout_s is None else float(timeout_s)
        tolerance_m = (
            self.cfg.move_tolerance_m if tolerance is None else float(tolerance)
        )
        mode = self.cfg.finish_mode if finish_mode is None else str(finish_mode)
        mode = mode.strip().lower()
        finish_time_s = (
            self.cfg.finish_time_s
            if finish_time_s is None
            else float(finish_time_s)
        )
        if mode not in {"goal", "stopped", "time"}:
            raise ValueError(
                f"Unknown RG2 finish_mode={mode!r}; use goal, stopped, or time"
            )
        if timeout_s <= 0 or tolerance_m < 0 or finish_time_s < 0:
            raise ValueError("RG2 timeout must be positive and tolerances/times non-negative")

        before = self._position_m()
        is_fully_open_goal = goal >= self.max_width_m - tolerance_m
        if (
            mode == "goal"
            and is_fully_open_goal
            and abs(before - goal) <= tolerance_m
        ):
            print(f"  gripper: already at goal={goal:.4f}m (present={before:.4f}m), skip")
            return before

        print(
            f"  gripper: moving {before:.4f}m -> {goal:.4f}m "
            f"(force={self.cfg.force_n:.1f}N, finish_mode={mode}, "
            f"timeout={timeout_s:.1f}s, tolerance={tolerance_m:.4f}m)"
        )
        # Send the final target directly. RG2-FT performs its own internal
        # motion; streaming changing intermediate targets makes it repeatedly
        # restart and produces the observed stop-start movement.
        if is_fully_open_goal:
            self._recorder.open_gripper(force_n=self.cfg.force_n)
            target_mode = "rest-at-open"
        elif goal <= 0.0:
            self._recorder.close_gripper(force_n=self.cfg.force_n)
            target_mode = "hold"
        else:
            self._recorder.set_width(goal, force_n=self.cfg.force_n)
            target_mode = "hold"
        print(
            f"  gripper: direct final target={goal:.4f}m "
            f"(mode={target_mode}; no 100Hz width interpolation)"
        )
        if not wait:
            return before

        start = time.monotonic()
        deadline = start + timeout_s
        stable_since = None
        stable_reference = before
        last_position = before
        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                position = self._position_m()
                last_position = position
                if abs(position - goal) <= tolerance_m:
                    break
                if mode == "time" and now - start >= finish_time_s:
                    break
                if mode == "stopped" and now - start >= finish_time_s:
                    if abs(position - stable_reference) <= self.cfg.stopped_tolerance_m:
                        if stable_since is None:
                            stable_since = now
                        elif now - stable_since >= self.cfg.stopped_stable_s:
                            break
                    else:
                        stable_reference = position
                        stable_since = None
                time.sleep(0.05)
            else:
                raise TimeoutError(
                    f"RG2-FT move timed out: present={last_position:.4f}m, "
                    f"goal={goal:.4f}m"
                )
        except TimeoutError as exc:
            print(f"  gripper: WARNING {exc}")
            if not optional:
                raise
            return last_position

        after = self._position_m()
        print(f"  gripper: done present={after:.4f}m (goal={goal:.4f}m)")
        return after


def _parse_rg2_config(
    data: dict[str, Any],
    *,
    cli_host: str | None,
    cli_port: int | None,
    cli_slave_id: int | None,
) -> RG2GripperConfig | None:
    raw = data.get("gripper")
    if raw is None and cli_host is None and cli_port is None and cli_slave_id is None:
        return None
    raw = dict(raw or {})
    if cli_host is not None:
        raw["host"] = cli_host
    if cli_port is not None:
        raw["port"] = cli_port
    if cli_slave_id is not None:
        raw["slave_id"] = cli_slave_id

    dynamixel_keys = {"baudrate", "open_position", "close_position", "dxl_id"}
    found_dynamixel = sorted(dynamixel_keys.intersection(raw))
    if found_dynamixel:
        raise ValueError(
            "This looks like a Dynamixel waypoint file "
            f"({', '.join(found_dynamixel)}). Use rulebase_indy_rg2.yaml or replace "
            "the gripper section with RG2-FT host/port/width settings."
        )

    gripper_type = str(raw.get("gripper_type", "rg2ft")).lower()
    if gripper_type not in GRIPPER_SPECS:
        raise ValueError(
            f"Unsupported gripper_type={gripper_type!r}; expected one of "
            f"{sorted(GRIPPER_SPECS)}"
        )
    physical_max_m = width_to_meters(GRIPPER_SPECS[gripper_type]["max_width"])
    cfg = RG2GripperConfig(
        host=str(raw.get("host", raw.get("ip", "192.168.2.1"))),
        port=int(raw.get("port", 502)),
        slave_id=int(raw.get("slave_id", 65)),
        gripper_type=gripper_type,
        frequency=float(raw.get("frequency", 100.0)),
        force_n=float(raw.get("force_n", 20.0)),
        open_width_m=float(raw.get("open_width_m", physical_max_m)),
        close_width_m=float(raw.get("close_width_m", 0.0)),
        move_timeout_s=float(raw.get("move_timeout_s", raw.get("timeout_s", 8.0))),
        move_tolerance_m=float(
            raw.get("move_tolerance_m", raw.get("tolerance_m", 0.005))
        ),
        finish_mode=str(raw.get("finish_mode", "goal")),
        finish_time_s=float(raw.get("finish_time_s", 1.0)),
        stopped_stable_s=float(raw.get("stopped_stable_s", 0.35)),
        stopped_tolerance_m=float(raw.get("stopped_tolerance_m", 0.0005)),
        launch_timeout_s=float(raw.get("launch_timeout_s", 5.0)),
    )
    max_force_n = GRIPPER_SPECS[gripper_type]["max_force"] / 10.0
    if not 0.0 <= cfg.close_width_m <= cfg.open_width_m <= physical_max_m:
        raise ValueError(
            "RG2 widths must satisfy "
            f"0 <= close_width_m <= open_width_m <= {physical_max_m:.4f}"
        )
    if not 0.0 <= cfg.force_n <= max_force_n:
        raise ValueError(f"RG2 force_n must be 0..{max_force_n:.1f}N")
    if cfg.frequency <= 0:
        raise ValueError("RG2 frequency must be positive")
    return cfg


def run_waypoints_rg2(
    indy,
    waypoint_file: Path,
    *,
    vel_ratio: int,
    acc_ratio: int,
    dwell_s: float,
    gripper_cfg: RG2GripperConfig | None,
    robot_ip: str | None = None,
) -> None:
    data = base._load_waypoint_file(waypoint_file)
    defaults = data.get("defaults", {}) or {}
    default_vel_ratio = int(defaults.get("vel_ratio", vel_ratio))
    default_acc_ratio = int(defaults.get("acc_ratio", acc_ratio))
    default_dwell_s = float(defaults.get("dwell_s", dwell_s))
    waypoints = data["waypoints"]

    needs_gripper = base._sequence_uses_gripper(waypoints)
    if needs_gripper and gripper_cfg is None:
        print("Gripper waypoints present; running arm only (--no-gripper).")
    print(f"Loaded {len(waypoints)} waypoint(s) from {waypoint_file}")

    gripper = (
        RG2GripperSession(gripper_cfg)
        if gripper_cfg is not None and needs_gripper
        else None
    )
    if gripper is not None:
        gripper.__enter__()
    try:
        for wp in waypoints:
            if not isinstance(wp, dict):
                raise ValueError(f"Each waypoint must be a mapping, got {wp!r}")
            indy = base._execute_waypoint(
                indy,
                wp,
                default_vel_ratio=default_vel_ratio,
                default_acc_ratio=default_acc_ratio,
                default_dwell_s=default_dwell_s,
                gripper=gripper,
                robot_ip=robot_ip,
            )
    finally:
        if gripper is not None:
            gripper.__exit__(*sys.exc_info())


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli() -> None:
    """Neuromeka movej/movel waypoint runner with an RG2-FT gripper."""


@cli.command("run")
@click.option("--ip", default="192.168.1.10", show_default=True, help="Robot controller IP")
@click.option(
    "--sim",
    is_flag=True,
    help="Use ROS2/Gazebo joint_trajectory_controller; gripper waypoints are skipped.",
)
@click.option(
    "--ros-action",
    default="/joint_trajectory_controller/follow_joint_trajectory",
    show_default=True,
)
@click.option("--joint-names", default=None, help="Comma-separated simulation joint names")
@click.option("--sim-duration-s", default=4.0, show_default=True, type=float)
@click.option("--ros-timeout-s", default=10.0, show_default=True, type=float)
@click.option(
    "--waypoints",
    "waypoint_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_WAYPOINTS,
    show_default=True,
)
@click.option("--vel-ratio", default=20, show_default=True, type=int)
@click.option("--acc-ratio", default=50, show_default=True, type=int)
@click.option("--dwell-s", default=0.3, show_default=True, type=float)
@click.option(
    "--gripper-ip",
    "--gripper-host",
    "gripper_host",
    default=None,
    help="RG2-FT Compute Box IP (overrides YAML gripper.host)",
)
@click.option(
    "--gripper-port",
    default=None,
    type=int,
    help="RG2-FT Modbus/TCP port (overrides YAML gripper.port)",
)
@click.option(
    "--gripper-slave-id",
    default=None,
    type=int,
    help="RG2-FT Modbus slave ID (overrides YAML gripper.slave_id)",
)
@click.option("--no-gripper", is_flag=True, help="Skip all gripper waypoints")
@click.option("--skip-ready", is_flag=True, help="Skip waiting for robot OP_IDLE")
def run_cmd(
    ip: str,
    sim: bool,
    ros_action: str,
    joint_names: str | None,
    sim_duration_s: float,
    ros_timeout_s: float,
    waypoint_file: Path,
    vel_ratio: int,
    acc_ratio: int,
    dwell_s: float,
    gripper_host: str | None,
    gripper_port: int | None,
    gripper_slave_id: int | None,
    no_gripper: bool,
    skip_ready: bool,
) -> None:
    """Execute robot and RG2-FT waypoints from a YAML/JSON file."""
    data = base._load_waypoint_file(waypoint_file)

    if sim:
        try:
            base.run_waypoints_ros2(
                waypoint_file,
                action_name=ros_action,
                joint_names_csv=joint_names,
                duration_s=sim_duration_s,
                action_timeout_s=ros_timeout_s,
                dwell_s=dwell_s,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            raise SystemExit(130) from None
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print("\nDone.")
        return

    try:
        gripper_cfg = None if no_gripper else _parse_rg2_config(
            data,
            cli_host=gripper_host,
            cli_port=gripper_port,
            cli_slave_id=gripper_slave_id,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    indy = base._connect_and_prepare(ip, skip_ready=skip_ready)
    try:
        run_waypoints_rg2(
            indy,
            waypoint_file,
            vel_ratio=vel_ratio,
            acc_ratio=acc_ratio,
            dwell_s=dwell_s,
            gripper_cfg=gripper_cfg,
            robot_ip=ip,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        base._handle_motion_errors(indy)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        base._handle_motion_errors(indy)
        raise SystemExit(1) from exc
    print("\nDone.")


# Status/demo are robot-only, so the proven commands from the original runner
# can be shared without any gripper-specific changes.
cli.add_command(base.status_cmd, name="status")
cli.add_command(base.demo_cmd, name="demo")


def main() -> None:
    # Preserve the original runner's backward-compatible default-to-demo mode.
    if len(sys.argv) > 1 and sys.argv[1] not in ("run", "status", "demo", "-h", "--help"):
        sys.argv.insert(1, "demo")
    cli()


if __name__ == "__main__":
    main()
