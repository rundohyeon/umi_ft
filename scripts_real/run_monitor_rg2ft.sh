#!/usr/bin/env bash
# Launch the RG2-FT live force/torque monitor (no recording, no capture card).
#
# Usage:
#   scripts_real/run_monitor_rg2ft.sh [extra args for monitor_rg2ft.py]
# Example:
#   scripts_real/run_monitor_rg2ft.sh
#   scripts_real/run_monitor_rg2ft.sh --fmax 40 --smooth 1.0
#
# Keys: o open | c close | z zero-FT | x clear-tare | p reset-peak | q quit
#
# NOTE: this opens its own Modbus master on the gripper, same as the recorder,
# so stop the OnRobot ROS driver (and the recorder) first.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# activate the recording env (same env as the recorder)
source /home/idim/anaconda3/etc/profile.d/conda.sh
conda activate umi-rec

exec python "$ROOT/scripts_real/monitor_rg2ft.py" "$@"
