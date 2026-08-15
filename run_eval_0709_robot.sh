#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_BIN="/home/idim/miniforge3/envs/umi/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi
CHECKPOINT="$ROOT/artifacts/0709_robot_tcp/checkpoints/latest.ckpt"
DATASET="$ROOT/artifacts/0709_robot_tcp/dataset_robot_tcp.zarr.zip"
ROBOT_CONFIG="$ROOT/example/eval_robots_config_indy.yaml"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/data/eval_0709_robot_tcp}"

for required in "$CHECKPOINT" "$DATASET" "$ROBOT_CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required local file: $required" >&2
    exit 1
  fi
done
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

# No -m/--match_dataset and no --sim_fov: the 0709 policy uses relative TCP
# actions and was trained on images without FOV rectification. Initial coordinate
# validation intentionally leaves action-latency compensation disabled.
exec "$PYTHON_BIN" "$ROOT/eval_real_indy.py" \
  --input "$CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --robot_config "$ROBOT_CONFIG" \
  --gripper_calib_zarr "$DATASET" \
  --allow_rotation \
  --no_direct_dynamixel_gripper \
  --print_motion_debug \
  --vis_pose \
  "$@"
