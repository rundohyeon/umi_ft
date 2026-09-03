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
CHECKPOINT="${RG2_CHECKPOINT:-$ROOT/umi_18.45.44_latest.ckpt}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/data/eval_indy_rg2}"
ROBOT_CONFIG="${RG2_ROBOT_CONFIG:-$ROOT/example/eval_robots_config_indy_rg2.yaml}"
MATCH_DATASET="${MATCH_DATASET:-$ROOT/data/dataset.zarr.zip}"
MATCH_EPISODE="${MATCH_EPISODE:-0}"
ACTION_SCALE="${ACTION_SCALE:-0.2}"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the UMI environment containing requirements_rg2ft.txt." >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the UMI environment containing requirements_rg2ft.txt." >&2
  exit 1
fi
for required in "$CHECKPOINT" "$ROBOT_CONFIG" "$MATCH_DATASET"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$(dirname -- "$OUTPUT_DIR")"

RG2_ENABLE_MOTION="${RG2_ENABLE_MOTION:-1}"
SAFETY_ARGS=(--plan_only --print_motion_debug --max_policy_iters 1)
if [[ "$RG2_ENABLE_MOTION" == "1" ]]; then
  SAFETY_ARGS=()
  echo "Motion default: robot and RG2-FT motion are enabled."
else
  echo "RG2_ENABLE_MOTION=$RG2_ENABLE_MOTION: plan-only; RG2-FT connects read-only and receives no width command."
fi

exec "$PYTHON_BIN" "$ROOT/eval_real_indy_rg2.py" \
  --input "$CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --robot_config "$ROBOT_CONFIG" \
  --match_dataset "$MATCH_DATASET" \
  --match_episode "$MATCH_EPISODE" \
  --allow_rotation \
  --action_scale "$ACTION_SCALE" \
  --vis_pose \
  "${SAFETY_ARGS[@]}" \
  "$@"
