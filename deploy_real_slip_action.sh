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
else
  echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

ACTION_CHECKPOINT="${RG2_CHECKPOINT:-$ROOT/data/contextFT-4/epoch=0008-val_loss=0.02815.ckpt}"
CONTEXT_CHECKPOINT="${VALVE_CLASSIFIER_CHECKPOINT:-$ROOT/data/contextFT-4/best.pt}"
ROBOT_CONFIG="${RG2_ROBOT_CONFIG:-$ROOT/example/eval_robots_config_indy_rg2.yaml}"
ARUCO_CONFIG="${RG2_ARUCO_CONFIG:-$ROOT/slam_pipeline_latest/calibration/aruco_config.yaml}"
MATCH_DATASET="${MATCH_DATASET:-$ROOT/data/context-3/dataset.zarr.zip}"
MATCH_EPISODE="${MATCH_EPISODE:-0}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/data/eval_contextFT4}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
MOTION_MOMENTUM_PREVIOUS_WEIGHT="${MOTION_MOMENTUM_PREVIOUS_WEIGHT:-0.0}"
CONTEXT_MOMENTUM_PREVIOUS_WEIGHT="${CONTEXT_MOMENTUM_PREVIOUS_WEIGHT:-0.5}"
RG2_FT_MAX_AGE_SEC="${RG2_FT_MAX_AGE_SEC:-0.020}"
RG2_LATEST_FT_MAX_AGE_SEC="${RG2_LATEST_FT_MAX_AGE_SEC:-0.050}"
RG2_MAX_RESULT_AGE_SEC="${RG2_MAX_RESULT_AGE_SEC:-0.400}"
RG2_MAX_PROCESSING_AGE_SEC="${RG2_MAX_PROCESSING_AGE_SEC:-0.250}"
RG2_ENABLE_MOTION="${RG2_ENABLE_MOTION:-0}"
RG2_ZERO_FT_ON_START="${RG2_ZERO_FT_ON_START:-1}"
FT_ZERO_SAMPLES="${FT_ZERO_SAMPLES:-25}"
AUTO_START_POLICY="${AUTO_START_POLICY:-0}"
MAX_POLICY_ITERS="${MAX_POLICY_ITERS:-1000000}"
MAX_DURATION="${MAX_DURATION:-2000}"
RG2_SAVE_POLICY_INPUTS="${RG2_SAVE_POLICY_INPUTS:-1}"

for required in "$ACTION_CHECKPOINT" "$CONTEXT_CHECKPOINT" "$ROBOT_CONFIG" "$ARUCO_CONFIG" "$MATCH_DATASET"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

if ! [[ "$RG2_FT_MAX_AGE_SEC" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($RG2_FT_MAX_AGE_SEC > 0) }"; then
  echo "RG2_FT_MAX_AGE_SEC must be a positive number, got: $RG2_FT_MAX_AGE_SEC" >&2
  exit 2
fi
if ! [[ "$RG2_LATEST_FT_MAX_AGE_SEC" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($RG2_LATEST_FT_MAX_AGE_SEC > 0) }"; then
  echo "RG2_LATEST_FT_MAX_AGE_SEC must be a positive number, got: $RG2_LATEST_FT_MAX_AGE_SEC" >&2
  exit 2
fi
if ! [[ "$RG2_MAX_RESULT_AGE_SEC" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($RG2_MAX_RESULT_AGE_SEC > 0) }"; then
  echo "RG2_MAX_RESULT_AGE_SEC must be a positive number, got: $RG2_MAX_RESULT_AGE_SEC" >&2
  exit 2
fi
if ! [[ "$RG2_MAX_PROCESSING_AGE_SEC" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($RG2_MAX_PROCESSING_AGE_SEC > 0) }"; then
  echo "RG2_MAX_PROCESSING_AGE_SEC must be a positive number, got: $RG2_MAX_PROCESSING_AGE_SEC" >&2
  exit 2
fi
if ! [[ "$MOTION_MOMENTUM_PREVIOUS_WEIGHT" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($MOTION_MOMENTUM_PREVIOUS_WEIGHT >= 0 && $MOTION_MOMENTUM_PREVIOUS_WEIGHT < 1) }"; then
  echo "MOTION_MOMENTUM_PREVIOUS_WEIGHT must be in [0,1), got: $MOTION_MOMENTUM_PREVIOUS_WEIGHT" >&2
  exit 2
fi
if ! [[ "$CONTEXT_MOMENTUM_PREVIOUS_WEIGHT" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($CONTEXT_MOMENTUM_PREVIOUS_WEIGHT >= 0 && $CONTEXT_MOMENTUM_PREVIOUS_WEIGHT < 1) }"; then
  echo "CONTEXT_MOMENTUM_PREVIOUS_WEIGHT must be in [0,1), got: $CONTEXT_MOMENTUM_PREVIOUS_WEIGHT" >&2
  exit 2
fi

MOTION_ARGS=(--plan_only)
if [[ "$RG2_ENABLE_MOTION" == "1" ]]; then
  MOTION_ARGS=(--execute)
  echo "Motion enabled. No terminal EXECUTE text is required."
else
  echo "RG2_ENABLE_MOTION=$RG2_ENABLE_MOTION: plan-only; waypoints will not be sent."
fi

START_ARGS=(--wait_for_start_key)
if [[ "$AUTO_START_POLICY" == "1" ]]; then
  START_ARGS=(--auto_start_policy)
  echo "Policy will start automatically after model warmup."
else
  echo "Use keyboard teleop to align the live/train overlap, then press c to start."
fi

TARE_ARGS=(--zero_ft_on_start)
if [[ "$RG2_ZERO_FT_ON_START" == "0" ]]; then
  TARE_ARGS=(--no_zero_ft_on_start)
  echo "[warn] Startup F/T software tare is disabled."
else
  echo "Startup F/T zero: ${FT_ZERO_SAMPLES} recent 100 Hz samples; keep both fingers unloaded and still."
fi

SAVE_ARGS=(--save_inputs)
if [[ "$RG2_SAVE_POLICY_INPUTS" == "0" ]]; then
  SAVE_ARGS=()
fi

echo "Action checkpoint:  $ACTION_CHECKPOINT"
echo "Context checkpoint: $CONTEXT_CHECKPOINT"
echo "Match dataset:      $MATCH_DATASET (episode $MATCH_EPISODE)"
echo "F/T age limits:     checkpoint window=12 ms, latest command guard=${RG2_LATEST_FT_MAX_AGE_SEC}s"
echo "Result age limits:  total=${RG2_MAX_RESULT_AGE_SEC}s, after expected camera latency=${RG2_MAX_PROCESSING_AGE_SEC}s"
echo "Legacy F/T value:   RG2_FT_MAX_AGE_SEC=${RG2_FT_MAX_AGE_SEC}s (does not override checkpoint contract)"
echo "Output base:        $OUTPUT_DIR"
echo "Motion momentum:    previous=$MOTION_MOMENTUM_PREVIOUS_WEIGHT policy=$(awk "BEGIN { print 1 - $MOTION_MOMENTUM_PREVIOUS_WEIGHT }")"
echo "Context momentum:   previous=$CONTEXT_MOMENTUM_PREVIOUS_WEIGHT current=$(awk "BEGIN { print 1 - $CONTEXT_MOMENTUM_PREVIOUS_WEIGHT }")"

exec "$PYTHON_BIN" "$ROOT/eval_real_slip_action.py" \
  --action_checkpoint "$ACTION_CHECKPOINT" \
  --context_checkpoint "$CONTEXT_CHECKPOINT" \
  --robot_config "$ROBOT_CONFIG" \
  --aruco_config "$ARUCO_CONFIG" \
  --match_dataset "$MATCH_DATASET" \
  --match_episode "$MATCH_EPISODE" \
  --output "$OUTPUT_DIR" \
  --action_scale "$ACTION_SCALE" \
  --motion_momentum_previous_weight "$MOTION_MOMENTUM_PREVIOUS_WEIGHT" \
  --context_momentum_previous_weight "$CONTEXT_MOMENTUM_PREVIOUS_WEIGHT" \
  --ft_max_age_sec "$RG2_FT_MAX_AGE_SEC" \
  --latest_ft_max_age_sec "$RG2_LATEST_FT_MAX_AGE_SEC" \
  --max_result_age "$RG2_MAX_RESULT_AGE_SEC" \
  --max_processing_age "$RG2_MAX_PROCESSING_AGE_SEC" \
  --ft_zero_samples "$FT_ZERO_SAMPLES" \
  --max_policy_iters "$MAX_POLICY_ITERS" \
  --max_duration "$MAX_DURATION" \
  "${MOTION_ARGS[@]}" \
  "${START_ARGS[@]}" \
  "${TARE_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "$@"
