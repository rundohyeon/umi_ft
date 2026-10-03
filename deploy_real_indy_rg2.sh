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
MATCH_DATASET="${MATCH_DATASET:-$ROOT/data/dataset_ft.zarr.zip}"
MATCH_EPISODE="${MATCH_EPISODE:-0}"
ACTION_SCALE="${ACTION_SCALE:-0.2}"
RG2_MOTION_MOMENTUM_PREVIOUS_WEIGHT="${RG2_MOTION_MOMENTUM_PREVIOUS_WEIGHT:-0}"
RG2_SAVE_FUSION_ATTENTION="${RG2_SAVE_FUSION_ATTENTION:-1}"
RG2_SAVE_CONTEXT_INPUTS="${RG2_SAVE_CONTEXT_INPUTS:-1}"
RG2_SAVE_POLICY_INPUTS="${RG2_SAVE_POLICY_INPUTS:-1}"
RG2_FT_MAX_AGE_SEC="${RG2_FT_MAX_AGE_SEC:-0.020}"
VALVE_CLASSIFIER_CHECKPOINT="${VALVE_CLASSIFIER_CHECKPOINT:-}"
VALVE_CLASSIFIER_DEVICE="${VALVE_CLASSIFIER_DEVICE:-auto}"
VALVE_CLASSIFIER_ALLOW_OVERRIDE="${VALVE_CLASSIFIER_ALLOW_OVERRIDE:-0}"

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

if ! [[ "$RG2_FT_MAX_AGE_SEC" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN { exit !($RG2_FT_MAX_AGE_SEC > 0) }"; then
  echo "RG2_FT_MAX_AGE_SEC must be a positive number of seconds, got: $RG2_FT_MAX_AGE_SEC" >&2
  exit 2
fi
echo "RG2-FT causal freshness limit: ${RG2_FT_MAX_AGE_SEC}s"

RG2_ENABLE_MOTION="${RG2_ENABLE_MOTION:-0}"
SAFETY_ARGS=(--plan_only --print_motion_debug)
if [[ "$RG2_ENABLE_MOTION" == "1" ]]; then
  SAFETY_ARGS=()
  echo "Motion default: robot and RG2-FT motion are enabled."
else
  echo "RG2_ENABLE_MOTION=$RG2_ENABLE_MOTION: plan-only; RG2-FT connects read-only and receives no width command."
fi

DIAGNOSTIC_ARGS=()
case "${RG2_SAVE_FUSION_ATTENTION,,}" in
  1|true|yes|on)
    DIAGNOSTIC_ARGS+=(--save_fusion_attention)
    echo "Eval diagnostics: F/T input, full policy outputs, comparison video, and fusion attention will be logged."
    ;;
  0|false|no|off)
    echo "Eval diagnostics: F/T input, full policy outputs, and comparison video will be logged (fusion attention disabled)."
    ;;
  *)
    echo "RG2_SAVE_FUSION_ATTENTION must be 0/1 or true/false, got: $RG2_SAVE_FUSION_ATTENTION" >&2
    exit 2
    ;;
esac

CONTEXT_CAPTURE_ARGS=()
case "${RG2_SAVE_CONTEXT_INPUTS,,}" in
  1|true|yes|on)
    CONTEXT_CAPTURE_ARGS+=(--save_context_inputs)
    echo "Context diagnostics: every classifier RGB/TCP/F-T temporal window and output will be logged."
    ;;
  0|false|no|off)
    CONTEXT_CAPTURE_ARGS+=(--no_save_context_inputs)
    echo "Context diagnostics: exact classifier-input capture disabled."
    ;;
  *)
    echo "RG2_SAVE_CONTEXT_INPUTS must be 0/1 or true/false, got: $RG2_SAVE_CONTEXT_INPUTS" >&2
    exit 2
    ;;
esac

POLICY_CAPTURE_ARGS=()
case "${RG2_SAVE_POLICY_INPUTS,,}" in
  1|true|yes|on)
    POLICY_CAPTURE_ARGS+=(--save_policy_inputs)
    echo "Policy input capture: exact pre-normalizer RGB, TCP/rotation-6D, causal F/T, and context will be logged."
    ;;
  0|false|no|off)
    POLICY_CAPTURE_ARGS+=(--no_save_policy_inputs)
    echo "Policy input capture disabled."
    ;;
  *)
    echo "RG2_SAVE_POLICY_INPUTS must be 0/1 or true/false, got: $RG2_SAVE_POLICY_INPUTS" >&2
    exit 2
    ;;
esac

# The evaluator ignores these options for the ordinary dual-F/T checkpoint.
# For a context checkpoint the evaluator normally resolves the observer path
# from the serialized policy config and validates its SHA before motion. Set
# VALVE_CLASSIFIER_CHECKPOINT only to override that path.
VALVE_CONTEXT_ARGS=()
if [[ -n "$VALVE_CLASSIFIER_CHECKPOINT" && -f "$VALVE_CLASSIFIER_CHECKPOINT" ]]; then
  VALVE_CONTEXT_ARGS=(
    --valve_classifier_checkpoint "$VALVE_CLASSIFIER_CHECKPOINT"
    --valve_classifier_device "$VALVE_CLASSIFIER_DEVICE"
  )
  case "${VALVE_CLASSIFIER_ALLOW_OVERRIDE,,}" in
    1|true|yes|on)
      VALVE_CONTEXT_ARGS+=(--allow_valve_classifier_override)
      echo "[warn] Explicit RGB/F-T classifier override enabled; policy observer SHA identity will differ."
      ;;
    0|false|no|off)
      ;;
    *)
      echo "VALVE_CLASSIFIER_ALLOW_OVERRIDE must be 0/1 or true/false, got: $VALVE_CLASSIFIER_ALLOW_OVERRIDE" >&2
      exit 2
      ;;
  esac
elif [[ -n "$VALVE_CLASSIFIER_CHECKPOINT" ]]; then
  echo "[warn] valve classifier not found at $VALVE_CLASSIFIER_CHECKPOINT" >&2
  echo "[warn] this is fatal only when RG2_CHECKPOINT is a valve-context policy." >&2
fi

exec "$PYTHON_BIN" "$ROOT/eval_real_indy_rg2.py" \
  --input "$CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --robot_config "$ROBOT_CONFIG" \
  --match_dataset "$MATCH_DATASET" \
  --match_episode "$MATCH_EPISODE" \
  --allow_rotation \
  --action_scale "$ACTION_SCALE" \
  --motion_momentum_previous_weight "$RG2_MOTION_MOMENTUM_PREVIOUS_WEIGHT" \
  --ft_max_age_sec "$RG2_FT_MAX_AGE_SEC" \
  --vis_pose \
  "${DIAGNOSTIC_ARGS[@]}" \
  "${CONTEXT_CAPTURE_ARGS[@]}" \
  "${POLICY_CAPTURE_ARGS[@]}" \
  "${VALVE_CONTEXT_ARGS[@]}" \
  "${SAFETY_ARGS[@]}" \
  "$@"
