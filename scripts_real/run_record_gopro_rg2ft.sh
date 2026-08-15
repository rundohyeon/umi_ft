#!/usr/bin/env bash
# Launch the GoPro(capture card) + RG2-FT synchronized recorder.
#
# Usage:
#   scripts_real/run_record_gopro_rg2ft.sh <session_dir> [extra args...]
# Example:
#   scripts_real/run_record_gopro_rg2ft.sh ~/umi_data/session_20260722
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <session_dir> [extra args for record_gopro_rg2ft.py]" >&2
  exit 1
fi
SESSION="$1"; shift

# activate the recording env (pymodbus-free raw-socket modbus + av + opencv)
source /home/idim/anaconda3/etc/profile.d/conda.sh
conda activate umi-rec

exec python "$ROOT/scripts_real/record_gopro_rg2ft.py" -o "$SESSION" "$@"
