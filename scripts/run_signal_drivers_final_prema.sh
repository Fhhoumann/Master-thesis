#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/run_logs
export MPLCONFIGDIR="${PWD}/artifacts/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

LOG_PATH="artifacts/run_logs/signal_drivers_final_prema.log"

echo "[$(date)] Starting appendix signal-driver analysis" | tee -a "${LOG_PATH}"
./.venv/bin/python scripts/analyze_signal_drivers.py \
  --outdir artifacts/appendix_signal_drivers_final_prema \
  --torch-permutation-repeats 3 \
  >> "${LOG_PATH}" 2>&1
echo "[$(date)] Finished appendix signal-driver analysis" | tee -a "${LOG_PATH}"
