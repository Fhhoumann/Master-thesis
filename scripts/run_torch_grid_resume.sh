#!/bin/sh

set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="artifacts/run_logs"
MPL_DIR="artifacts/.matplotlib_cache"
mkdir -p "$LOG_DIR" "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

run_config() {
  config_path="$1"
  experiment_name="$2"
  log_path="$LOG_DIR/${experiment_name}.log"
  summary_path="artifacts/${experiment_name}/onestep/summary.csv"

  if [ -f "$summary_path" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping ${experiment_name} (already finished)"
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config_path}" | tee -a "$log_path"
  ./.venv/bin/python apps/train_onestep/main.py --config "$config_path" >> "$log_path" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config_path}" | tee -a "$log_path"
}

run_config "configs/experiments/macroext_onestep_torch_supervisor_ra050_v1.yaml" "macroext_onestep_torch_supervisor_ra050_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra100_v1.yaml" "macroext_onestep_torch_supervisor_ra100_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra200_v1.yaml" "macroext_onestep_torch_supervisor_ra200_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra300_v1.yaml" "macroext_onestep_torch_supervisor_ra300_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra500_v1.yaml" "macroext_onestep_torch_supervisor_ra500_v1"
