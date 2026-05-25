#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="artifacts/run_logs"
mkdir -p "$LOG_DIR"

run_experiment() {
  local config_path="$1"
  local log_path="$2"
  local pipeline="$3"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config_path}" | tee -a "$log_path"
  if [[ "$pipeline" == "twostep" ]]; then
    ./.venv/bin/python apps/train_twostep/main.py --config "$config_path" >> "$log_path" 2>&1
  else
    ./.venv/bin/python apps/train_onestep/main.py --config "$config_path" >> "$log_path" 2>&1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config_path}" | tee -a "$log_path"
}

run_experiment \
  "configs/experiments/macroext_twostep_elasticnet_supervisor_constrained_v1.yaml" \
  "$LOG_DIR/macroext_twostep_elasticnet_supervisor_constrained_v1.log" \
  "twostep"

run_experiment \
  "configs/experiments/macroext_twostep_random_forest_supervisor_constrained_v1.yaml" \
  "$LOG_DIR/macroext_twostep_random_forest_supervisor_constrained_v1.log" \
  "twostep"

run_experiment \
  "configs/experiments/macroext_onestep_torch_supervisor_ra050_v1.yaml" \
  "$LOG_DIR/macroext_onestep_torch_supervisor_ra050_v1.log" \
  "onestep"

run_experiment \
  "configs/experiments/macroext_onestep_torch_supervisor_ra100_v1.yaml" \
  "$LOG_DIR/macroext_onestep_torch_supervisor_ra100_v1.log" \
  "onestep"

run_experiment \
  "configs/experiments/macroext_onestep_torch_supervisor_ra200_v1.yaml" \
  "$LOG_DIR/macroext_onestep_torch_supervisor_ra200_v1.log" \
  "onestep"

run_experiment \
  "configs/experiments/macroext_onestep_torch_supervisor_ra300_v1.yaml" \
  "$LOG_DIR/macroext_onestep_torch_supervisor_ra300_v1.log" \
  "onestep"

run_experiment \
  "configs/experiments/macroext_onestep_torch_supervisor_ra500_v1.yaml" \
  "$LOG_DIR/macroext_onestep_torch_supervisor_ra500_v1.log" \
  "onestep"
