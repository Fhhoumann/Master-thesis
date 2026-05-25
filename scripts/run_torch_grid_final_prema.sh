#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/run_logs
export MPLCONFIGDIR="${PWD}/artifacts/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

LOG_DIR="artifacts/run_logs"

run_config() {
  local config_path="$1"
  local experiment_name="$2"
  local log_path="${LOG_DIR}/${experiment_name}.log"
  local summary_path="artifacts/${experiment_name}/onestep/summary.csv"

  if [[ -f "${summary_path}" ]]; then
    echo "[$(date)] Skipping ${experiment_name} (summary already present)" | tee -a "${log_path}"
    return 0
  fi

  echo "[$(date)] Starting ${config_path}" | tee -a "${log_path}"
  ./.venv/bin/python apps/train_onestep/main.py --config "${config_path}" >> "${log_path}" 2>&1
  echo "[$(date)] Finished ${config_path}" | tee -a "${log_path}"
}

echo "[$(date)] Starting frozen pre-MA Torch gamma-grid pipeline"

run_config "configs/experiments/macroext_onestep_torch_supervisor_ra010_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra010_final_prema_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra050_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra050_final_prema_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra100_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra100_final_prema_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra200_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra200_final_prema_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra300_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra300_final_prema_v1"
run_config "configs/experiments/macroext_onestep_torch_supervisor_ra500_final_prema_v1.yaml" "macroext_onestep_torch_supervisor_ra500_final_prema_v1"

echo "[$(date)] Frozen pre-MA Torch gamma-grid pipeline completed"
