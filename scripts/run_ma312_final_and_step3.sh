#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/run_logs
export MPLCONFIGDIR="${PWD}/artifacts/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

EN_CONFIG="configs/experiments/macroext_twostep_elasticnet_supervisor_constrained_v1.yaml"
RF_CONFIG="configs/experiments/macroext_twostep_random_forest_supervisor_constrained_v1.yaml"
TORCH_CONFIG="configs/experiments/macroext_onestep_torch_supervisor_ra500_v1.yaml"

STEP3_OUTDIR="artifacts/step3_sharpe_tests_selected_calibrated_ma312_fixedsignals"

echo "[$(date)] Starting MA(3,12) rerun pipeline with fixed feature signals"

echo "[$(date)] Running Elastic Net constrained final spec"
./.venv/bin/python apps/train_twostep/main.py --config "${EN_CONFIG}"

echo "[$(date)] Running Random Forest constrained final spec"
./.venv/bin/python apps/train_twostep/main.py --config "${RF_CONFIG}"

echo "[$(date)] Running Torch gamma 5.0 final spec"
./.venv/bin/python apps/train_onestep/main.py --config "${TORCH_CONFIG}"

echo "[$(date)] Running updated Step 3 Sharpe-difference tests"
./.venv/bin/python apps/sharpe_tests/main.py \
  --enet-backtest artifacts/macroext_twostep_elasticnet_supervisor_constrained_v1/twostep/backtest.parquet \
  --rf-backtest artifacts/macroext_twostep_random_forest_supervisor_constrained_v1/twostep/backtest.parquet \
  --torch-backtest artifacts/macroext_onestep_torch_supervisor_ra500_v1/onestep/backtest.parquet \
  --outdir "${STEP3_OUTDIR}" \
  --factor-cache artifacts/factors/ken_french_monthly.parquet \
  --cost-convention one_way \
  --cost-bps 0.0 \
  --bootstrap-resamples 999 \
  --alpha 0.05 \
  --seed 42 \
  --calibrate-block-size \
  --candidate-block-sizes 1,2,4,6,8,10 \
  --calibration-pseudo-sequences 49 \
  --calibration-bootstrap-resamples 99 \
  --residual-bootstrap-avg-block-size 5.0

echo "[$(date)] Overnight MA(3,12) rerun pipeline completed"
