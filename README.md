# Data Research Framework

Research code and final empirical outputs for point-in-time equity panel experiments.

## Repository Contents

- `apps/`, `core/`, `scripts/`, `tests/`: source code, command-line entry points, and tests.
- `configs/`: dataset, feature, backtest, and experiment configurations.
- `notebooks/`: notebooks used for visualisation and descriptive analysis.
- `FINAL_RUNS_MANIFEST.md`: map of the frozen final model runs and analysis outputs.
- `artifacts/`: final run outputs and selected result artifacts used by the thesis.
- `reports/`: generated figures, tables, and report-facing outputs.

## Local-Only Files

Raw data, local archives, virtual environments, caches, run logs, and machine-specific runtime state are intentionally excluded through `.gitignore`. They remain available locally but are not suitable for GitHub upload.

The raw `DataCube_1990_2024.mat` file is not tracked. Reproducing the full pipeline requires access to the underlying data source.
