# Final Runs Manifest

This manifest identifies the frozen final artifacts used for the empirical results in the thesis. It is intended to help readers locate the final model runs, result tables, figures, and descriptive notebook outputs without relying on intermediate development artifacts.

## Main Empirical Specification

The final empirical specification is the final setup:

- `tsmom_sign_12` is included and is not winsorized.
- Models are trained with a 120-month rolling training window.
- Portfolios are rebalanced monthly.
- The out-of-sample period runs from December 1999 to December 2024.
- The main comparison uses 0 bps transaction costs.
- Positive transaction-cost scenarios are evaluation-layer sensitivity checks based on the stored final portfolio weights.

## Final Model Run Folders

The final model runs used in the main empirical comparison are:

- Elastic Net two-step model:
  `artifacts/macroext_twostep_elasticnet_supervisor_constrained_final_prema_v1/twostep/`
- Random Forest two-step model:
  `artifacts/macroext_twostep_random_forest_supervisor_constrained_final_prema_v1/twostep/`
- Torch one-step model, gamma 5.0:
  `artifacts/macroext_onestep_torch_supervisor_ra500_final_prema_v1/onestep/`

## Final Analysis Output Folders

The final analysis outputs are stored in:

- Factor attribution:
  `artifacts/factor_attribution_final_prema/`
- Pairwise Sharpe tests:
  `artifacts/step3_sharpe_tests_final_prema/`
- Transaction-cost sensitivity:
  `artifacts/transaction_cost_sensitivity_final_prema/`
- Appendix signal-driver outputs:
  `artifacts/appendix_signal_drivers_final_prema/`
- Notebook-exported tables:
  `artifacts/tables/notebooks/`

## Final Figures

Final thesis figures are stored in:

- `artifacts/figures/`

This folder includes the final model comparison, cumulative return, rolling return, gamma-grid, and notebook-derived correlation figures used for the thesis presentation of results and descriptive evidence.

## Notebook-Based Descriptive Outputs

The notebook used for descriptive correlation-matrix outputs is:

- `notebooks/correlation_matrices.ipynb`

Notebook-derived descriptive figures are stored under:

- `artifacts/figures/notebooks/`

Notebook-derived tables are stored under:

- `artifacts/tables/notebooks/`

## Note on Other Artifacts

Other non-final artifacts, if present, should be interpreted as development history, robustness checks, diagnostic outputs, or intermediate results. They are not used as main thesis evidence unless explicitly referenced in the thesis text.
