# Transaction-Cost Sensitivity

This folder contains evaluation-layer transaction-cost sensitivity results for the frozen final strategies.

## Setup
- Strategies: Elastic Net, Random Forest, Torch (gamma=5.0).
- Cost convention: one-way turnover.
- Cost levels: 0, 10, 25, and 50 basis points.
- Models are not retrained; costs are applied to stored final weights and realised returns.

## Summary
| strategy | cost_bps | annual_return | annual_volatility | sharpe | max_drawdown | avg_turnover | cumulative_return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Elastic Net | 0.0 | 0.027398 | 0.302869 | 0.184818 | -0.735387 | 0.563031 | 0.965487 |
| Elastic Net | 10.0 | 0.020477 | 0.302823 | 0.162557 | -0.751520 | 0.563031 | 0.659893 |
| Elastic Net | 25.0 | 0.010174 | 0.302757 | 0.129151 | -0.773911 | 0.563031 | 0.287972 |
| Elastic Net | 50.0 | -0.006788 | 0.302655 | 0.073442 | -0.806875 | 0.563031 | -0.156574 |
| Random Forest | 0.0 | 0.089429 | 0.223740 | 0.416871 | -0.511382 | 0.733938 | 7.510787 |
| Random Forest | 10.0 | 0.079921 | 0.223709 | 0.377647 | -0.523004 | 0.733938 | 5.835903 |
| Random Forest | 25.0 | 0.065800 | 0.223667 | 0.318784 | -0.539938 | 0.733938 | 3.919157 |
| Random Forest | 50.0 | 0.042637 | 0.223612 | 0.220619 | -0.566884 | 0.733938 | 1.840068 |
| Torch (gamma=5.0) | 0.0 | 0.134830 | 0.193328 | 0.663060 | -0.565771 | 0.533884 | 22.619535 |
| Torch (gamma=5.0) | 10.0 | 0.127643 | 0.193348 | 0.629912 | -0.571349 | 0.533884 | 19.150644 |
| Torch (gamma=5.0) | 25.0 | 0.116939 | 0.193382 | 0.580187 | -0.579587 | 0.533884 | 14.875914 |
| Torch (gamma=5.0) | 50.0 | 0.099304 | 0.193449 | 0.497312 | -0.592985 | 0.533884 | 9.664732 |

## 0 bps validation
| strategy | annual_return_abs_diff | sharpe_abs_diff | avg_turnover_abs_diff |
| --- | --- | --- | --- |
| Elastic Net | 8.674e-17 | 2.776e-17 | 0.000e+00 |
| Random Forest | 0.000e+00 | 5.551e-17 | 0.000e+00 |
| Torch (gamma=5.0) | 8.327e-17 | 0.000e+00 | 0.000e+00 |
