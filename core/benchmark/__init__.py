"""Benchmark portfolios and factor attribution utilities."""

from core.benchmark.attribution import compute_attribution_table, run_factor_regressions
from core.benchmark.portfolios import build_ew_market, build_hml_proxy, build_umd_momentum
from core.benchmark.sharpe_tests import (
    build_pairwise_sharpe_test_table,
    calibrate_block_size_algorithm_3_1,
    ledoit_wolf_sharpe_difference_test,
    prepare_pairwise_excess_returns,
)

__all__ = [
    "build_ew_market",
    "build_hml_proxy",
    "build_umd_momentum",
    "compute_attribution_table",
    "run_factor_regressions",
    "prepare_pairwise_excess_returns",
    "ledoit_wolf_sharpe_difference_test",
    "calibrate_block_size_algorithm_3_1",
    "build_pairwise_sharpe_test_table",
]
