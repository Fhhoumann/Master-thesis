"""Performance metrics and reporting utilities."""

from core.metrics.performance import (
    infer_periods_per_year,
    max_drawdown,
    summarize_backtest_frame,
    summarize_returns,
)

__all__ = ["infer_periods_per_year", "max_drawdown", "summarize_backtest_frame", "summarize_returns"]
