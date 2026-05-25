from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictBaseModel):
    datacube_path: Path = Field(default=Path("DataCube_1990_2024.mat"))
    panel_output_dir: Path = Field(default=Path("artifacts/datasets/panel"))
    sample_assets: PositiveInt | None = None
    sample_time: PositiveInt | None = None


class FeatureConfig(StrictBaseModel):
    construction_mode: Literal["panel_rolling", "direct_cube"] = "direct_cube"
    datacube_path: Path = Field(default=Path("DataCube_1990_2024.mat"))
    panel_input_dir: Path = Field(default=Path("artifacts/datasets/panel"))
    feature_output_dir: Path = Field(default=Path("artifacts/features"))
    macro_factors_path: Path | None = None
    macro_sheet_name: str | int = 0
    macro_columns: list[str] = Field(
        default_factory=lambda: [
            "macro_vix",
            "macro_epu",
            "macro_credit_spread",
            "macro_term_spread",
        ]
    )
    macro_lag_periods: int = Field(default=0, ge=0, le=12)
    winsorize_lower: float = Field(default=0.01, ge=0.0, le=0.25)
    winsorize_upper: float = Field(default=0.99, ge=0.75, le=1.0)
    min_history: PositiveInt = Field(default=26)
    rsi_window: PositiveInt = Field(default=14)
    sma_windows: list[PositiveInt] = Field(default=[3, 6, 12])
    volatility_windows: list[PositiveInt] = Field(default=[3, 6, 12])
    bollinger_window: PositiveInt = Field(default=20)


class CVConfig(StrictBaseModel):
    scheme: Literal["expanding", "rolling"] = "expanding"
    train_min_periods: PositiveInt = 60
    validation_periods: PositiveInt = 1
    step_periods: PositiveInt = 1
    train_window_periods: PositiveInt | None = None


class TwoStepModelConfig(StrictBaseModel):
    model_type: Literal["elasticnet", "random_forest", "lightgbm"] = "elasticnet"
    params: dict[str, Any] = Field(default_factory=dict)
    feature_columns: list[str] | None = None
    feature_set_name: str | None = None
    target_column: str = "target_ret_1m"


class OneStepModelConfig(StrictBaseModel):
    feature_columns: list[str] | None = None
    feature_set_name: str | None = None
    target_column: str = "target_ret_1m"
    learning_rate: PositiveFloat = 0.01
    epochs: PositiveInt = 200
    batch_dates: PositiveInt = 24
    l2_penalty: float = Field(default=1e-4, ge=0.0)
    turnover_penalty: float = Field(default=0.1, ge=0.0)
    risk_aversion: float = Field(default=0.1, ge=0.0)
    weight_clip: float = Field(default=0.10, gt=0.0, le=1.0)
    selection_quantile: float = Field(default=0.10, gt=0.0, lt=1.0)
    random_seed: int = 42


class BacktestConfig(StrictBaseModel):
    top_quantile: float = Field(default=0.10, gt=0.0, lt=0.5)
    two_step_portfolio_rule: Literal["equal_weight_top_quantile", "constrained_mean_variance"] = (
        "equal_weight_top_quantile"
    )
    two_step_weight_cap: float = Field(default=0.10, gt=0.0, le=1.0)
    two_step_mv_risk_aversion: float = Field(default=5.0, gt=0.0)
    two_step_covariance_shrinkage: float = Field(default=0.50, ge=0.0, le=1.0)
    two_step_optimizer_steps: PositiveInt = Field(default=250)
    two_step_optimizer_step_size: PositiveFloat = Field(default=0.05)
    transaction_cost_bps: list[NonNegativeFloat] = Field(default=[10.0, 25.0, 50.0])
    conventions: list[Literal["one_way", "round_trip"]] = Field(
        default=["one_way", "round_trip"]
    )
    initial_turnover_from_zero: bool = True
    turnover_use_effective_weights: bool = True
    missing_return_policy: Literal["exclude_renorm", "fill_zero"] = Field(
        default="exclude_renorm"
    )


class TrackingConfig(StrictBaseModel):
    enabled: bool = True
    tracking_uri: str = Field(default="sqlite:///artifacts/mlflow.db")
    experiment_name: str = Field(default="fhdata-experiments")


class ExperimentConfig(StrictBaseModel):
    name: str
    enforce_feature_consistency: bool = False
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    twostep: TwoStepModelConfig = Field(default_factory=TwoStepModelConfig)
    onestep: OneStepModelConfig = Field(default_factory=OneStepModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class RegimeConfig(StrictBaseModel):
    name: str
    start: date
    end: date

    @model_validator(mode="after")
    def _validate_date_order(self) -> "RegimeConfig":
        if self.start > self.end:
            raise ValueError(
                f"Regime `{self.name}` has start > end ({self.start} > {self.end})."
            )
        return self


class EDAConfig(StrictBaseModel):
    datacube_path: Path = Field(default=Path("DataCube_1990_2024.mat"))
    panel_input_path: Path = Field(default=Path("artifacts/datasets/panel/panel.parquet"))
    feature_input_path: Path = Field(default=Path("artifacts/features/features.parquet"))
    artifacts_root: Path = Field(default=Path("artifacts"))
    output_dir: Path = Field(default=Path("reports/eda"))
    factor_cache_path: Path = Field(default=Path("artifacts/factors/ken_french_monthly.parquet"))
    momentum_factor_cache_path: Path = Field(
        default=Path("artifacts/factors/ken_french_momentum_monthly.parquet")
    )
    factor_refresh: bool = False
    regression_min_obs: PositiveInt = Field(default=24)
    benchmark_top_quantile: float = Field(default=0.10, gt=0.0, lt=0.5)
    benchmark_return_types: list[Literal["gross", "net"]] = Field(default=["gross", "net"])
    benchmark_models: list[Literal["capm", "ff3", "carhart4", "ff5"]] = Field(
        default=["capm", "ff3", "ff5"]
    )
    mode: Literal["full", "quick"] = Field(default="full")
    quick_sample_assets: PositiveInt = Field(default=250)
    quick_sample_time: PositiveInt = Field(default=240)
    baseline_cost_convention: Literal["one_way", "round_trip"] = Field(default="one_way")
    baseline_cost_bps: PositiveFloat = Field(default=25.0)
    rolling_window_months: PositiveInt = Field(default=12)
    min_cross_section_count: PositiveInt = Field(default=50)
    top_n_features: PositiveInt = Field(default=6)
    regimes: list[RegimeConfig] = Field(
        default_factory=lambda: [
            RegimeConfig(name="1990s", start=date(1990, 1, 1), end=date(1999, 12, 31)),
            RegimeConfig(name="2000+", start=date(2000, 1, 1), end=date(2024, 12, 31)),
        ]
    )
