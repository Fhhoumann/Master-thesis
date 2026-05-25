from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from core.config.schemas import (
    BacktestConfig,
    CVConfig,
    DatasetConfig,
    EDAConfig,
    ExperimentConfig,
    FeatureConfig,
    OneStepModelConfig,
    TwoStepModelConfig,
    TrackingConfig,
)

TConfig = TypeVar("TConfig", bound=BaseModel)


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")
    return raw


def load_typed_config(path: str | Path, schema: type[TConfig]) -> TConfig:
    raw = load_yaml(path)
    return schema.model_validate(raw)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return load_typed_config(path, DatasetConfig)


def load_feature_config(path: str | Path) -> FeatureConfig:
    return load_typed_config(path, FeatureConfig)


def load_cv_config(path: str | Path) -> CVConfig:
    return load_typed_config(path, CVConfig)


def load_twostep_config(path: str | Path) -> TwoStepModelConfig:
    return load_typed_config(path, TwoStepModelConfig)


def load_onestep_config(path: str | Path) -> OneStepModelConfig:
    return load_typed_config(path, OneStepModelConfig)


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return load_typed_config(path, BacktestConfig)


def load_tracking_config(path: str | Path) -> TrackingConfig:
    return load_typed_config(path, TrackingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_typed_config(path, ExperimentConfig)


def load_eda_config(path: str | Path) -> EDAConfig:
    return load_typed_config(path, EDAConfig)
