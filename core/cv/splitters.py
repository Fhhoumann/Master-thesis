from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config.schemas import CVConfig


@dataclass(slots=True)
class TimeSeriesSplit:
    train_dates: np.ndarray
    validation_dates: np.ndarray
    fold_id: int


class BaseTimeSeriesSplitter:
    def __init__(
        self,
        train_min_periods: int,
        validation_periods: int,
        step_periods: int,
    ) -> None:
        self.train_min_periods = int(train_min_periods)
        self.validation_periods = int(validation_periods)
        self.step_periods = int(step_periods)

    def split(self, unique_dates: np.ndarray) -> list[TimeSeriesSplit]:
        raise NotImplementedError


class ExpandingWindowSplitter(BaseTimeSeriesSplitter):
    def split(self, unique_dates: np.ndarray) -> list[TimeSeriesSplit]:
        n_dates = len(unique_dates)
        splits: list[TimeSeriesSplit] = []
        fold_id = 0
        for val_start in range(
            self.train_min_periods,
            n_dates - self.validation_periods + 1,
            self.step_periods,
        ):
            train_dates = unique_dates[:val_start]
            validation_dates = unique_dates[val_start : val_start + self.validation_periods]
            splits.append(
                TimeSeriesSplit(
                    train_dates=train_dates,
                    validation_dates=validation_dates,
                    fold_id=fold_id,
                )
            )
            fold_id += 1
        return splits


class RollingWindowSplitter(BaseTimeSeriesSplitter):
    def __init__(
        self,
        train_min_periods: int,
        validation_periods: int,
        step_periods: int,
        train_window_periods: int,
    ) -> None:
        super().__init__(train_min_periods, validation_periods, step_periods)
        self.train_window_periods = int(train_window_periods)

    def split(self, unique_dates: np.ndarray) -> list[TimeSeriesSplit]:
        n_dates = len(unique_dates)
        splits: list[TimeSeriesSplit] = []
        fold_id = 0
        for val_start in range(
            self.train_min_periods,
            n_dates - self.validation_periods + 1,
            self.step_periods,
        ):
            train_start = max(0, val_start - self.train_window_periods)
            train_dates = unique_dates[train_start:val_start]
            if len(train_dates) < self.train_min_periods:
                continue
            validation_dates = unique_dates[val_start : val_start + self.validation_periods]
            splits.append(
                TimeSeriesSplit(
                    train_dates=train_dates,
                    validation_dates=validation_dates,
                    fold_id=fold_id,
                )
            )
            fold_id += 1
        return splits


def build_splitter(config: CVConfig) -> BaseTimeSeriesSplitter:
    if config.scheme == "expanding":
        return ExpandingWindowSplitter(
            train_min_periods=config.train_min_periods,
            validation_periods=config.validation_periods,
            step_periods=config.step_periods,
        )

    train_window = config.train_window_periods or config.train_min_periods
    return RollingWindowSplitter(
        train_min_periods=config.train_min_periods,
        validation_periods=config.validation_periods,
        step_periods=config.step_periods,
        train_window_periods=train_window,
    )


def get_time_splits(frame: pd.DataFrame, date_col: str, config: CVConfig) -> list[TimeSeriesSplit]:
    unique_dates = np.sort(frame[date_col].dropna().unique())
    splitter = build_splitter(config)
    return splitter.split(unique_dates)

