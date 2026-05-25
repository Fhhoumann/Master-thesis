from __future__ import annotations

import pandas as pd

from core.config.schemas import CVConfig
from core.cv import get_time_splits


def test_expanding_splitter_schedule() -> None:
    dates = pd.date_range("2010-01-31", periods=10, freq="ME")
    frame = pd.DataFrame({"date": dates})
    cfg = CVConfig(scheme="expanding", train_min_periods=5, validation_periods=1, step_periods=2)

    splits = get_time_splits(frame, "date", cfg)
    assert len(splits) == 3
    assert len(splits[0].train_dates) == 5
    assert len(splits[0].validation_dates) == 1
    assert splits[0].validation_dates[0] == dates[5]


def test_rolling_splitter_window_size() -> None:
    dates = pd.date_range("2010-01-31", periods=12, freq="ME")
    frame = pd.DataFrame({"date": dates})
    cfg = CVConfig(
        scheme="rolling",
        train_min_periods=4,
        validation_periods=2,
        step_periods=2,
        train_window_periods=5,
    )

    splits = get_time_splits(frame, "date", cfg)
    assert len(splits) > 0
    assert all(len(split.train_dates) <= 5 for split in splits)
    assert all(len(split.validation_dates) == 2 for split in splits)
