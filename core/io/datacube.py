from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataCube:
    lookback: int
    month_end_raw: np.ndarray
    month_end: list[pd.Timestamp | None]
    permno: np.ndarray
    returns: np.ndarray
    volume: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.returns.shape)

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_lookback(self) -> int:
        return int(self.returns.shape[1])

    @property
    def n_time(self) -> int:
        return int(self.returns.shape[2])

    @property
    def current_returns(self) -> np.ndarray:
        return self.returns[:, -1, :]

    @property
    def current_volume(self) -> np.ndarray:
        return self.volume[:, -1, :]


def _parse_month_end_value(value: float) -> pd.Timestamp | None:
    if not np.isfinite(value):
        return None

    rounded = int(round(float(value)))
    raw = str(abs(rounded))
    if len(raw) == 8:
        try:
            return pd.Timestamp(datetime.strptime(str(rounded), "%Y%m%d"))
        except ValueError:
            pass

    try:
        ordinal = int(value)
        dt = datetime.fromordinal(ordinal) + timedelta(days=float(value) % 1) - timedelta(days=366)
        return pd.Timestamp(dt)
    except (OverflowError, ValueError):
        return None


def _validate_axis_alignment(
    permno: np.ndarray, returns: np.ndarray, volume: np.ndarray, month_end: np.ndarray
) -> None:
    if permno.ndim != 2:
        raise ValueError(f"`permno` must be 2D, got shape {permno.shape}")
    if returns.ndim != 3:
        raise ValueError(f"`returns` must be 3D, got shape {returns.shape}")
    if volume.ndim != 3:
        raise ValueError(f"`volume` must be 3D, got shape {volume.shape}")
    if month_end.ndim != 1:
        raise ValueError(f"`month_end` must flatten to 1D, got shape {month_end.shape}")

    n_assets, lookback, n_time = returns.shape
    if volume.shape != returns.shape:
        raise ValueError(
            f"`volume` shape must match `returns` shape, got {volume.shape} vs {returns.shape}"
        )
    if permno.shape != (n_assets, n_time):
        raise ValueError(
            "`permno` shape must match (n_assets, n_time) from returns; "
            f"got {permno.shape} vs {(n_assets, n_time)}"
        )
    if month_end.shape[0] != n_time:
        raise ValueError(
            f"`month_end` length must match time axis {n_time}; got {month_end.shape[0]}"
        )
    if lookback <= 0:
        raise ValueError(f"lookback axis must be positive, got {lookback}")


def load_datacube(
    path: str | Path,
    *,
    sample_assets: int | None = None,
    sample_time: int | None = None,
) -> DataCube:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"DataCube file not found: {input_path}")

    with h5py.File(input_path, "r") as file:
        if "DataCube" not in file:
            raise KeyError("Expected top-level group '/DataCube' not found.")
        cube = file["DataCube"]
        required = {"lookback", "month_end", "permno", "returns", "volume"}
        missing = [name for name in required if name not in cube]
        if missing:
            raise KeyError(f"Missing required datasets under '/DataCube': {missing}")

        lookback_raw = np.asarray(cube["lookback"][()])
        month_end = np.asarray(cube["month_end"][()]).reshape(-1)
        permno = np.asarray(cube["permno"][()])
        returns = np.asarray(cube["returns"][()])
        volume = np.asarray(cube["volume"][()])

    lookback = int(lookback_raw.reshape(-1)[0])

    if sample_assets is not None:
        if sample_assets <= 0:
            raise ValueError(f"`sample_assets` must be positive, got {sample_assets}")
        n_assets = min(int(sample_assets), int(returns.shape[0]))
        permno = permno[:n_assets, :]
        returns = returns[:n_assets, :, :]
        volume = volume[:n_assets, :, :]

    if sample_time is not None:
        if sample_time <= 0:
            raise ValueError(f"`sample_time` must be positive, got {sample_time}")
        n_time = min(int(sample_time), int(returns.shape[2]))
        permno = permno[:, :n_time]
        returns = returns[:, :, :n_time]
        volume = volume[:, :, :n_time]
        month_end = month_end[:n_time]

    _validate_axis_alignment(permno, returns, volume, month_end)

    month_end_parsed = [_parse_month_end_value(float(x)) for x in month_end]
    return DataCube(
        lookback=lookback,
        month_end_raw=month_end.astype(np.float64, copy=False),
        month_end=month_end_parsed,
        permno=permno.astype(np.float64, copy=False),
        returns=returns.astype(np.float64, copy=False),
        volume=volume.astype(np.float64, copy=False),
    )

