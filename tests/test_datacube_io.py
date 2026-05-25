from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from core.io.datacube import load_datacube


def _write_sample_datacube(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        cube = handle.create_group("DataCube")
        cube.create_dataset("lookback", data=np.array([[3]], dtype=np.int32))
        cube.create_dataset("month_end", data=np.array([[20200131.0, 20200229.0]], dtype=np.float64))
        cube.create_dataset("permno", data=np.array([[10001.0, 10001.0]], dtype=np.float64))
        cube.create_dataset(
            "returns",
            data=np.array([[[0.01, 0.02], [0.02, 0.01], [0.03, -0.01]]], dtype=np.float64),
        )
        cube.create_dataset(
            "volume",
            data=np.array([[[100.0, 110.0], [120.0, 130.0], [140.0, 150.0]]], dtype=np.float64),
        )


def test_load_datacube_parses_dates_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    _write_sample_datacube(path)

    cube = load_datacube(path)
    assert cube.shape == (1, 3, 2)
    assert cube.lookback == 3
    assert cube.month_end[0] is not None
    assert cube.month_end[0].strftime("%Y-%m-%d") == "2020-01-31"
    assert cube.current_returns.shape == (1, 2)


def test_load_datacube_sampling(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    _write_sample_datacube(path)

    cube = load_datacube(path, sample_assets=1, sample_time=1)
    assert cube.shape == (1, 3, 1)
    assert cube.permno.shape == (1, 1)
    assert len(cube.month_end) == 1

