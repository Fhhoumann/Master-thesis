from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.io.datacube import DataCube


def _asset_id_array(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    out[finite] = np.round(values[finite])
    return out


def build_panel_table(datacube: DataCube) -> pd.DataFrame:
    current_returns = datacube.current_returns
    current_volume = datacube.current_volume
    n_assets, n_time = current_returns.shape

    row_index = np.repeat(np.arange(n_assets, dtype=np.int32), n_time)
    time_index = np.tile(np.arange(n_time, dtype=np.int32), n_assets)
    date_values = pd.to_datetime(
        np.tile(np.array(datacube.month_end, dtype="datetime64[ns]"), n_assets), errors="coerce"
    )
    ret_values = current_returns.reshape(-1)
    vol_values = current_volume.reshape(-1)
    permno_values = _asset_id_array(datacube.permno.reshape(-1))

    panel = pd.DataFrame(
        {
            "date": date_values,
            "time_index": time_index,
            "row_id": row_index,
            "asset_id": permno_values,
            "ret_1m": ret_values,
            "volume_1m": vol_values,
        }
    )
    panel["has_asset_id"] = panel["asset_id"].notna()
    panel["has_finite_return"] = np.isfinite(panel["ret_1m"])
    return panel


def write_partitioned_panel_parquet(
    panel: pd.DataFrame,
    output_dir: str | Path,
    *,
    partition_cols: tuple[str, ...] = ("year",),
) -> Path:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel_out = panel.copy()
    panel_out["year"] = panel_out["date"].dt.year.fillna(-1).astype(np.int32)
    panel_out["month"] = panel_out["date"].dt.month.fillna(-1).astype(np.int8)
    output_path = outdir / "panel.parquet"
    panel_out.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path
