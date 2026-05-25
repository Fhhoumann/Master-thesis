from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config.schemas import FeatureConfig

RAW_TO_STANDARD = {
    "date": "date",
    "vix": "macro_vix",
    "epu": "macro_epu",
    "credit_spread": "macro_credit_spread",
    "term_spread": "macro_term_spread",
}


def _read_macro_frame(path: Path, sheet_name: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported macro factors file type: {path.suffix}")


def load_macro_factors(cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    if cfg.macro_factors_path is None:
        return pd.DataFrame(), []

    path = Path(cfg.macro_factors_path)
    if not path.exists():
        raise FileNotFoundError(f"Macro factors file not found: {path}")

    frame = _read_macro_frame(path, cfg.macro_sheet_name).copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    normalized = {col: RAW_TO_STANDARD.get(col.strip().lower(), col.strip().lower()) for col in frame.columns}
    frame = frame.rename(columns=normalized)

    required = {"date", *cfg.macro_columns}
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Macro factors file is missing required columns: {missing}")

    macro = frame.loc[:, ["date", *cfg.macro_columns]].copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro = macro.dropna(subset=["date"]).sort_values("date", kind="stable")
    macro["month_key"] = macro["date"].dt.to_period("M")
    macro = macro.drop_duplicates(subset=["month_key"], keep="last")

    for column in cfg.macro_columns:
        macro[column] = pd.to_numeric(macro[column], errors="coerce")

    if cfg.macro_lag_periods:
        macro["month_key"] = macro["month_key"] + int(cfg.macro_lag_periods)

    return macro[["month_key", *cfg.macro_columns]].reset_index(drop=True), list(cfg.macro_columns)


def merge_macro_features(frame: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    macro, macro_columns = load_macro_factors(cfg)
    if not macro_columns:
        return frame, []

    out = frame.copy()
    out["month_key"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M")
    out = out.merge(macro, on="month_key", how="left", validate="many_to_one")
    out = out.drop(columns=["month_key"])
    return out, macro_columns
