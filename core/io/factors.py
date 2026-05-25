from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from pandas.tseries.offsets import MonthEnd


KEN_FRENCH_FF5_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
KEN_FRENCH_UMD_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)

FACTOR_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "rf")
UMD_COLUMN = "umd"


def _download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:  # noqa: S310
        return response.read()


def _extract_zip_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError("Ken French archive did not contain a CSV file.")
        raw = archive.read(members[0])
    return raw.decode("latin-1")


def _parse_ff5_monthly_text(text: str) -> pd.DataFrame:
    header_idx: int | None = None
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        normalized = line.strip().lower().replace(" ", "")
        if normalized.startswith("mkt-rf,smb,hml,rmw,cma,rf"):
            header_idx = idx
            break
        if normalized.startswith(",mkt-rf,smb,hml,rmw,cma,rf"):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("Unable to locate FF5 monthly header in Ken French CSV.")

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for raw_line in lines[header_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if not parts:
            continue
        date_token = parts[0]
        if not (len(date_token) == 6 and date_token.isdigit()):
            # Monthly section ends before annual section/footer.
            break
        if len(parts) < 7:
            continue
        rows.append(
            (
                date_token,
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],
                parts[6],
            )
        )

    if not rows:
        raise ValueError("No monthly factor rows were parsed from Ken French CSV.")

    frame = pd.DataFrame(
        rows,
        columns=["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    )
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce") / 100.0
    frame["date"] = pd.to_datetime(frame["yyyymm"], format="%Y%m", errors="coerce") + MonthEnd(0)
    frame = frame.drop(columns=["yyyymm"])
    frame = frame.dropna(subset=["date"])
    frame = frame[["date", *FACTOR_COLUMNS]].dropna(subset=["rf"])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return frame


def load_factor_panel(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Factor cache not found: {path}")
    frame = pd.read_parquet(path)
    required = {"date", *FACTOR_COLUMNS}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Factor cache is missing required columns: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in FACTOR_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=list(FACTOR_COLUMNS)).reset_index(drop=True)
    return frame


def fetch_ken_french_monthly(cache_path: str | Path, refresh: bool = False) -> pd.DataFrame:
    path = Path(cache_path)
    if path.exists() and not refresh:
        return load_factor_panel(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _download_bytes(KEN_FRENCH_FF5_MONTHLY_URL)
    text = _extract_zip_text(payload)
    frame = _parse_ff5_monthly_text(text)
    frame.to_parquet(path, index=False)
    return frame


def _parse_umd_monthly_text(text: str) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    started = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if not parts:
            continue
        date_token = parts[0]
        if len(date_token) == 6 and date_token.isdigit():
            if len(parts) >= 2:
                rows.append((date_token, parts[1]))
                started = True
            continue
        if started:
            # Monthly section ended before annual/footer section.
            break

    if not rows:
        raise ValueError("No monthly UMD rows were parsed from Ken French CSV.")

    frame = pd.DataFrame(rows, columns=["yyyymm", UMD_COLUMN])
    frame[UMD_COLUMN] = pd.to_numeric(frame[UMD_COLUMN], errors="coerce") / 100.0
    frame["date"] = pd.to_datetime(frame["yyyymm"], format="%Y%m", errors="coerce") + MonthEnd(0)
    frame = frame.drop(columns=["yyyymm"])
    frame = frame.dropna(subset=["date", UMD_COLUMN])
    frame = frame[["date", UMD_COLUMN]].sort_values("date").drop_duplicates("date", keep="last").reset_index(
        drop=True
    )
    return frame


def load_umd_factor_panel(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"UMD factor cache not found: {path}")
    frame = pd.read_parquet(path)
    required = {"date", UMD_COLUMN}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"UMD factor cache is missing required columns: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[UMD_COLUMN] = pd.to_numeric(frame[UMD_COLUMN], errors="coerce")
    frame = frame.dropna(subset=["date", UMD_COLUMN]).sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def fetch_ken_french_momentum_monthly(cache_path: str | Path, refresh: bool = False) -> pd.DataFrame:
    path = Path(cache_path)
    if path.exists() and not refresh:
        return load_umd_factor_panel(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _download_bytes(KEN_FRENCH_UMD_MONTHLY_URL)
    text = _extract_zip_text(payload)
    frame = _parse_umd_monthly_text(text)
    frame.to_parquet(path, index=False)
    return frame
