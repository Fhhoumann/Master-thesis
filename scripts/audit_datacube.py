#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_PATHS = {
    "lookback": (1, 1),
    "month_end": (1, None),
    "permno": (None, None),
    "returns": (None, None, None),
    "volume": (None, None, None),
}


def parse_month_end_to_datetime(value: float) -> datetime | None:
    if not np.isfinite(value):
        return None

    rounded = int(round(float(value)))
    raw = str(abs(rounded))
    if len(raw) == 8:
        try:
            return datetime.strptime(str(rounded), "%Y%m%d")
        except ValueError:
            pass

    try:
        ordinal = int(value)
        return datetime.fromordinal(ordinal) + timedelta(days=float(value) % 1) - timedelta(
            days=366
        )
    except (OverflowError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: as_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_native(v) for v in value]
    if isinstance(value, tuple):
        return [as_native(v) for v in value]
    return value


def format_number(value: float | None, decimals: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def profile_array(array: np.ndarray) -> dict[str, Any]:
    total_count = int(array.size)
    finite_mask = np.isfinite(array)
    finite_count = int(finite_mask.sum())
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    zero_count = int((array == 0).sum())

    profile: dict[str, Any] = {
        "shape": list(array.shape),
        "total_count": total_count,
        "finite_count": finite_count,
        "finite_rate": finite_count / total_count if total_count else None,
        "nan_count": nan_count,
        "nan_rate": nan_count / total_count if total_count else None,
        "inf_count": inf_count,
        "inf_rate": inf_count / total_count if total_count else None,
        "zero_count": zero_count,
        "zero_rate": zero_count / total_count if total_count else None,
    }

    if finite_count == 0:
        profile["stats"] = {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
        return profile

    finite_values = array[finite_mask]
    profile["stats"] = {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "p01": float(np.quantile(finite_values, 0.01)),
        "p05": float(np.quantile(finite_values, 0.05)),
        "p50": float(np.quantile(finite_values, 0.50)),
        "p95": float(np.quantile(finite_values, 0.95)),
        "p99": float(np.quantile(finite_values, 0.99)),
    }
    return profile


def top_extremes_by_abs(array: np.ndarray, top_n: int = 10) -> list[dict[str, Any]]:
    flat = array.reshape(-1)
    if flat.size == 0:
        return []

    scores = np.where(np.isfinite(flat), np.abs(flat), -np.inf)
    finite_total = int(np.isfinite(scores).sum())
    if finite_total == 0:
        return []

    k = min(top_n, finite_total)
    part = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[part])[::-1]
    indices = part[order]

    results: list[dict[str, Any]] = []
    for idx in indices:
        if not np.isfinite(scores[idx]):
            continue
        multi_index = np.unravel_index(int(idx), array.shape)
        results.append(
            {
                "index": [int(x) for x in multi_index],
                "value": float(flat[idx]),
                "abs_value": float(abs(flat[idx])),
            }
        )
    return results


def top_indices(values: np.ndarray, top_n: int = 10) -> list[dict[str, Any]]:
    if values.size == 0:
        return []
    k = min(top_n, int(values.size))
    part = np.argpartition(values, -k)[-k:]
    order = np.argsort(values[part])[::-1]
    return [{"index": int(i), "value": float(values[i])} for i in part[order]]


def shape_is_compatible(shape: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    if len(shape) != len(expected):
        return False
    for got, want in zip(shape, expected):
        if want is not None and got != want:
            return False
    return True


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_markdown_report(profile: dict[str, Any]) -> str:
    generated_at = profile["meta"]["generated_at_utc"]
    input_file = profile["meta"]["input_file"]
    file_size_mb = profile["meta"]["file_size_mb"]
    sample = profile["meta"]["sample"]
    shapes = profile["meta"]["shapes"]

    lines: list[str] = []
    lines.append("# DataCube Audit")
    lines.append("")
    lines.append(f"- Generated (UTC): `{generated_at}`")
    lines.append(f"- Input file: `{input_file}`")
    lines.append(f"- File size (MB): `{file_size_mb}`")
    lines.append(f"- Sample mode: `{sample}`")
    lines.append("")

    lines.append("## Dataset Shapes")
    lines.append("")
    lines.append("| Dataset | Shape |")
    lines.append("|---|---|")
    for key, shape in shapes.items():
        lines.append(f"| `{key}` | `{tuple(shape)}` |")
    lines.append("")

    lines.append("## Consistency Checks")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for check in profile["consistency_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {check['name']} | {status} | {check['detail']} |")
    lines.append("")

    lines.append("## Data Quality")
    lines.append("")
    lines.append("| Dataset | Total | Finite % | NaN % | Inf % | Zero % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset_name, quality in profile["quality"].items():
        lines.append(
            "| {dataset} | {total} | {finite:.2f} | {nan:.2f} | {inf:.2f} | {zero:.2f} |".format(
                dataset=dataset_name,
                total=quality["total_count"],
                finite=100.0 * (quality["finite_rate"] or 0.0),
                nan=100.0 * (quality["nan_rate"] or 0.0),
                inf=100.0 * (quality["inf_rate"] or 0.0),
                zero=100.0 * (quality["zero_rate"] or 0.0),
            )
        )
    lines.append("")

    lines.append("## Returns and Volume Distribution")
    lines.append("")
    lines.append("| Dataset | Min | P01 | P05 | Median | P95 | P99 | Max | Mean | Std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset_name in ("returns", "volume"):
        stats = profile["distributions"][dataset_name]
        lines.append(
            "| {dataset} | {minv} | {p01} | {p05} | {p50} | {p95} | {p99} | {maxv} | {mean} | {std} |".format(
                dataset=dataset_name,
                minv=format_number(to_float(stats["min"])),
                p01=format_number(to_float(stats["p01"])),
                p05=format_number(to_float(stats["p05"])),
                p50=format_number(to_float(stats["p50"])),
                p95=format_number(to_float(stats["p95"])),
                p99=format_number(to_float(stats["p99"])),
                maxv=format_number(to_float(stats["max"])),
                mean=format_number(to_float(stats["mean"])),
                std=format_number(to_float(stats["std"])),
            )
        )
    lines.append("")

    coverage = profile["time_coverage"]
    lines.append("## Time Coverage")
    lines.append("")
    lines.append(f"- First month_end: `{coverage['first_date']}`")
    lines.append(f"- Last month_end: `{coverage['last_date']}`")
    lines.append(f"- Count of month_end entries: `{coverage['count']}`")
    lines.append(f"- Monotonic non-decreasing: `{coverage['monotonic_non_decreasing']}`")
    lines.append("")

    anomalies = profile["anomalies"]
    lines.append("## Key Anomalies")
    lines.append("")
    lines.append(f"- Returns |z| > 6 count: `{anomalies['returns_outlier_count_z6']}`")
    lines.append(
        "- Returns abs(value) > 5 count: `{}`".format(anomalies["returns_abs_gt_5_count"])
    )
    lines.append(f"- Negative volume count: `{anomalies['volume_negative_count']}`")
    lines.append(
        "- Zero-volume (all-lookback zeros) slices: `{}`".format(
            anomalies["volume_zero_lookback_slice_count"]
        )
    )
    lines.append(
        "- Near-constant returns slices (std <= 1e-12): `{}`".format(
            anomalies["returns_constant_slice_count"]
        )
    )
    lines.append("")

    lines.append("## Top Missingness by Time")
    lines.append("")
    lines.append("| Dataset | Time Index | Month End | Missing % |")
    lines.append("|---|---:|---|---:|")
    for dataset_name in ("returns", "volume"):
        for item in profile["missingness"]["top_time_missingness"][dataset_name]:
            lines.append(
                "| {dataset} | {idx} | {month_end} | {pct:.2f} |".format(
                    dataset=dataset_name,
                    idx=item["index"],
                    month_end=item.get("month_end", "n/a"),
                    pct=100.0 * item["value"],
                )
            )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DataCube MAT v7.3 contents.")
    parser.add_argument(
        "--input",
        default="DataCube_1990_2024.mat",
        help="Path to MAT v7.3 file.",
    )
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Directory for markdown/json outputs.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional max size for both asset and time dimensions.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    profile: dict[str, Any] = {
        "meta": {
            "input_file": str(input_path.resolve()),
            "file_size_bytes": int(os.path.getsize(input_path)),
            "file_size_mb": round(os.path.getsize(input_path) / (1024 * 1024), 2),
            "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sample": args.sample if args.sample is not None else "full",
            "shapes": {},
        },
        "consistency_checks": [],
        "quality": {},
        "distributions": {},
        "missingness": {
            "per_time_rate": {},
            "per_asset_rate": {},
            "per_lookback_rate": {},
            "top_time_missingness": {},
            "top_asset_missingness": {},
        },
        "time_coverage": {},
        "anomalies": {},
    }

    with h5py.File(input_path, "r") as file:
        if "DataCube" not in file:
            raise KeyError("Expected top-level group '/DataCube' not found.")
        cube = file["DataCube"]

        for name, expected_shape in EXPECTED_PATHS.items():
            present = name in cube
            if not present:
                profile["consistency_checks"].append(
                    {"name": f"dataset_present:{name}", "passed": False, "detail": "missing"}
                )
                continue
            dataset = cube[name]
            shape = tuple(int(x) for x in dataset.shape)
            profile["meta"]["shapes"][name] = list(shape)
            profile["consistency_checks"].append(
                {
                    "name": f"shape_compatible:{name}",
                    "passed": shape_is_compatible(shape, expected_shape),
                    "detail": f"shape={shape}, expected={expected_shape}",
                }
            )

        required = {"lookback", "month_end", "permno", "returns", "volume"}
        missing_required = [k for k in required if k not in cube]
        if missing_required:
            raise KeyError(f"Missing required datasets under /DataCube: {missing_required}")

        lookback = cube["lookback"][()]
        month_end = cube["month_end"][()]
        permno = cube["permno"][()]
        returns = cube["returns"][()]
        volume = cube["volume"][()]

        if args.sample is not None and args.sample > 0:
            n_assets = min(args.sample, returns.shape[0])
            n_time = min(args.sample, returns.shape[2])
            permno = permno[:n_assets, :n_time]
            returns = returns[:n_assets, :, :n_time]
            volume = volume[:n_assets, :, :n_time]
            month_end = month_end[:, :n_time]
            profile["consistency_checks"].append(
                {
                    "name": "sample_applied",
                    "passed": True,
                    "detail": f"assets={n_assets}, time={n_time}",
                }
            )
        elif args.sample is not None and args.sample <= 0:
            profile["consistency_checks"].append(
                {
                    "name": "sample_applied",
                    "passed": False,
                    "detail": f"invalid --sample value: {args.sample}",
                }
            )

        dims_ok = (
            permno.shape[0] == returns.shape[0] == volume.shape[0]
            and permno.shape[1] == returns.shape[2] == volume.shape[2] == month_end.shape[1]
            and returns.shape[1] == volume.shape[1]
        )
        profile["consistency_checks"].append(
            {
                "name": "axis_alignment",
                "passed": bool(dims_ok),
                "detail": "permno[N,T], returns[N,L,T], volume[N,L,T], month_end[1,T]",
            }
        )

        for dataset_name, array in (
            ("lookback", np.asarray(lookback)),
            ("month_end", np.asarray(month_end)),
            ("permno", np.asarray(permno)),
            ("returns", np.asarray(returns)),
            ("volume", np.asarray(volume)),
        ):
            summary = profile_array(array)
            profile["quality"][dataset_name] = {
                k: v for k, v in summary.items() if k != "stats"
            }
            profile["distributions"][dataset_name] = summary["stats"]

        month_end_values = np.asarray(month_end).reshape(-1)
        finite_month_end = month_end_values[np.isfinite(month_end_values)]
        month_datetimes = [parse_month_end_to_datetime(float(x)) for x in month_end_values]
        month_dates = [d.strftime("%Y-%m-%d") if d is not None else None for d in month_datetimes]
        if finite_month_end.size:
            finite_datetimes = [d for d in month_datetimes if d is not None]
            if finite_datetimes:
                date_ordinals = np.array([d.toordinal() for d in finite_datetimes], dtype=np.int64)
                day_diffs = np.diff(date_ordinals)
                monotonic = bool(np.all(day_diffs >= 0)) if day_diffs.size else True
                median_step_days = float(np.median(day_diffs)) if day_diffs.size else 0.0
                first_date = finite_datetimes[0].strftime("%Y-%m-%d")
                last_date = finite_datetimes[-1].strftime("%Y-%m-%d")
            else:
                monotonic = False
                median_step_days = None
                first_date = None
                last_date = None
            profile["time_coverage"] = {
                "count": int(month_end_values.size),
                "first_date": first_date,
                "last_date": last_date,
                "monotonic_non_decreasing": monotonic,
                "median_step_days": median_step_days,
            }
        else:
            profile["time_coverage"] = {
                "count": int(month_end_values.size),
                "first_date": None,
                "last_date": None,
                "monotonic_non_decreasing": False,
                "median_step_days": None,
            }

        returns_finite = np.isfinite(returns)
        volume_finite = np.isfinite(volume)

        returns_missing = (~returns_finite).astype(np.float64)
        volume_missing = (~volume_finite).astype(np.float64)

        returns_time_missing = returns_missing.mean(axis=(0, 1))
        returns_asset_missing = returns_missing.mean(axis=(1, 2))
        returns_lookback_missing = returns_missing.mean(axis=(0, 2))

        volume_time_missing = volume_missing.mean(axis=(0, 1))
        volume_asset_missing = volume_missing.mean(axis=(1, 2))
        volume_lookback_missing = volume_missing.mean(axis=(0, 2))

        profile["missingness"]["per_time_rate"]["returns"] = returns_time_missing.tolist()
        profile["missingness"]["per_time_rate"]["volume"] = volume_time_missing.tolist()
        profile["missingness"]["per_asset_rate"]["returns"] = returns_asset_missing.tolist()
        profile["missingness"]["per_asset_rate"]["volume"] = volume_asset_missing.tolist()
        profile["missingness"]["per_lookback_rate"]["returns"] = returns_lookback_missing.tolist()
        profile["missingness"]["per_lookback_rate"]["volume"] = volume_lookback_missing.tolist()

        top_returns_time = top_indices(returns_time_missing, top_n=10)
        top_volume_time = top_indices(volume_time_missing, top_n=10)
        for item in top_returns_time:
            idx = item["index"]
            item["month_end"] = month_dates[idx] if idx < len(month_dates) else None
        for item in top_volume_time:
            idx = item["index"]
            item["month_end"] = month_dates[idx] if idx < len(month_dates) else None

        profile["missingness"]["top_time_missingness"]["returns"] = top_returns_time
        profile["missingness"]["top_time_missingness"]["volume"] = top_volume_time
        profile["missingness"]["top_asset_missingness"]["returns"] = top_indices(
            returns_asset_missing, top_n=10
        )
        profile["missingness"]["top_asset_missingness"]["volume"] = top_indices(
            volume_asset_missing, top_n=10
        )

        returns_stats = profile["distributions"]["returns"]
        returns_mean = returns_stats["mean"]
        returns_std = returns_stats["std"]
        if returns_mean is not None and returns_std is not None and returns_std > 0:
            z_abs = np.abs((returns - returns_mean) / returns_std)
            outlier_mask = np.isfinite(z_abs) & (z_abs > 6.0)
            returns_outlier_count = int(outlier_mask.sum())
        else:
            returns_outlier_count = 0

        returns_abs_gt_5 = int((np.isfinite(returns) & (np.abs(returns) > 5.0)).sum())
        volume_negative = int((np.isfinite(volume) & (volume < 0)).sum())

        volume_zero_lookback_mask = np.all(volume == 0, axis=1)
        volume_zero_lookback_count = int(volume_zero_lookback_mask.sum())
        zero_pairs = np.argwhere(volume_zero_lookback_mask)
        zero_examples: list[dict[str, Any]] = []
        for pair in zero_pairs[:10]:
            asset_idx = int(pair[0])
            time_idx = int(pair[1])
            zero_examples.append(
                {
                    "asset_index": asset_idx,
                    "time_index": time_idx,
                    "month_end": month_dates[time_idx] if time_idx < len(month_dates) else None,
                }
            )

        returns_std_by_slice = np.nanstd(returns, axis=1)
        returns_finite_by_slice = np.isfinite(returns).sum(axis=1)
        returns_constant_mask = (returns_finite_by_slice > 1) & (returns_std_by_slice <= 1e-12)
        returns_constant_count = int(returns_constant_mask.sum())
        constant_pairs = np.argwhere(returns_constant_mask)
        constant_examples: list[dict[str, Any]] = []
        for pair in constant_pairs[:10]:
            asset_idx = int(pair[0])
            time_idx = int(pair[1])
            constant_examples.append(
                {
                    "asset_index": asset_idx,
                    "time_index": time_idx,
                    "month_end": month_dates[time_idx] if time_idx < len(month_dates) else None,
                    "std": float(returns_std_by_slice[asset_idx, time_idx]),
                }
            )

        top_extreme_returns = top_extremes_by_abs(returns, top_n=10)
        top_extreme_volume = top_extremes_by_abs(volume, top_n=10)
        for row in top_extreme_returns:
            time_idx = row["index"][2]
            row["month_end"] = month_dates[time_idx] if time_idx < len(month_dates) else None
        for row in top_extreme_volume:
            time_idx = row["index"][2]
            row["month_end"] = month_dates[time_idx] if time_idx < len(month_dates) else None

        permno_finite = permno[np.isfinite(permno)]
        unique_permno = int(np.unique(permno_finite).size) if permno_finite.size else 0
        row_changes = 0
        row_total = 0
        for row in permno:
            finite_row = row[np.isfinite(row)]
            if finite_row.size >= 2:
                row_total += 1
                row_changes += int(np.any(np.diff(finite_row) != 0))
        permno_change_rate = row_changes / row_total if row_total else None

        profile["anomalies"] = {
            "returns_outlier_count_z6": returns_outlier_count,
            "returns_abs_gt_5_count": returns_abs_gt_5,
            "returns_constant_slice_count": returns_constant_count,
            "returns_constant_slice_examples": constant_examples,
            "volume_negative_count": volume_negative,
            "volume_zero_lookback_slice_count": volume_zero_lookback_count,
            "volume_zero_lookback_slice_examples": zero_examples,
            "top_extreme_returns_abs": top_extreme_returns,
            "top_extreme_volume_abs": top_extreme_volume,
            "permno_unique_count": unique_permno,
            "permno_rows_with_changes": row_changes,
            "permno_change_rate": permno_change_rate,
        }

    table_files: list[str] = []

    top_time_rows: list[dict[str, Any]] = []
    for dataset_name in ("returns", "volume"):
        for row in profile["missingness"]["top_time_missingness"][dataset_name]:
            top_time_rows.append(
                {
                    "dataset": dataset_name,
                    "time_index": row["index"],
                    "month_end": row.get("month_end"),
                    "missing_rate": row["value"],
                }
            )
    top_time_path = tables_dir / "top_time_missingness.csv"
    write_csv(
        top_time_path,
        ["dataset", "time_index", "month_end", "missing_rate"],
        top_time_rows,
    )
    table_files.append(str(top_time_path))

    top_asset_rows: list[dict[str, Any]] = []
    for dataset_name in ("returns", "volume"):
        for row in profile["missingness"]["top_asset_missingness"][dataset_name]:
            top_asset_rows.append(
                {
                    "dataset": dataset_name,
                    "asset_index": row["index"],
                    "missing_rate": row["value"],
                }
            )
    top_asset_path = tables_dir / "top_asset_missingness.csv"
    write_csv(
        top_asset_path,
        ["dataset", "asset_index", "missing_rate"],
        top_asset_rows,
    )
    table_files.append(str(top_asset_path))

    extreme_returns_rows: list[dict[str, Any]] = []
    for row in profile["anomalies"]["top_extreme_returns_abs"]:
        extreme_returns_rows.append(
            {
                "asset_index": row["index"][0],
                "lookback_index": row["index"][1],
                "time_index": row["index"][2],
                "month_end": row.get("month_end"),
                "value": row["value"],
                "abs_value": row["abs_value"],
            }
        )
    extreme_returns_path = tables_dir / "top_extreme_returns_abs.csv"
    write_csv(
        extreme_returns_path,
        ["asset_index", "lookback_index", "time_index", "month_end", "value", "abs_value"],
        extreme_returns_rows,
    )
    table_files.append(str(extreme_returns_path))

    extreme_volume_rows: list[dict[str, Any]] = []
    for row in profile["anomalies"]["top_extreme_volume_abs"]:
        extreme_volume_rows.append(
            {
                "asset_index": row["index"][0],
                "lookback_index": row["index"][1],
                "time_index": row["index"][2],
                "month_end": row.get("month_end"),
                "value": row["value"],
                "abs_value": row["abs_value"],
            }
        )
    extreme_volume_path = tables_dir / "top_extreme_volume_abs.csv"
    write_csv(
        extreme_volume_path,
        ["asset_index", "lookback_index", "time_index", "month_end", "value", "abs_value"],
        extreme_volume_rows,
    )
    table_files.append(str(extreme_volume_path))

    profile["meta"]["table_files"] = table_files

    json_path = outdir / "datacube_profile.json"
    md_path = outdir / "datacube_audit.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(as_native(profile), handle, indent=2)
        handle.write("\n")

    markdown = build_markdown_report(profile)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
