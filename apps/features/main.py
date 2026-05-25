from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config import load_feature_config
from core.features import generate_features, generate_features_from_datacube
from core.io import load_datacube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate point-in-time technical features.")
    parser.add_argument(
        "--config",
        default="configs/features/default.yaml",
        help="Path to feature YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_feature_config(args.config)

    panel_input_path: Path | None = None
    if cfg.construction_mode == "direct_cube":
        cube = load_datacube(cfg.datacube_path)
        result = generate_features_from_datacube(cube, cfg)
    else:
        panel_input_path = Path(cfg.panel_input_dir)
        if panel_input_path.is_dir():
            panel_input_path = panel_input_path / "panel.parquet"
        panel = pd.read_parquet(panel_input_path)
        result = generate_features(panel, cfg)

    output_path = Path(cfg.feature_output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame = result.frame.copy()
    feature_frame.to_parquet(output_path, engine="pyarrow", index=False)

    meta = {
        "construction_mode": cfg.construction_mode,
        "datacube_path": str(cfg.datacube_path),
        "panel_input_path": str(panel_input_path) if panel_input_path is not None else None,
        "feature_output_path": str(output_path),
        "macro_factors_path": str(cfg.macro_factors_path) if cfg.macro_factors_path is not None else None,
        "macro_columns": cfg.macro_columns,
        "macro_lag_periods": cfg.macro_lag_periods,
        "rows_written": int(feature_frame.shape[0]),
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
    }
    meta_path = output_path.parent / "_metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print(f"Wrote feature parquet file: {output_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
