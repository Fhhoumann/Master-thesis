from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import load_dataset_config
from core.io import build_panel_table, load_datacube, write_partitioned_panel_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical panel parquet from DataCube MAT.")
    parser.add_argument(
        "--config",
        default="configs/datasets/default.yaml",
        help="Path to dataset YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)

    datacube = load_datacube(
        cfg.datacube_path,
        sample_assets=cfg.sample_assets,
        sample_time=cfg.sample_time,
    )
    panel = build_panel_table(datacube)
    output_path = write_partitioned_panel_parquet(panel, cfg.panel_output_dir)

    metadata = {
        "datacube_path": str(cfg.datacube_path),
        "panel_output_path": str(output_path),
        "shape": {
            "n_assets": datacube.n_assets,
            "n_lookback": datacube.n_lookback,
            "n_time": datacube.n_time,
        },
        "rows_written": int(panel.shape[0]),
    }
    meta_path = output_path.parent / "_metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Wrote panel parquet file: {output_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
