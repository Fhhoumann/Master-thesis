from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from core.benchmark.attribution import compute_attribution_table
from core.config import load_experiment_config
from core.io.factors import fetch_ken_french_monthly, load_factor_panel
from core.metrics import summarize_backtest_frame
from core.portfolio import run_backtest_with_costs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "experiments"
ARTIFACTS = ROOT / "artifacts"
REPORT_TABLES = ROOT / "reports" / "eda" / "tables"
REPORT_FIGS = ROOT / "reports" / "eda" / "figures"


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ensure_dirs() -> None:
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)


@dataclass
class RunSpec:
    key: str
    pipeline: str
    cfg: str
    artifact_run_name: str


MAIN_SPECS = [
    RunSpec(
        key="EN_twostep",
        pipeline="twostep",
        cfg="runpack_twostep_elasticnet_rolling120_step3_nocost.yaml",
        artifact_run_name="runpack_twostep_elasticnet_rolling120_step3_nocost",
    ),
    RunSpec(
        key="RF_twostep",
        pipeline="twostep",
        cfg="runpack_twostep_random_forest_rolling120_step3_nocost.yaml",
        artifact_run_name="runpack_twostep_random_forest_rolling120_step3_nocost",
    ),
    RunSpec(
        key="Torch_onestep",
        pipeline="onestep",
        cfg="runpack_onestep_torch_rolling120_step3_nocost.yaml",
        artifact_run_name="runpack_onestep_torch_rolling120_step3_nocost",
    ),
]


def run_main_models() -> None:
    for spec in MAIN_SPECS:
        cfg_path = CONFIGS / spec.cfg
        if spec.pipeline == "twostep":
            _run([
                "uv",
                "run",
                "--with",
                "pandas",
                "--with",
                "pyarrow",
                "--with",
                "pyyaml",
                "--with",
                "numpy",
                "--with",
                "scikit-learn",
                "--with",
                "lightgbm",
                "python",
                "apps/train_twostep/main.py",
                "--config",
                str(cfg_path),
            ])
        else:
            _run([
                "uv",
                "run",
                "--with",
                "pandas",
                "--with",
                "pyarrow",
                "--with",
                "pyyaml",
                "--with",
                "numpy",
                "--with",
                "torch",
                "python",
                "apps/train_onestep/main.py",
                "--config",
                str(cfg_path),
            ])


def _load_nocost_backtest(spec: RunSpec) -> pd.DataFrame:
    path = ARTIFACTS / spec.artifact_run_name / spec.pipeline / "backtest.parquet"
    bt = pd.read_parquet(path)
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
    bt = bt[(bt["cost_convention"] == "one_way") & (bt["cost_bps"] == 0.0)].copy()
    bt["run_key"] = spec.key
    return bt


def _load_rf_monthly() -> pd.DataFrame:
    cache = ARTIFACTS / "factors" / "ken_french_monthly.parquet"
    if cache.exists():
        return load_factor_panel(cache)
    return fetch_ken_french_monthly(cache, refresh=False)


def build_main_outputs() -> None:
    backtests = [_load_nocost_backtest(spec) for spec in MAIN_SPECS]
    merged_bt = pd.concat(backtests, ignore_index=True)

    # Summary metrics
    rows = []
    for spec in MAIN_SPECS:
        s = summarize_backtest_frame(
            merged_bt.loc[merged_bt["run_key"] == spec.key, [
                "date", "cost_convention", "cost_bps", "net_return", "turnover_one_way", "turnover_round_trip"
            ]],
            risk_free_frame=_load_rf_monthly(),
        )
        row = s.iloc[0].to_dict()
        row["run_key"] = spec.key
        rows.append(row)
    summary = pd.DataFrame(rows)

    # CAPM attribution with EW market proxy from feature panel
    features = pd.read_parquet(ARTIFACTS / "features" / "features.parquet")
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    ew = (
        features.dropna(subset=["date", "ret_1m"]).groupby("date", observed=True)["ret_1m"].mean().reset_index()
    )
    ew = ew.rename(columns={"ret_1m": "ew_mkt"})

    rf = _load_rf_monthly()[["date", "rf"]].copy()
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
    factors = ew.merge(rf, on="date", how="inner")
    factors["mkt_rf"] = factors["ew_mkt"] - factors["rf"]

    returns_rows = []
    for spec in MAIN_SPECS:
        bt = merged_bt[merged_bt["run_key"] == spec.key]
        for _, r in bt.iterrows():
            returns_rows.append(
                {
                    "date": r["date"],
                    "run_key": spec.key,
                    "return_type": "gross",
                    "cost_convention": "none",
                    "cost_bps": 0.0,
                    "period_return": float(r["net_return"]),
                }
            )
    returns_frame = pd.DataFrame(returns_rows)
    regimes = []
    capm = compute_attribution_table(
        returns_frame=returns_frame,
        factors=factors[["date", "rf", "mkt_rf"]],
        regimes=regimes,
        models=("capm",),
        min_obs=24,
    )
    capm = capm[(capm["regime"] == "full_sample") & (capm["model"] == "capm")].copy()
    capm = capm[["run_key", "alpha_annualized", "alpha_tstat", "beta_mkt_rf", "r2"]]

    main_table = summary.merge(capm, on="run_key", how="left")
    main_table = main_table[[
        "run_key",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "alpha_annualized",
        "alpha_tstat",
        "beta_mkt_rf",
        "r2",
    ]]
    main_table.to_csv(REPORT_TABLES / "final_120_nocost_main_table.csv", index=False)

    # cumulative plot
    plt.figure(figsize=(12, 6))
    for spec in MAIN_SPECS:
        bt = merged_bt[merged_bt["run_key"] == spec.key].sort_values("date")
        cum = (1.0 + bt["net_return"]).cumprod() - 1.0
        plt.plot(bt["date"], cum, label=spec.key)

    bench = ew.sort_values("date")
    cum_bench = (1.0 + bench["ew_mkt"]).cumprod() - 1.0
    plt.plot(bench["date"], cum_bench, label="EW_benchmark", linestyle="--")

    plt.title("Cumulative Returns (120, No-Cost)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = REPORT_FIGS / "final_120_nocost_cumulative_returns.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # persist capm table
    capm.to_csv(REPORT_TABLES / "final_120_nocost_capm_table.csv", index=False)


def build_cost_outputs() -> None:
    rows = []
    for spec in MAIN_SPECS:
        cfg = load_experiment_config(CONFIGS / spec.cfg)
        run_dir = ARTIFACTS / spec.artifact_run_name / spec.pipeline

        if spec.pipeline == "twostep":
            preds = pd.read_parquet(run_dir / "predictions.parquet")
            weights = pd.read_parquet(run_dir / "weights.parquet")
            realized = preds.rename(columns={cfg.twostep.target_column: "ret_1m"})[["date", "asset_id", "ret_1m"]]
        else:
            weights = pd.read_parquet(run_dir / "weights.parquet")
            realized = weights.rename(columns={"target_ret_1m": "ret_1m"})[["date", "asset_id", "ret_1m"]]

        bt = run_backtest_with_costs(
            weights=weights[["date", "asset_id", "weight"]],
            realized_returns=realized,
            cost_bps=[10.0, 25.0, 50.0],
            conventions=["one_way"],
            initial_turnover_from_zero=cfg.backtest.initial_turnover_from_zero,
            turnover_use_effective_weights=cfg.backtest.turnover_use_effective_weights,
            missing_return_policy=cfg.backtest.missing_return_policy,
        )
        bt["run_key"] = spec.key
        bt.to_parquet(run_dir / "backtest_costs.parquet", index=False)

        summary = summarize_backtest_frame(bt, risk_free_frame=_load_rf_monthly())
        summary["run_key"] = spec.key
        rows.append(summary)

    out = pd.concat(rows, ignore_index=True)
    out = out[[
        "run_key",
        "cost_convention",
        "cost_bps",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
    ]]
    out.to_csv(REPORT_TABLES / "final_120_costs_main_table.csv", index=False)


def _write_variant_config(base_cfg: Path, out_cfg: Path, variant: str) -> None:
    cfg = yaml.safe_load(base_cfg.read_text())
    cols = cfg["twostep"]["feature_columns"] if "twostep" in cfg else cfg["onestep"]["feature_columns"]

    if variant == "both":
        pass
    elif variant == "only_tsmom12":
        cols = [c for c in cols if c != "tsmom_sign_12"]
    elif variant == "only_tsmom_sign12":
        cols = [c for c in cols if c != "tsmom_12"]
    else:
        raise ValueError(variant)

    if "twostep" in cfg:
        cfg["twostep"]["feature_columns"] = cols
    else:
        cfg["onestep"]["feature_columns"] = cols

    cfg["name"] = f"{Path(base_cfg).stem}_robust_{variant}"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_tsmom_ablation() -> None:
    variants = ["both", "only_tsmom12", "only_tsmom_sign12"]
    base_cfgs = [
        CONFIGS / "final_twostep_elasticnet_rolling_nocost.yaml",
        CONFIGS / "final_twostep_random_forest_rolling_nocost.yaml",
        CONFIGS / "final_onestep_torch_rolling_nocost_v2.yaml",
    ]

    robust_cfgs: list[Path] = []
    for base in base_cfgs:
        for variant in variants:
            out = CONFIGS / f"{base.stem}_robust_{variant}.yaml"
            _write_variant_config(base, out, variant)
            robust_cfgs.append(out)

    for cfg in robust_cfgs:
        if "twostep" in cfg.stem:
            _run([
                "uv",
                "run",
                "--with",
                "pandas",
                "--with",
                "pyarrow",
                "--with",
                "pyyaml",
                "--with",
                "numpy",
                "--with",
                "scikit-learn",
                "--with",
                "lightgbm",
                "python",
                "apps/train_twostep/main.py",
                "--config",
                str(cfg),
            ])
        else:
            _run([
                "uv",
                "run",
                "--with",
                "pandas",
                "--with",
                "pyarrow",
                "--with",
                "pyyaml",
                "--with",
                "numpy",
                "--with",
                "torch",
                "python",
                "apps/train_onestep/main.py",
                "--config",
                str(cfg),
            ])


def write_run_manifest(run_robustness: bool) -> None:
    manifest = {
        "main_specs": [spec.__dict__ for spec in MAIN_SPECS],
        "run_robustness": run_robustness,
        "outputs": {
            "main_table": str(REPORT_TABLES / "final_120_nocost_main_table.csv"),
            "capm_table": str(REPORT_TABLES / "final_120_nocost_capm_table.csv"),
            "costs_table": str(REPORT_TABLES / "final_120_costs_main_table.csv"),
            "cum_plot": str(REPORT_FIGS / "final_120_nocost_cumulative_returns.png"),
        },
    }
    (REPORT_TABLES / "final_120_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final 120 model pack and reporting outputs.")
    parser.add_argument("--run-robustness", action="store_true", help="Run tsmom ablation variants after main pack.")
    args = parser.parse_args()

    _ensure_dirs()
    run_main_models()
    build_main_outputs()
    build_cost_outputs()
    if args.run_robustness:
        run_tsmom_ablation()
    write_run_manifest(args.run_robustness)
    print("DONE: final 120 pack complete")


if __name__ == "__main__":
    main()
