from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.config.schemas import TrackingConfig


@contextmanager
def start_run(
    tracking_config: TrackingConfig,
    *,
    run_name: str | None = None,
) -> Iterator[object | None]:
    if not tracking_config.enabled:
        yield None
        return

    run = None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_config.tracking_uri)
        mlflow.set_experiment(tracking_config.experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as exc:
        print(f"MLflow tracking disabled for this run due to error: {exc}")
        if run is None:
            return


def log_params(run: object | None, params: dict) -> None:
    if run is None:
        return
    import mlflow

    mlflow.log_params(params)


def log_metrics(run: object | None, metrics: dict[str, float]) -> None:
    if run is None:
        return
    import mlflow

    mlflow.log_metrics(metrics)


def log_json_artifact(run: object | None, payload: dict, artifact_file: str) -> None:
    if run is None:
        return
    import mlflow

    temp_dir = Path("artifacts/tmp_mlflow")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / artifact_file
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    mlflow.log_artifact(str(file_path))
