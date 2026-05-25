from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from core.config.schemas import ExperimentConfig
from core.features.technical import FEATURE_COLUMNS, get_feature_columns

PipelineType = Literal["twostep", "onestep"]


@dataclass(slots=True)
class FeatureAuditResult:
    pipeline_type: PipelineType
    experiment_name: str
    enforce_feature_consistency: bool
    feature_columns_used: list[str]
    feature_count: int
    duplicate_features: list[str]
    missing_in_frame: list[str]
    canonical_missing_in_frame: list[str]
    extra_in_frame: list[str]
    all_feature_columns_present_in_frame: bool
    all_canonical_columns_present_in_frame: bool
    has_duplicates: bool
    uses_canonical_feature_order: bool
    canonical_feature_columns: list[str]
    cs_mom_included: bool
    cs_mom_in_canonical_list: bool
    cs_mom_winsorized_via_feature_loop: bool
    hard_fail_active: bool

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_feature_columns(cfg: ExperimentConfig, pipeline_type: PipelineType) -> list[str]:
    if pipeline_type == "twostep":
        configured = cfg.twostep.feature_columns
        feature_set_name = cfg.twostep.feature_set_name
    else:
        configured = cfg.onestep.feature_columns
        feature_set_name = cfg.onestep.feature_set_name
    if configured:
        return list(configured)
    if feature_set_name:
        return get_feature_columns(feature_set_name)
    return FEATURE_COLUMNS.copy()


def _find_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates


def build_feature_audit(
    frame: pd.DataFrame,
    *,
    cfg: ExperimentConfig,
    pipeline_type: PipelineType,
    feature_columns: list[str],
) -> FeatureAuditResult:
    if pipeline_type == "twostep":
        feature_set_name = cfg.twostep.feature_set_name
    else:
        feature_set_name = cfg.onestep.feature_set_name
    canonical_feature_columns = (
        get_feature_columns(feature_set_name) if feature_set_name else FEATURE_COLUMNS.copy()
    )

    duplicates = _find_duplicates(feature_columns)
    missing_in_frame = [column for column in feature_columns if column not in frame.columns]
    canonical_missing_in_frame = [
        column for column in canonical_feature_columns if column not in frame.columns
    ]
    extra_in_frame = [column for column in frame.columns if column not in feature_columns]
    uses_canonical = feature_columns == canonical_feature_columns
    hard_fail_active = cfg.enforce_feature_consistency or cfg.name.startswith("final_")

    return FeatureAuditResult(
        pipeline_type=pipeline_type,
        experiment_name=cfg.name,
        enforce_feature_consistency=cfg.enforce_feature_consistency,
        feature_columns_used=feature_columns.copy(),
        feature_count=len(feature_columns),
        duplicate_features=duplicates,
        missing_in_frame=missing_in_frame,
        canonical_missing_in_frame=canonical_missing_in_frame,
        extra_in_frame=extra_in_frame,
        all_feature_columns_present_in_frame=not missing_in_frame,
        all_canonical_columns_present_in_frame=not canonical_missing_in_frame,
        has_duplicates=bool(duplicates),
        uses_canonical_feature_order=uses_canonical,
        canonical_feature_columns=canonical_feature_columns.copy(),
        cs_mom_included="cs_mom_12_1" in feature_columns,
        cs_mom_in_canonical_list="cs_mom_12_1" in canonical_feature_columns,
        cs_mom_winsorized_via_feature_loop="cs_mom_12_1" in canonical_feature_columns,
        hard_fail_active=hard_fail_active,
    )


def validate_feature_audit(audit: FeatureAuditResult) -> None:
    errors: list[str] = []
    if not audit.feature_columns_used:
        errors.append("feature list is empty")
    if audit.has_duplicates:
        errors.append(f"duplicate features: {audit.duplicate_features}")
    if not audit.all_feature_columns_present_in_frame:
        errors.append(f"missing columns in dataframe: {audit.missing_in_frame}")

    if audit.hard_fail_active and not audit.uses_canonical_feature_order:
        errors.append(
            "feature order/list does not match canonical FEATURE_COLUMNS for final/enforced run. "
            f"used={audit.feature_columns_used}, canonical={audit.canonical_feature_columns}"
        )
    if audit.hard_fail_active and not audit.all_canonical_columns_present_in_frame:
        errors.append(
            "canonical FEATURE_COLUMNS missing in dataframe: "
            f"{audit.canonical_missing_in_frame}"
        )

    if errors:
        raise ValueError("Feature audit failed: " + " | ".join(errors))
