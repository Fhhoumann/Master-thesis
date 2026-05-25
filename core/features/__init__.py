"""Feature engineering primitives and pipelines."""

from core.features.audit import build_feature_audit, resolve_feature_columns, validate_feature_audit
from core.features.macro import load_macro_factors, merge_macro_features
from core.features.technical import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SET_REGISTRY,
    MACRO_FEATURE_COLUMNS,
    SUPERVISOR_MACRO_INTERACTION_COLUMNS,
    FeatureResult,
    generate_features,
    generate_features_from_datacube,
    get_feature_columns,
    list_feature_sets,
)

__all__ = [
    "FEATURE_COLUMNS",
    "BASE_FEATURE_COLUMNS",
    "FEATURE_SET_REGISTRY",
    "MACRO_FEATURE_COLUMNS",
    "SUPERVISOR_MACRO_INTERACTION_COLUMNS",
    "build_feature_audit",
    "FeatureResult",
    "generate_features",
    "generate_features_from_datacube",
    "get_feature_columns",
    "load_macro_factors",
    "list_feature_sets",
    "merge_macro_features",
    "resolve_feature_columns",
    "validate_feature_audit",
]
