from collections.abc import Iterable

import pandas as pd

from .constants import ANOMALY_TYPES, SUBMISSION_COLUMNS, TEST_COLUMNS, TRAIN_COLUMNS


class SchemaError(ValueError):
    pass


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"missing columns: {missing}")


def _validate_ids(frame: pd.DataFrame) -> None:
    if frame["id"].isna().any():
        raise SchemaError("id contains null values")
    if frame["id"].duplicated().any():
        raise SchemaError("duplicate ids are not allowed")


def validate_training_frame(frame: pd.DataFrame) -> None:
    _require_columns(frame, TRAIN_COLUMNS)
    _validate_ids(frame)
    if frame.empty:
        raise SchemaError("training frame is empty")
    if not set(frame["has_anomaly"].astype(int)).issubset({0, 1}):
        raise SchemaError("has_anomaly must contain only 0 or 1")
    known = set(ANOMALY_TYPES) | {"none"}
    unknown = set(frame["primary_anomaly_type"].astype(str)) - known
    if unknown:
        raise SchemaError(f"unknown anomaly types: {sorted(unknown)}")


def validate_test_frame(frame: pd.DataFrame) -> None:
    _require_columns(frame, TEST_COLUMNS)
    _validate_ids(frame)
    if frame.empty:
        raise SchemaError("test frame is empty")


def validate_prediction_frame(frame: pd.DataFrame, expected_ids: Iterable[int]) -> None:
    _require_columns(frame, SUBMISSION_COLUMNS)
    _validate_ids(frame)
    expected = list(expected_ids)
    if frame["id"].tolist() != expected:
        raise SchemaError("prediction ids or order do not match the expected ids")
    normal = frame["has_anomaly"].astype(int).eq(0)
    if not frame.loc[normal, "primary_start_idx"].astype(int).eq(-1).all():
        raise SchemaError("normal rows must use -1 for primary_start_idx")
    if not frame.loc[normal, "primary_end_idx"].astype(int).eq(-1).all():
        raise SchemaError("normal rows must use -1 for primary_end_idx")
    if not frame.loc[normal, "primary_anomaly_type"].astype(str).eq("none").all():
        raise SchemaError("normal rows must use none as primary_anomaly_type")
    anomalous = ~normal
    unknown = set(frame.loc[anomalous, "primary_anomaly_type"].astype(str)) - set(
        ANOMALY_TYPES
    )
    if unknown:
        raise SchemaError(f"unknown predicted anomaly types: {sorted(unknown)}")
    if (frame.loc[anomalous, "primary_start_idx"] < 0).any():
        raise SchemaError("anomalous rows require non-negative start indices")
    if (
        frame.loc[anomalous, "primary_end_idx"].astype(int)
        < frame.loc[anomalous, "primary_start_idx"].astype(int)
    ).any():
        raise SchemaError("primary_end_idx must not be less than primary_start_idx")
