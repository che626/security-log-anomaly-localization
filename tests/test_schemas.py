import pandas as pd
import pytest

from seclog.schemas import (
    SchemaError,
    validate_prediction_frame,
    validate_test_frame,
    validate_training_frame,
)


def valid_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1],
            "log_text": ["line one\nline two"],
            "has_anomaly": [1],
            "primary_start_idx": [0],
            "primary_end_idx": [1],
            "primary_anomaly_type": ["timeout_retry"],
            "all_spans": ["0|1|timeout_retry"],
        }
    )


def test_training_schema_accepts_valid_frame() -> None:
    validate_training_frame(valid_training_frame())


def test_training_schema_rejects_duplicate_ids() -> None:
    frame = pd.concat([valid_training_frame(), valid_training_frame()], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_training_frame(frame)


def test_prediction_schema_rejects_invalid_normal_span() -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "has_anomaly": [0],
            "primary_start_idx": [0],
            "primary_end_idx": [0],
            "primary_anomaly_type": ["none"],
            "all_spans": [""],
        }
    )
    with pytest.raises(SchemaError, match="-1"):
        validate_prediction_frame(frame, expected_ids=[1])


def test_test_schema_rejects_empty_frame() -> None:
    frame = pd.DataFrame({"id": pd.Series(dtype=int), "log_text": pd.Series(dtype=str)})
    with pytest.raises(SchemaError, match="empty"):
        validate_test_frame(frame)


def test_training_schema_rejects_missing_columns() -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        validate_training_frame(pd.DataFrame({"id": [1]}))
