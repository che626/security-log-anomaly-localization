from pathlib import Path

import pandas as pd
import pytest

from seclog.config import load_config
from seclog.inference import predict, predict_text
from seclog.training import run_training


def test_smoke_prediction_preserves_test_ids(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    training = run_training(
        Path("tests/fixtures/synthetic_data"),
        tmp_path / "train",
        config,
        "cpu",
    )
    arguments = {
        "test_path": Path("tests/fixtures/synthetic_data/test.csv"),
        "checkpoint_paths": [item.path for item in training.checkpoints],
        "config": config,
        "output_path": tmp_path / "submission.csv",
        "device_name": "cpu",
    }
    with pytest.raises(ValueError, match="same anomaly class"):
        predict(**arguments)
    output = predict(**arguments, allow_degenerate=True)
    test_ids = pd.read_csv("tests/fixtures/synthetic_data/test.csv")["id"].tolist()
    assert output["id"].tolist() == test_ids
    assert (tmp_path / "submission.csv").exists()
    demo = predict_text(
        ["request started", "attempt timed out", "retry scheduled"],
        [item.path for item in training.checkpoints],
        config,
        "cpu",
    )
    assert 0.0 <= demo["confidence"] <= 1.0
    assert demo["primary_anomaly_type"] in {
        "none",
        "timeout_retry",
        "resource_exhaustion",
        "slow_burn_warning",
        "state_conflict",
        "parameter_drift",
        "out_of_order",
        "missing_step",
        "duplicate_event",
        "cross_component_mismatch",
        "partial_recovery_loop",
    }
