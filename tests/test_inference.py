from pathlib import Path

import pandas as pd
import pytest

from seclog.config import load_config
from seclog.inference import predict
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
