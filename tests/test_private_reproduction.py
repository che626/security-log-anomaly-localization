import numpy as np
import pandas as pd

from scripts.run_private_reproduction import (
    normalize_oof_predictions,
    prediction_distribution,
    validate_legacy_run_args,
)
from seclog.config import load_config


def test_legacy_oof_and_run_args_are_normalized() -> None:
    config = load_config("configs/smoke.yaml")
    run_args = {
        "vocab_size": 512,
        "emb_dim": 8,
        "hidden": 16,
        "layers": 1,
        "dropout": 0.0,
        "max_tokens": 16,
        "seed": 7,
        "seeds": "7",
        "folds": 2,
        "epochs": 1,
        "batch_size": 4,
        "eval_batch_size": 4,
        "lr": 0.001,
        "weight_decay": 0.0,
        "o_weight": 0.16,
        "boundary_loss_weight": 0.20,
        "global_loss_weight": 0.12,
        "patience": 1,
    }
    validate_legacy_run_args(run_args, config)
    prediction = {
        "tag": np.zeros((2, 21), dtype=np.float32),
        "start": np.zeros((2, 10), dtype=np.float32),
        "end": np.zeros((2, 10), dtype=np.float32),
        "global": np.zeros(11, dtype=np.float32),
    }
    normalized = normalize_oof_predictions({1: [prediction]})
    assert normalized[1]["tag"].shape == (2, 21)


def test_prediction_distribution_contains_no_row_ids() -> None:
    frame = pd.DataFrame(
        {
            "has_anomaly": [0, 1, 1],
            "primary_anomaly_type": ["none", "timeout_retry", "timeout_retry"],
        }
    )
    assert prediction_distribution(frame) == {
        "has_anomaly": {"0": 1, "1": 2},
        "primary_anomaly_type": {"timeout_retry": 2, "none": 1},
    }
