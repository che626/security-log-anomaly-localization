from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ProjectConfig
from .constants import ANOMALY_TYPES
from .data import Sample
from .decode import decode_logits_to_frame
from .features import encode_log_line, nonempty_log_lines
from .model import LogBoundaryNetwork
from .schemas import SchemaError, validate_prediction_frame, validate_test_frame
from .training import (
    Prediction,
    average_logits,
    build_dataloader,
    choose_torch_device,
    collect_model_outputs,
)


def validate_checkpoint_metadata(payload: dict[str, Any], config: ProjectConfig) -> None:
    if payload.get("labels") != list(ANOMALY_TYPES):
        raise ValueError("checkpoint labels do not match this package")
    if payload.get("model_config") != asdict(config.model):
        raise ValueError("checkpoint model dimensions do not match the configuration")
    if payload.get("feature_config") != asdict(config.features):
        raise ValueError("checkpoint feature configuration does not match")


def load_checkpoint_model(
    checkpoint_path: Path,
    config: ProjectConfig,
    device: torch.device,
) -> LogBoundaryNetwork:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_metadata(payload, config)
    model = LogBoundaryNetwork(
        vocab_size=config.model.vocab_size,
        emb_dim=config.model.emb_dim,
        hidden=config.model.hidden,
        num_layers=config.model.layers,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _test_samples(frame: pd.DataFrame, config: ProjectConfig) -> list[Sample]:
    samples: list[Sample] = []
    for _, row in frame.iterrows():
        lines = nonempty_log_lines(row["log_text"])
        if not lines:
            raise SchemaError(f"id {row['id']} has no non-empty log lines")
        samples.append(
            Sample(
                sid=int(row["id"]),
                lines=lines,
                token_ids=[
                    encode_log_line(
                        line,
                        vocab_size=config.model.vocab_size,
                        max_tokens=config.features.max_tokens,
                    )
                    for line in lines
                ],
            )
        )
    return samples


def _validate_finite(predictions: dict[int, Prediction]) -> None:
    for sample_id, prediction in predictions.items():
        for head, values in prediction.items():
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {head} logits for id {sample_id}")


def predict(
    test_path: Path,
    checkpoint_paths: list[Path],
    config: ProjectConfig,
    output_path: Path,
    device_name: str = "auto",
    allow_degenerate: bool = False,
) -> pd.DataFrame:
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint path is required")
    test_frame = pd.read_csv(test_path)
    validate_test_frame(test_frame)
    samples = _test_samples(test_frame, config)
    device = choose_torch_device(device_name)
    loader = build_dataloader(
        samples,
        config.training.eval_batch_size,
        pin_memory=device.type == "cuda",
    )
    per_checkpoint: list[dict[int, Prediction]] = []
    for checkpoint_path in checkpoint_paths:
        model = load_checkpoint_model(checkpoint_path, config, device)
        per_checkpoint.append(collect_model_outputs(model, loader, device))
    averaged = {
        sample.sid: average_logits(
            [checkpoint_predictions[sample.sid] for checkpoint_predictions in per_checkpoint]
        )
        for sample in samples
    }
    _validate_finite(averaged)
    ordered_predictions = [averaged[sample.sid] for sample in samples]
    output = decode_logits_to_frame(
        samples,
        ordered_predictions,
        config.decoder,
        length_stats={},
    )
    expected_ids = test_frame["id"].tolist()
    validate_prediction_frame(output, expected_ids=expected_ids)
    if len(output) != len(test_frame):
        raise ValueError("prediction row count does not match the test data")
    if not allow_degenerate and len(output) > 1 and output["has_anomaly"].nunique() == 1:
        raise ValueError(
            "all predictions have the same anomaly class; inspect the model or pass "
            "allow_degenerate=True explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    return output
