"""Training path for binary public datasets using the existing log encoder."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .public_metrics import apply_temperature, choose_f1_threshold, fit_temperature
from .public_model import (
    PublicModelSample,
    PublicSequenceDataset,
    build_public_model_samples,
    create_public_model,
    pack_public_batch,
    public_line_labels,
)
from .public_protocol import (
    PreparedSample,
    PublicPrediction,
    PublicProtocolError,
    TaskProfile,
    spans_from_mask,
)
from .training import amp_enabled, choose_torch_device, fix_all_seeds


@dataclass(frozen=True)
class PublicNeuralConfig:
    vocab_size: int = 4096
    max_tokens: int = 96
    emb_dim: int = 64
    hidden: int = 96
    layers: int = 1
    dropout: float = 0.15
    epochs: int = 12
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    boundary_loss_weight: float = 0.20
    global_loss_weight: float = 0.20
    patience: int = 3
    seed: int = 20260711

    def validate(self) -> None:
        if min(self.vocab_size, self.max_tokens, self.emb_dim, self.hidden, self.layers) <= 0:
            raise PublicProtocolError("public neural dimensions must be positive")
        if min(self.epochs, self.batch_size, self.eval_batch_size, self.patience) <= 0:
            raise PublicProtocolError("public neural training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise PublicProtocolError("public neural optimizer settings are invalid")
        if self.boundary_loss_weight < 0 or self.global_loss_weight < 0:
            raise PublicProtocolError("public neural loss weights cannot be negative")


@dataclass(frozen=True)
class PublicNeuralRun:
    profile: str
    threshold: float
    temperature: float
    epochs_trained: int
    validation_predictions: tuple[PublicPrediction, ...]
    test_predictions: tuple[PublicPrediction, ...]
    metadata: dict[str, Any]


def load_public_neural_config(path: Path) -> PublicNeuralConfig:
    """Load a deliberately small, strict configuration for public neural runs."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PublicProtocolError(f"cannot read public neural config: {path}") from exc
    if not isinstance(raw, dict):
        raise PublicProtocolError("public neural config root must be a mapping")
    payload = raw.get("public_neural", raw)
    if not isinstance(payload, dict):
        raise PublicProtocolError("public_neural config section must be a mapping")
    allowed = {field.name for field in fields(PublicNeuralConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PublicProtocolError(f"public neural config contains unknown fields: {unknown}")
    try:
        config = PublicNeuralConfig(**payload)
    except TypeError as exc:
        raise PublicProtocolError("public neural config contains invalid values") from exc
    config.validate()
    return config


def _loader(samples: tuple[PublicModelSample, ...], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        PublicSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=pack_public_batch,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "offsets", "owner", "mask", "pos_feats", "tags", "starts", "ends", "global_labels")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def _loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    profile: TaskProfile,
    config: PublicNeuralConfig,
    device: torch.device,
    global_class_weights: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, enabled=amp_enabled(device)):
        tags, starts, ends, global_logits = model(
            batch["input_ids"], batch["offsets"], batch["owner"], batch["mask"], batch["pos_feats"]
        )
        global_loss = F.cross_entropy(
            global_logits, batch["global_labels"], weight=global_class_weights
        )
        if profile is TaskProfile.SEQUENCE_BINARY:
            return global_loss
        tag_weight = torch.tensor((0.2, 1.0, 1.0), dtype=torch.float32, device=device)
        tag_loss = F.cross_entropy(
            tags.reshape(-1, 3), batch["tags"].reshape(-1), weight=tag_weight, ignore_index=-100
        )
        valid = batch["mask"].unsqueeze(-1).expand_as(starts)
        positives = batch["starts"][valid].sum().clamp_min(1.0)
        negatives = valid.sum().to(starts.dtype) - positives
        pos_weight = torch.sqrt((negatives / positives).clamp(min=3.0, max=25.0))
        start_loss = F.binary_cross_entropy_with_logits(
            starts, batch["starts"], pos_weight=pos_weight, reduction="none"
        )[valid].mean()
        end_loss = F.binary_cross_entropy_with_logits(
            ends, batch["ends"], pos_weight=pos_weight, reduction="none"
        )[valid].mean()
        return tag_loss + config.boundary_loss_weight * (start_loss + end_loss) + config.global_loss_weight * global_loss


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    profile: TaskProfile,
    config: PublicNeuralConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    global_class_weights: torch.Tensor,
) -> float:
    model.train(optimizer is not None)
    losses: list[float] = []
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = _loss(
            model,
            _to_device(batch, device),
            profile,
            config,
            device,
            global_class_weights,
        )
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _global_class_weights(samples: tuple[PublicModelSample, ...]) -> torch.Tensor:
    """Inverse-square-root class weights fitted strictly on training samples."""

    counts = np.ones(2, dtype=np.float64)
    for sample in samples:
        counts[sample.global_label] += 1
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def _collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    model.eval()
    output: dict[str, dict[str, np.ndarray]] = {}
    for batch in loader:
        inputs = _to_device(batch, device)
        tags, starts, ends, global_logits = model(
            inputs["input_ids"], inputs["offsets"], inputs["owner"], inputs["mask"], inputs["pos_feats"]
        )
        tags_array = tags.detach().float().cpu().numpy()
        starts_array = starts.detach().float().cpu().numpy()
        ends_array = ends.detach().float().cpu().numpy()
        global_array = global_logits.detach().float().cpu().numpy()
        for index, item in enumerate(batch["samples"]):
            length = len(item.prepared.lines)
            output[item.prepared.sid] = {
                "tags": tags_array[index, :length],
                "starts": starts_array[index, :length, 0],
                "ends": ends_array[index, :length, 0],
                "global": global_array[index],
            }
    return output


def _probability(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 1e-6, 1.0 - 1e-6)


def _sequence_scores(
    samples: tuple[PublicModelSample, ...], output: dict[str, dict[str, np.ndarray]]
) -> np.ndarray:
    values = []
    for item in samples:
        logits = output[item.prepared.sid]["global"]
        exp = np.exp(logits - np.max(logits))
        values.append(float(exp[0] / exp.sum()))
    return _probability(np.asarray(values))


def _span_line_scores(
    samples: tuple[PublicModelSample, ...], output: dict[str, dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    scores: list[float] = []
    labels: list[int] = []
    locations: list[tuple[int, int]] = []
    for sample_index, item in enumerate(samples):
        value = output[item.prepared.sid]
        tag_logits = value["tags"]
        tag_exp = np.exp(tag_logits - tag_logits.max(axis=1, keepdims=True))
        tag_prob = tag_exp / tag_exp.sum(axis=1, keepdims=True)
        anomaly_tag = tag_prob[:, 1] + tag_prob[:, 2]
        start = 1.0 / (1.0 + np.exp(-np.clip(value["starts"], -60, 60)))
        end = 1.0 / (1.0 + np.exp(-np.clip(value["ends"], -60, 60)))
        combined = _probability(0.70 * anomaly_tag + 0.15 * start + 0.15 * end)
        scores.extend(float(score) for score in combined)
        labels.extend(public_line_labels(item))
        locations.extend((sample_index, line_index) for line_index in range(len(combined)))
    return np.asarray(scores), np.asarray(labels, dtype=int), locations


def _span_predictions(
    samples: tuple[PublicModelSample, ...],
    scores: np.ndarray,
    locations: list[tuple[int, int]],
    threshold: float,
) -> tuple[PublicPrediction, ...]:
    per_sample: list[list[float]] = [[] for _ in samples]
    for score, (sample_index, _line_index) in zip(scores, locations):
        per_sample[sample_index].append(float(score))
    result: list[PublicPrediction] = []
    for item, line_scores in zip(samples, per_sample):
        spans = spans_from_mask(score >= threshold for score in line_scores)
        result.append(
            PublicPrediction(
                sid=item.prepared.sid,
                score=float(max(line_scores)),
                has_anomaly=int(bool(spans)),
                spans=spans,
            )
        )
    return tuple(result)


def save_public_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: PublicNeuralConfig,
    profile: TaskProfile,
    manifest_sha256: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "checkpoint_kind": "public_binary_log_benchmark",
            "public_profile": profile.value,
            "public_labels": ("anomaly", "normal"),
            "model_config": asdict(config),
            "data_manifest_sha256": manifest_sha256,
        },
        path,
    )


def load_public_checkpoint(
    path: Path,
    model: torch.nn.Module,
    profile: TaskProfile,
    device: torch.device,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_kind") != "public_binary_log_benchmark":
        raise PublicProtocolError("checkpoint is not a public binary benchmark checkpoint")
    if payload.get("public_profile") != profile.value:
        raise PublicProtocolError("public checkpoint profile does not match")
    if payload.get("public_labels") != ("anomaly", "normal"):
        raise PublicProtocolError("public checkpoint label vocabulary does not match")
    if manifest_sha256 is not None and payload.get("data_manifest_sha256") != manifest_sha256:
        raise PublicProtocolError("public checkpoint manifest does not match")
    model.load_state_dict(payload["state_dict"])
    return payload


def train_public_neural(
    profile: TaskProfile,
    train_samples: Iterable[PreparedSample],
    validation_samples: Iterable[PreparedSample],
    test_samples: Iterable[PreparedSample],
    config: PublicNeuralConfig,
    *,
    device_name: str = "auto",
    checkpoint_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> PublicNeuralRun:
    """Train one public binary run and select calibration/thresholds on validation only."""

    config.validate()
    train_prepared = tuple(train_samples)
    validation_prepared = tuple(validation_samples)
    test_prepared = tuple(test_samples)
    if not train_prepared or not validation_prepared or not test_prepared:
        raise PublicProtocolError("public neural train, validation, and test samples must be non-empty")
    device = choose_torch_device(device_name)
    fix_all_seeds(config.seed)
    train = build_public_model_samples(
        train_prepared, profile, vocab_size=config.vocab_size, max_tokens=config.max_tokens
    )
    validation = build_public_model_samples(
        validation_prepared, profile, vocab_size=config.vocab_size, max_tokens=config.max_tokens
    )
    test = build_public_model_samples(
        test_prepared, profile, vocab_size=config.vocab_size, max_tokens=config.max_tokens
    )
    model = create_public_model(
        vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    global_class_weights = _global_class_weights(train).to(device)
    train_loader = _loader(train, config.batch_size, shuffle=True)
    validation_loader = _loader(validation, config.eval_batch_size, shuffle=False)
    test_loader = _loader(test, config.eval_batch_size, shuffle=False)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    epochs_trained = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        _epoch(model, train_loader, profile, config, device, optimizer, global_class_weights)
        validation_loss = _epoch(
            model, validation_loader, profile, config, device, None, global_class_weights
        )
        epochs_trained = epoch + 1
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("public neural training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        save_public_checkpoint(checkpoint_path, model, config, profile, manifest_sha256)
    validation_output = _collect(model, validation_loader, device)
    test_output = _collect(model, test_loader, device)
    if profile is TaskProfile.SEQUENCE_BINARY:
        validation_raw = _sequence_scores(validation, validation_output)
        validation_labels = np.asarray([item.prepared.has_anomaly for item in validation], dtype=int)
        temperature = fit_temperature(validation_raw, validation_labels)
        validation_scores = apply_temperature(validation_raw, temperature)
        threshold = choose_f1_threshold(validation_scores, validation_labels)
        test_scores = apply_temperature(_sequence_scores(test, test_output), temperature)
        validation_predictions = tuple(
            PublicPrediction(item.prepared.sid, float(score), int(score >= threshold))
            for item, score in zip(validation, validation_scores)
        )
        test_predictions = tuple(
            PublicPrediction(item.prepared.sid, float(score), int(score >= threshold))
            for item, score in zip(test, test_scores)
        )
    elif profile is TaskProfile.SPAN_BINARY:
        validation_raw, validation_labels, validation_locations = _span_line_scores(
            validation, validation_output
        )
        temperature = fit_temperature(validation_raw, validation_labels)
        validation_scores = apply_temperature(validation_raw, temperature)
        threshold = choose_f1_threshold(validation_scores, validation_labels)
        test_raw, _test_labels, test_locations = _span_line_scores(test, test_output)
        test_scores = apply_temperature(test_raw, temperature)
        validation_predictions = _span_predictions(
            validation, validation_scores, validation_locations, threshold
        )
        test_predictions = _span_predictions(test, test_scores, test_locations, threshold)
    else:
        raise PublicProtocolError(f"unsupported public profile: {profile}")
    elapsed = time.perf_counter() - started
    return PublicNeuralRun(
        profile=profile.value,
        threshold=float(threshold),
        temperature=float(temperature),
        epochs_trained=epochs_trained,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        metadata={
            "device": device.type,
            "elapsed_seconds": float(elapsed),
            "best_validation_loss": float(best_loss),
            "seed": config.seed,
            "global_class_weights": [float(value) for value in global_class_weights.detach().cpu()],
        },
    )


def predict_public_neural(
    profile: TaskProfile,
    samples: Iterable[PreparedSample],
    config: PublicNeuralConfig,
    *,
    checkpoint_path: Path,
    source_manifest_sha256: str,
    threshold: float,
    temperature: float,
    device_name: str = "auto",
) -> tuple[PublicPrediction, ...]:
    """Apply a validated public checkpoint to a target public corpus.

    Threshold and temperature are supplied from the source run's validation
    record. Target labels are deliberately not consulted in this function.
    """

    config.validate()
    if not 0.0 <= threshold <= 1.0:
        raise PublicProtocolError("public neural threshold must be between zero and one")
    prepared = tuple(samples)
    if not prepared:
        raise PublicProtocolError("cannot predict an empty public target dataset")
    device = choose_torch_device(device_name)
    model_samples = build_public_model_samples(
        prepared, profile, vocab_size=config.vocab_size, max_tokens=config.max_tokens
    )
    model = create_public_model(
        vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    load_public_checkpoint(
        checkpoint_path,
        model,
        profile,
        device,
        manifest_sha256=source_manifest_sha256,
    )
    output = _collect(model, _loader(model_samples, config.eval_batch_size, shuffle=False), device)
    if profile is TaskProfile.SEQUENCE_BINARY:
        scores = apply_temperature(_sequence_scores(model_samples, output), temperature)
        return tuple(
            PublicPrediction(item.prepared.sid, float(score), int(score >= threshold))
            for item, score in zip(model_samples, scores)
        )
    if profile is TaskProfile.SPAN_BINARY:
        raw, _labels, locations = _span_line_scores(model_samples, output)
        scores = apply_temperature(raw, temperature)
        return _span_predictions(model_samples, scores, locations, threshold)
    raise PublicProtocolError(f"unsupported public profile: {profile}")
