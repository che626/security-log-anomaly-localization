import hashlib
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .constants import ANOMALY_TYPES, N_LABELS
from .data import (
    DatasetManifest,
    Sample,
    SpanSequenceDataset,
    load_dataset,
    pack_log_batch,
)
from .decode import decode_logits_to_frame, samples_to_truth_frame
from .metrics import evaluate_predictions
from .model import LogBoundaryNetwork, boundary_positive_weights, sequence_loss_weights

Prediction = dict[str, np.ndarray]


@dataclass(frozen=True)
class FoldCheckpoint:
    fold: int
    seed: int
    path: Path


@dataclass(frozen=True)
class TrainingResult:
    checkpoints: tuple[FoldCheckpoint, ...]
    oof_path: Path
    metrics_path: Path
    manifest_path: Path


def fix_all_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def choose_torch_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(normalized)


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def build_dataloader(
    samples: list[Sample],
    batch_size: int,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        SpanSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=pack_log_batch,
        pin_memory=pin_memory,
    )


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "offsets",
        "owner",
        "mask",
        "pos_feats",
        "labels",
        "start_labels",
        "end_labels",
        "global_labels",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def create_model(config: ProjectConfig, device: torch.device) -> LogBoundaryNetwork:
    return LogBoundaryNetwork(
        vocab_size=config.model.vocab_size,
        emb_dim=config.model.emb_dim,
        hidden=config.model.hidden,
        num_layers=config.model.layers,
        dropout=config.model.dropout,
    ).to(device)


def compute_training_loss(
    model: nn.Module,
    batch_dev: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    boundary_pos_weights: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, enabled=amp_enabled(device)):
        tag, start, end, global_logits = model(
            batch_dev["input_ids"],
            batch_dev["offsets"],
            batch_dev["owner"],
            batch_dev["mask"],
            batch_dev["pos_feats"],
        )
        tag_loss = F.cross_entropy(
            tag.reshape(-1, N_LABELS),
            batch_dev["labels"].reshape(-1),
            weight=class_weights,
            ignore_index=-100,
        )
        valid = batch_dev["mask"].unsqueeze(-1).expand_as(start)
        start_weights = torch.where(
            batch_dev["start_labels"] > 0.5,
            boundary_pos_weights.view(1, 1, -1),
            torch.ones_like(batch_dev["start_labels"]),
        )
        end_weights = torch.where(
            batch_dev["end_labels"] > 0.5,
            boundary_pos_weights.view(1, 1, -1),
            torch.ones_like(batch_dev["end_labels"]),
        )
        start_loss = F.binary_cross_entropy_with_logits(
            start,
            batch_dev["start_labels"],
            weight=start_weights,
            reduction="none",
        )[valid].mean()
        end_loss = F.binary_cross_entropy_with_logits(
            end,
            batch_dev["end_labels"],
            weight=end_weights,
            reduction="none",
        )[valid].mean()
        global_loss = F.cross_entropy(global_logits, batch_dev["global_labels"])
        return (
            tag_loss
            + config.training.boundary_loss_weight * (start_loss + end_loss)
            + config.training.global_loss_weight * global_loss
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    boundary_pos_weights: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_training_loss(
            model,
            batch_to_device(batch, device),
            class_weights,
            boundary_pos_weights,
            config,
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    boundary_pos_weights: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        loss = compute_training_loss(
            model,
            batch_to_device(batch, device),
            class_weights,
            boundary_pos_weights,
            config,
            device,
        )
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_model_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[int, Prediction]:
    model.eval()
    predictions: dict[int, Prediction] = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        offsets = batch["offsets"].to(device)
        owner = batch["owner"].to(device)
        mask = batch["mask"].to(device)
        pos_feats = batch["pos_feats"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled(device)):
            tag, start, end, global_logits = model(
                input_ids, offsets, owner, mask, pos_feats
            )
        tag_values = tag.detach().float().cpu().numpy()
        start_values = start.detach().float().cpu().numpy()
        end_values = end.detach().float().cpu().numpy()
        global_values = global_logits.detach().float().cpu().numpy()
        for batch_index, sample in enumerate(batch["samples"]):
            length = len(sample.lines)
            predictions[sample.sid] = {
                "tag": tag_values[batch_index, :length].copy(),
                "start": start_values[batch_index, :length].copy(),
                "end": end_values[batch_index, :length].copy(),
                "global": global_values[batch_index].copy(),
            }
    return predictions


def average_logits(predictions: list[Prediction]) -> Prediction:
    if not predictions:
        raise ValueError("cannot average an empty prediction list")
    return {
        key: sum(prediction[key].astype(np.float32) for prediction in predictions)
        / len(predictions)
        for key in predictions[0]
    }


def _manifest_hash(manifest: DatasetManifest) -> str:
    encoded = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ProjectConfig,
    fold: int,
    seed: int,
    manifest_hash: str,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_config": asdict(config.model),
        "feature_config": asdict(config.features),
        "labels": list(ANOMALY_TYPES),
        "fold": fold,
        "seed": seed,
        "data_manifest_sha256": manifest_hash,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    config: ProjectConfig,
    device: torch.device,
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("labels") != list(ANOMALY_TYPES):
        raise ValueError("checkpoint labels do not match this package")
    if payload.get("model_config") != asdict(config.model):
        raise ValueError("checkpoint model dimensions do not match the configuration")
    if payload.get("feature_config") != asdict(config.features):
        raise ValueError("checkpoint feature configuration does not match")
    if (
        expected_manifest_hash is not None
        and payload.get("data_manifest_sha256") != expected_manifest_hash
    ):
        raise ValueError("checkpoint data manifest does not match")
    model.load_state_dict(payload["state_dict"])
    return payload


def train_fold_and_collect(
    seed: int,
    fold: int,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    config: ProjectConfig,
    device: torch.device,
    checkpoint_path: Path,
    manifest_hash: str,
) -> dict[int, Prediction]:
    fix_all_seeds(seed + fold)
    model = create_model(config, device)
    class_weights = sequence_loss_weights(
        train_samples, config.training.o_weight
    ).to(device)
    positive_weights = boundary_positive_weights(train_samples).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = build_dataloader(
        train_samples,
        config.training.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = build_dataloader(
        validation_samples,
        config.training.eval_batch_size,
        pin_memory=device.type == "cuda",
    )
    best_loss = float("inf")
    no_improvement = 0
    for _epoch in range(config.training.epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            positive_weights,
            config,
            device,
        )
        validation_loss = validate_epoch_loss(
            model,
            validation_loader,
            class_weights,
            positive_weights,
            config,
            device,
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            save_checkpoint(
                checkpoint_path, model, config, fold, seed, manifest_hash
            )
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config.training.patience:
                break
    load_checkpoint(
        checkpoint_path,
        model,
        config,
        device,
        expected_manifest_hash=manifest_hash,
    )
    return collect_model_outputs(model, validation_loader, device)


def run_training(
    data_dir: Path,
    output_dir: Path,
    config: ProjectConfig,
    device_name: str = "auto",
) -> TrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_torch_device(device_name)
    train_frame, _test_frame, train_samples, _test_samples, manifest = load_dataset(
        data_dir=data_dir,
        cache_dir=output_dir / "cache",
        vocab_size=config.model.vocab_size,
        max_tokens=config.features.max_tokens,
    )
    manifest_hash = _manifest_hash(manifest)
    manifest_path = output_dir / "training-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": asdict(manifest),
                "dataset_manifest_sha256": manifest_hash,
                "config": asdict(config),
                "device": device.type,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stratification = train_frame.apply(
        lambda row: (
            "normal"
            if int(row["has_anomaly"]) == 0
            else str(row["primary_anomaly_type"])
        ),
        axis=1,
    ).to_numpy()
    checkpoints: list[FoldCheckpoint] = []
    oof_predictions: dict[int, Prediction] = {}
    for seed in config.training.seeds:
        splitter = StratifiedKFold(
            n_splits=config.training.folds,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(np.arange(len(train_samples)), stratification)
        ):
            fold_train = [train_samples[index] for index in train_indices]
            fold_validation = [train_samples[index] for index in validation_indices]
            checkpoint_path = output_dir / f"model-seed{seed}-fold{fold}.pt"
            fold_predictions = train_fold_and_collect(
                seed,
                fold,
                fold_train,
                fold_validation,
                config,
                device,
                checkpoint_path,
                manifest_hash,
            )
            if len(config.training.seeds) == 1:
                oof_predictions.update(fold_predictions)
            else:
                for sample_id, prediction in fold_predictions.items():
                    existing = oof_predictions.get(sample_id)
                    oof_predictions[sample_id] = (
                        prediction
                        if existing is None
                        else average_logits([existing, prediction])
                    )
            checkpoints.append(FoldCheckpoint(fold=fold, seed=seed, path=checkpoint_path))
    oof_path = output_dir / "oof-predictions.pkl"
    with oof_path.open("wb") as handle:
        pickle.dump(oof_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ordered_predictions = [oof_predictions[sample.sid] for sample in train_samples]
    prediction_frame = decode_logits_to_frame(
        train_samples, ordered_predictions, config.decoder, length_stats={}
    )
    metrics = evaluate_predictions(prediction_frame, samples_to_truth_frame(train_samples))
    metrics_path = output_dir / "oof-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return TrainingResult(
        checkpoints=tuple(checkpoints),
        oof_path=oof_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )
