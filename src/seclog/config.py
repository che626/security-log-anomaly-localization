from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    emb_dim: int
    hidden: int
    layers: int
    dropout: float


@dataclass(frozen=True)
class FeatureConfig:
    max_tokens: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    seeds: list[int]
    folds: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    o_weight: float
    boundary_loss_weight: float
    global_loss_weight: float
    patience: int


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    features: FeatureConfig
    training: TrainingConfig
    decoder: dict[str, object]


def _section(cls, name: str, payload: object):
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    return cls(**payload)


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    expected = {"model", "features", "training", "decoder"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(f"configuration sections invalid; missing={missing}, unknown={unknown}")
    model = _section(ModelConfig, "model", raw["model"])
    features = _section(FeatureConfig, "features", raw["features"])
    training = _section(TrainingConfig, "training", raw["training"])
    decoder = raw["decoder"]
    if not isinstance(decoder, dict):
        raise ValueError("decoder must be a mapping")
    if model.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if training.folds < 2:
        raise ValueError("folds must be at least 2")
    if min(training.epochs, training.batch_size, training.eval_batch_size) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    return ProjectConfig(model, features, training, dict(decoder))
