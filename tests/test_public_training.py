import pytest
import torch

from seclog.public_model import build_public_model_samples
from seclog.public_protocol import PreparedSample, PublicProtocolError, PublicSpan, TaskProfile, template_signature
from seclog.public_training import (
    PublicNeuralConfig,
    _global_class_weights,
    create_public_model,
    load_public_checkpoint,
    predict_public_neural,
    train_public_neural,
)


def _samples(prefix: str, count: int, profile: TaskProfile) -> tuple[PreparedSample, ...]:
    result = []
    for index in range(count):
        anomaly = int(index % 2 == 1)
        lines = ("service healthy", "disk failure" if anomaly else "service healthy")
        result.append(
            PreparedSample(
                sid=f"{prefix}-{index}",
                lines=lines,
                has_anomaly=anomaly,
                spans=(PublicSpan(1, 1),) if anomaly and profile is TaskProfile.SPAN_BINARY else (),
                source_group=f"{prefix}-group-{index}",
                source_line_ids=(f"{prefix}-{index}-0", f"{prefix}-{index}-1"),
                template_key=template_signature(lines),
            )
        )
    return tuple(result)


def _config() -> PublicNeuralConfig:
    return PublicNeuralConfig(
        vocab_size=128,
        max_tokens=16,
        emb_dim=8,
        hidden=12,
        layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=4,
        eval_batch_size=4,
        patience=2,
        seed=3,
    )


def test_public_sequence_training_cpu_and_checkpoint_boundary(tmp_path) -> None:
    profile = TaskProfile.SEQUENCE_BINARY
    checkpoint = tmp_path / "public.pt"
    run = train_public_neural(
        profile,
        _samples("train", 8, profile),
        _samples("validation", 4, profile),
        _samples("test", 4, profile),
        _config(),
        device_name="cpu",
        checkpoint_path=checkpoint,
        manifest_sha256="a" * 64,
    )
    assert len(run.test_predictions) == 4
    model = create_public_model(vocab_size=128, emb_dim=8, hidden=12, layers=1, dropout=0.0)
    load_public_checkpoint(checkpoint, model, profile, torch.device("cpu"), manifest_sha256="a" * 64)
    with pytest.raises(PublicProtocolError, match="profile"):
        load_public_checkpoint(checkpoint, model, TaskProfile.SPAN_BINARY, torch.device("cpu"))

    target = _samples("target", 4, profile)
    prediction = predict_public_neural(
        profile,
        target,
        _config(),
        checkpoint_path=checkpoint,
        source_manifest_sha256="a" * 64,
        threshold=run.threshold,
        temperature=run.temperature,
        device_name="cpu",
    )
    assert len(prediction) == len(target)


def test_public_span_training_cpu() -> None:
    profile = TaskProfile.SPAN_BINARY
    run = train_public_neural(
        profile,
        _samples("train", 8, profile),
        _samples("validation", 4, profile),
        _samples("test", 4, profile),
        _config(),
        device_name="cpu",
    )
    assert len(run.validation_predictions) == 4
    assert run.profile == "span_binary"


def test_global_class_weights_are_fitted_from_training_samples_only() -> None:
    profile = TaskProfile.SEQUENCE_BINARY
    samples = _samples("train", 8, profile)
    weights = _global_class_weights(build_public_model_samples(samples, profile, vocab_size=128, max_tokens=16))
    assert weights.shape == (2,)
    assert float(weights[0]) == float(weights[1])
