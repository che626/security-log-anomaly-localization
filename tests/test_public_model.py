from seclog.public_model import build_public_model_samples, create_public_model, pack_public_batch
from seclog.public_protocol import PreparedSample, PublicSpan, TaskProfile, template_signature


def _sample(anomaly: int) -> PreparedSample:
    lines = ("service started", "disk failure")
    return PreparedSample(
        sid=f"sample-{anomaly}",
        lines=lines,
        has_anomaly=anomaly,
        spans=(PublicSpan(1, 1),) if anomaly else (),
        source_group=f"group-{anomaly}",
        source_line_ids=(f"{anomaly}-0", f"{anomaly}-1"),
        template_key=template_signature(lines),
    )


def test_public_binary_model_uses_three_tags_and_two_global_classes() -> None:
    samples = build_public_model_samples(
        (_sample(0), _sample(1)), TaskProfile.SPAN_BINARY, vocab_size=128, max_tokens=16
    )
    batch = pack_public_batch(list(samples))
    model = create_public_model(vocab_size=128, emb_dim=8, hidden=12, layers=1, dropout=0.0)
    tags, starts, ends, global_logits = model(
        batch["input_ids"], batch["offsets"], batch["owner"], batch["mask"], batch["pos_feats"]
    )
    assert tuple(tags.shape) == (2, 2, 3)
    assert tuple(starts.shape) == (2, 2, 1)
    assert tuple(ends.shape) == (2, 2, 1)
    assert tuple(global_logits.shape) == (2, 2)
    assert batch["tags"][1].tolist() == [0, 1]
    assert batch["global_labels"].tolist() == [1, 0]
