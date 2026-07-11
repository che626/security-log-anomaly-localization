import json

import pytest

from seclog.public_protocol import (
    PreparedManifest,
    PreparedSample,
    PublicProtocolError,
    PublicSpan,
    TaskProfile,
    mask_from_spans,
    read_prepared_dataset,
    spans_from_mask,
    template_signature,
    write_manifest,
    write_prepared_dataset,
)


def _sample(sid: str = "one", anomaly: int = 1) -> PreparedSample:
    spans = (PublicSpan(1, 1),) if anomaly else ()
    return PreparedSample(
        sid=sid,
        lines=("2026-01-01 service started", "2026-01-01 timeout happened"),
        has_anomaly=anomaly,
        spans=spans,
        source_group=f"group-{sid}",
        source_line_ids=(f"{sid}-0", f"{sid}-1"),
        template_key=template_signature(("service started", "timeout happened")),
        timestamp="2026-01-01T00:00:00",
    )


def test_span_masks_round_trip() -> None:
    spans = spans_from_mask([0, 1, 1, 0, 1])
    assert spans == (PublicSpan(1, 2), PublicSpan(4, 4))
    assert mask_from_spans(5, spans) == [0, 1, 1, 0, 1]


def test_span_profile_requires_span_for_anomaly() -> None:
    invalid = _sample()
    invalid = PreparedSample(**{**invalid.__dict__, "spans": ()})
    with pytest.raises(PublicProtocolError, match="requires a span"):
        invalid.validate(TaskProfile.SPAN_BINARY)


def test_prepared_dataset_is_deterministic_and_validated(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    written = write_prepared_dataset(path, (_sample(), _sample("two", 0)), TaskProfile.SPAN_BINARY)
    assert len(written) == 2
    assert read_prepared_dataset(path, TaskProfile.SPAN_BINARY) == written
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["sid"] == "one"


def test_manifest_rejects_invalid_hash(tmp_path) -> None:
    manifest = PreparedManifest(
        schema_version=1,
        dataset="fixture",
        profile="sequence_binary",
        source_sha256={"log": "not-a-hash"},
        source_metadata={"source": "test"},
        preparation={},
        sample_count=2,
        anomalous_sample_count=1,
        source_line_count=4,
    )
    with pytest.raises(PublicProtocolError, match="SHA256"):
        write_manifest(tmp_path / "manifest.json", manifest)
