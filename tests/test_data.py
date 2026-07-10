import torch

from seclog.data import (
    Sample,
    build_endpoint_labels,
    build_sequence_labels,
    pack_log_batch,
    parse_annotation_spans,
)


def test_parse_annotation_spans_deduplicates_and_sorts() -> None:
    spans = parse_annotation_spans(
        "3|4|missing_step;1|2|timeout_retry;1|2|timeout_retry",
        has_anomaly=1,
        primary_start=1,
        primary_end=2,
        primary_type="timeout_retry",
    )
    assert spans == [(1, 2, "timeout_retry"), (3, 4, "missing_step")]


def test_sequence_and_endpoint_labels() -> None:
    spans = [(1, 2, "timeout_retry")]
    sequence = build_sequence_labels(4, spans)
    starts, ends = build_endpoint_labels(4, spans)
    assert sequence.tolist() == [0, 1, 11, 0]
    assert starts[1, 0] == 1
    assert ends[2, 0] == 1


def test_pack_log_batch_shapes() -> None:
    sample = Sample(sid=7, lines=["a", "b"], token_ids=[[1, 2], [3]])
    batch = pack_log_batch([sample])
    assert batch["input_ids"].tolist() == [1, 2, 3]
    assert tuple(batch["mask"].shape) == (1, 2)
    assert tuple(batch["pos_feats"].shape) == (1, 2, 10)
    assert batch["mask"].dtype == torch.bool
