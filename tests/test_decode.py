import numpy as np

from seclog.decode import (
    constrained_viterbi,
    decode_single_item,
    spans_from_bio_path,
    stable_softmax,
)


def test_viterbi_disallows_inside_tag_at_first_line() -> None:
    logits = np.full((2, 21), -5.0, dtype=np.float32)
    logits[0, 11] = 10
    logits[0, 1] = 9
    logits[1, 11] = 10
    path = constrained_viterbi(logits, mask_len=2)
    assert path == [1, 11]


def test_spans_from_bio_path_returns_inclusive_span() -> None:
    probabilities = stable_softmax(np.eye(21, dtype=np.float32)[[0, 1, 11, 0]])
    spans = spans_from_bio_path([0, 1, 11, 0], probabilities)
    assert spans[0][:3] == (1, 2, "timeout_retry")


def test_softmax_rows_sum_to_one() -> None:
    result = stable_softmax(np.array([[1000.0, 1001.0]], dtype=np.float32))
    assert np.allclose(result.sum(axis=1), 1.0)


def prediction_with_tags(tag_ids: list[int]) -> dict[str, np.ndarray]:
    tag = np.full((len(tag_ids), 21), -10.0, dtype=np.float32)
    for index, tag_id in enumerate(tag_ids):
        tag[index, tag_id] = 10.0
    global_logits = np.full(11, -10.0, dtype=np.float32)
    global_logits[0] = 10.0
    return {
        "tag": tag,
        "start": np.zeros((len(tag_ids), 10), dtype=np.float32),
        "end": np.zeros((len(tag_ids), 10), dtype=np.float32),
        "global": global_logits,
    }


def decoder_params() -> dict[str, object]:
    return {
        "min_conf": 0.0,
        "bridge_gap": 0,
        "refine_radius": 0,
        "use_boundary_candidates": False,
        "fallback_global_boundary": False,
        "post_adjust_mode": "none",
    }


def test_decode_single_item_returns_normal_sentinels() -> None:
    decoded = decode_single_item(
        prediction_with_tags([0, 0, 0]),
        n_lines=3,
        params=decoder_params(),
        length_stats={},
    )
    assert decoded == {
        "has_anomaly": 0,
        "primary_start_idx": -1,
        "primary_end_idx": -1,
        "primary_anomaly_type": "none",
        "all_spans": "",
    }


def test_decode_single_item_returns_inclusive_timeout_span() -> None:
    decoded = decode_single_item(
        prediction_with_tags([1, 11, 0]),
        n_lines=3,
        params=decoder_params(),
        length_stats={"timeout_retry": {"p95": 3}},
    )
    assert decoded == {
        "has_anomaly": 1,
        "primary_start_idx": 0,
        "primary_end_idx": 1,
        "primary_anomaly_type": "timeout_retry",
        "all_spans": "0|1|timeout_retry",
    }
