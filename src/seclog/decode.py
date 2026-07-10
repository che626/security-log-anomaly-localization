import math
from typing import Any

import numpy as np
import pandas as pd

from .constants import ANOMALY_TYPES, N_LABELS, N_TYPES, SUBMISSION_COLUMNS, TYPE_TO_ID
from .data import Sample

DecodedSpan = tuple[int, int, str, float]


def tag_to_type(label_id: int) -> str | None:
    if label_id == 0:
        return None
    if 1 <= label_id <= N_TYPES:
        return ANOMALY_TYPES[label_id - 1]
    if 1 + N_TYPES <= label_id < N_LABELS:
        return ANOMALY_TYPES[label_id - 1 - N_TYPES]
    return None


def build_bio_transition(
    anomaly_bias: float = 0.0,
    switch_penalty: float = -1.8,
    continue_bonus: float = 0.20,
) -> np.ndarray:
    transition = np.zeros((N_LABELS, N_LABELS), dtype=np.float32)
    for previous in range(N_LABELS):
        for current in range(N_LABELS):
            current_type = tag_to_type(current)
            previous_type = tag_to_type(previous)
            if current >= 1 + N_TYPES:
                current_type_id = current - 1 - N_TYPES
                if previous == 0 or previous_type != ANOMALY_TYPES[current_type_id]:
                    transition[previous, current] = -1e4
                else:
                    transition[previous, current] += continue_bonus
            if (
                previous_type is not None
                and current_type is not None
                and previous_type != current_type
            ):
                transition[previous, current] += switch_penalty
            if current_type is not None:
                transition[previous, current] += anomaly_bias
    return transition


def constrained_viterbi(
    logits: np.ndarray,
    mask_len: int,
    anomaly_bias: float = 0.0,
    switch_penalty: float = -1.8,
    continue_bonus: float = 0.20,
) -> list[int]:
    scores = logits[:mask_len].astype(np.float32)
    normalized = scores - scores.max(axis=1, keepdims=True)
    scores = normalized - np.log(np.exp(normalized).sum(axis=1, keepdims=True))
    transition = build_bio_transition(anomaly_bias, switch_penalty, continue_bonus)
    dynamic = np.full((mask_len, N_LABELS), -1e9, dtype=np.float32)
    back = np.zeros((mask_len, N_LABELS), dtype=np.int16)
    dynamic[0] = scores[0]
    dynamic[0, 1 + N_TYPES :] = -1e9
    for index in range(1, mask_len):
        values = dynamic[index - 1][:, None] + transition
        back[index] = values.argmax(axis=0)
        dynamic[index] = values.max(axis=0) + scores[index]
    path = [int(dynamic[-1].argmax())]
    for index in range(mask_len - 1, 0, -1):
        path.append(int(back[index, path[-1]]))
    path.reverse()
    return path


def stable_softmax(values: np.ndarray) -> np.ndarray:
    normalized = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(normalized)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def bounded_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def spans_from_bio_path(
    tags: list[int],
    probs: np.ndarray | None = None,
    min_conf: float = 0.0,
    bridge_gap: int = 0,
    edge_prob: float = 0.10,
) -> list[DecodedSpan]:
    raw: list[DecodedSpan] = []
    index, length = 0, len(tags)
    while index < length:
        anomaly_type = tag_to_type(tags[index])
        if anomaly_type is None:
            index += 1
            continue
        start = index
        values: list[float] = []
        while index < length and tag_to_type(tags[index]) == anomaly_type:
            if probs is not None:
                type_id = TYPE_TO_ID[anomaly_type]
                values.append(
                    float(max(probs[index, 1 + type_id], probs[index, 1 + N_TYPES + type_id]))
                )
            index += 1
        end = index - 1
        confidence = float(np.mean(values)) if values else 1.0
        if confidence >= min_conf:
            raw.append((start, end, anomaly_type, confidence))
    if bridge_gap > 0 and probs is not None and len(raw) >= 2:
        merged: list[DecodedSpan] = []
        current = raw[0]
        for following in raw[1:]:
            start, end, anomaly_type, confidence = current
            next_start, next_end, next_type, next_confidence = following
            gap = next_start - end - 1
            if anomaly_type == next_type and 0 < gap <= bridge_gap:
                type_id = TYPE_TO_ID[anomaly_type]
                gap_probability = np.mean(
                    [
                        max(probs[item, 1 + type_id], probs[item, 1 + N_TYPES + type_id])
                        for item in range(end + 1, next_start)
                    ]
                )
                if gap_probability >= edge_prob:
                    current = (
                        start,
                        next_end,
                        anomaly_type,
                        float((confidence + next_confidence) / 2),
                    )
                    continue
            merged.append(current)
            current = following
        merged.append(current)
        raw = merged
    return raw


def add_endpoint_candidates(
    spans: list[DecodedSpan],
    tag_probs: np.ndarray,
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    global_probs: np.ndarray,
    params: dict[str, Any],
) -> list[DecodedSpan]:
    if not params.get("use_boundary_candidates", False):
        return spans
    n_lines = tag_probs.shape[0]
    output = list(spans)
    start_threshold = float(params.get("boundary_start_th", 0.45))
    end_threshold = float(params.get("boundary_end_th", 0.45))
    max_length = int(params.get("candidate_max_len", 12))
    min_support = float(params.get("candidate_tag_support", 0.12))
    for type_id, anomaly_type in enumerate(ANOMALY_TYPES):
        type_global = float(global_probs[type_id])
        if type_global < float(params.get("candidate_global_min", 0.03)):
            continue
        starts = np.where(start_probs[:, type_id] >= start_threshold)[0]
        for start in starts[:8]:
            low, high = start, min(n_lines - 1, start + max_length - 1)
            if high < low:
                continue
            end_region = end_probs[low : high + 1, type_id]
            if len(end_region) == 0:
                continue
            end = int(low + end_region.argmax())
            if end_probs[end, type_id] < end_threshold:
                continue
            support = float(
                np.mean(
                    np.maximum(
                        tag_probs[start : end + 1, 1 + type_id],
                        tag_probs[start : end + 1, 1 + N_TYPES + type_id],
                    )
                )
            )
            if support < min_support:
                continue
            confidence = (
                0.45 * support
                + 0.25 * float(start_probs[start, type_id])
                + 0.25 * float(end_probs[end, type_id])
                + 0.05 * type_global
            )
            output.append((int(start), int(end), anomaly_type, float(confidence)))
    output.sort(key=lambda item: (item[0], item[1], item[2], -item[3]))
    deduplicated: list[DecodedSpan] = []
    for candidate in output:
        start, end, anomaly_type, confidence = candidate
        replaced = False
        for index, old in enumerate(deduplicated):
            old_start, old_end, old_type, old_confidence = old
            if (
                old_type == anomaly_type
                and max(0, min(end, old_end) - max(start, old_start) + 1) > 0
            ):
                if confidence > old_confidence:
                    deduplicated[index] = candidate
                replaced = True
                break
        if not replaced:
            deduplicated.append(candidate)
    return deduplicated


def polish_span_boundary(
    span: DecodedSpan,
    tag_probs: np.ndarray,
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    global_probs: np.ndarray,
    length_stats: dict[str, dict[str, float]],
    params: dict[str, Any],
) -> DecodedSpan:
    start, end, anomaly_type, confidence = span
    type_id = TYPE_TO_ID[anomaly_type]
    n_lines = tag_probs.shape[0]
    radius = int(params.get("refine_radius", 1))
    if radius <= 0:
        return span
    stats = length_stats.get(anomaly_type, {"p50": 5.0, "p95": 10.0})
    median = float(stats.get("p50", 5.0))
    max_reasonable = int(max(3, min(16, math.ceil(float(stats.get("p95", 10.0)) + 2))))
    best = (start, end, confidence, -1e9)
    for candidate_start in range(max(0, start - radius), min(n_lines - 1, start + radius) + 1):
        for candidate_end in range(
            max(candidate_start, end - radius), min(n_lines - 1, end + radius) + 1
        ):
            length = candidate_end - candidate_start + 1
            if length > max_reasonable:
                continue
            support = float(
                np.mean(
                    np.maximum(
                        tag_probs[candidate_start : candidate_end + 1, 1 + type_id],
                        tag_probs[candidate_start : candidate_end + 1, 1 + N_TYPES + type_id],
                    )
                )
            )
            boundary = float(
                start_probs[candidate_start, type_id] + end_probs[candidate_end, type_id]
            )
            length_penalty = -abs(length - median) / max(3.0, median)
            score = (
                support
                + float(params.get("boundary_weight", 0.35)) * boundary
                + float(params.get("global_weight", 0.10)) * float(global_probs[type_id])
                + float(params.get("length_weight", 0.08)) * length_penalty
            )
            if score > best[3]:
                best = (candidate_start, candidate_end, support, score)
    return int(best[0]), int(best[1]), anomaly_type, float(max(confidence, best[2]))


def pick_primary_span(
    spans: list[DecodedSpan],
    global_probs: np.ndarray | None = None,
    strategy: str = "earliest",
) -> DecodedSpan | None:
    if not spans:
        return None
    if strategy == "longest":
        return max(spans, key=lambda item: (item[1] - item[0] + 1, item[3], -item[0]))
    if strategy == "score":
        if global_probs is None:
            return max(spans, key=lambda item: (item[3], item[1] - item[0] + 1, -item[0]))
        return max(
            spans,
            key=lambda item: (
                item[3] + 0.15 * global_probs[TYPE_TO_ID[item[2]]],
                item[1] - item[0] + 1,
                -item[0],
            ),
        )
    if strategy == "earliest_score":
        return min(spans, key=lambda item: (item[0] // 3, -item[3], -(item[1] - item[0] + 1)))
    return min(spans, key=lambda item: (item[0], -(item[1] - item[0] + 1), -item[3]))


def parse_type_filter(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"all", "*", "none"}:
        return None
    return {item.strip() for item in text.split(",") if item.strip() in TYPE_TO_ID}


def maybe_adjust_primary_span(
    primary: DecodedSpan,
    n_lines: int,
    params: dict[str, Any],
    tag_probs: np.ndarray,
    global_probs: np.ndarray,
) -> DecodedSpan:
    mode = str(params.get("post_adjust_mode", "none")).lower()
    if mode in {"none", "", "false", "0"}:
        return primary
    start, end, anomaly_type, confidence = primary
    if anomaly_type not in TYPE_TO_ID or n_lines <= 0:
        return primary
    type_filter = parse_type_filter(params.get("post_tweak_types", "all"))
    if type_filter is not None and anomaly_type not in type_filter:
        return primary
    type_id = TYPE_TO_ID[anomaly_type]
    length = int(end - start + 1)
    output_start, output_end = int(start), int(end)

    def neighbor_ok(index: int, threshold: float) -> bool:
        if index < 0 or index >= n_lines:
            return False
        support = float(
            max(tag_probs[index, 1 + type_id], tag_probs[index, 1 + N_TYPES + type_id])
        )
        return support >= threshold

    if mode in {"short_expand", "both"}:
        short_max_length = int(params.get("post_short_max_len", 2))
        max_confidence = float(params.get("post_short_max_conf", 1.01))
        min_global = float(params.get("post_global_min", 0.0))
        neighbor_threshold = float(params.get("post_neighbor_support", 0.0))
        if (
            length <= short_max_length
            and confidence <= max_confidence
            and float(global_probs[type_id]) >= min_global
        ):
            for _ in range(max(0, int(params.get("post_expand_left", 0)))):
                if output_start > 0 and neighbor_ok(output_start - 1, neighbor_threshold):
                    output_start -= 1
            for _ in range(max(0, int(params.get("post_expand_right", 0)))):
                if output_end + 1 < n_lines and neighbor_ok(output_end + 1, neighbor_threshold):
                    output_end += 1
    if mode in {"long_shrink", "both"}:
        long_min_length = int(params.get("post_long_min_len", 9))
        max_confidence = float(params.get("post_long_max_conf", 1.01))
        if output_end - output_start + 1 >= long_min_length and confidence <= max_confidence:
            output_start = min(
                output_end, output_start + max(0, int(params.get("post_shrink_left", 0)))
            )
            output_end = max(
                output_start, output_end - max(0, int(params.get("post_shrink_right", 0)))
            )
    return (
        int(max(0, min(output_start, n_lines - 1))),
        int(max(0, min(output_end, n_lines - 1))),
        anomaly_type,
        float(confidence),
    )


def decode_single_item(
    pred: dict[str, np.ndarray],
    n_lines: int,
    params: dict[str, Any],
    length_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    tag_logits = pred["tag"][:n_lines]
    tag_probs = stable_softmax(tag_logits)
    start_probs = bounded_sigmoid(pred["start"][:n_lines])
    end_probs = bounded_sigmoid(pred["end"][:n_lines])
    global_probs = stable_softmax(pred["global"][None, :])[0]
    tags = constrained_viterbi(
        tag_logits,
        n_lines,
        anomaly_bias=float(params.get("anomaly_bias", 0.0)),
        switch_penalty=float(params.get("switch_penalty", -1.8)),
        continue_bonus=float(params.get("continue_bonus", 0.20)),
    )
    spans = spans_from_bio_path(
        tags,
        probs=tag_probs,
        min_conf=float(params.get("min_conf", 0.0)),
        bridge_gap=int(params.get("bridge_gap", 0)),
        edge_prob=float(params.get("edge_prob", 0.10)),
    )
    spans = add_endpoint_candidates(
        spans, tag_probs, start_probs, end_probs, global_probs, params
    )
    filtered: list[DecodedSpan] = []
    for span in spans:
        start, end, anomaly_type, confidence = span
        length = end - start + 1
        if length < int(params.get("min_len", 1)):
            continue
        stats = length_stats.get(anomaly_type, {"p95": 10})
        hard_max = int(params.get("hard_max_len", 18))
        type_max = int(max(4, math.ceil(float(stats.get("p95", 10)) + 4)))
        if length > min(hard_max, type_max) and confidence < float(
            params.get("long_span_conf", 0.60)
        ):
            continue
        if params.get("refine_radius", 1) > 0:
            span = polish_span_boundary(
                span,
                tag_probs,
                start_probs,
                end_probs,
                global_probs,
                length_stats,
                params,
            )
        filtered.append(span)
    primary = pick_primary_span(
        filtered, global_probs, strategy=str(params.get("primary_strategy", "earliest"))
    )
    if primary is None and params.get("fallback_global_boundary", False):
        type_id = int(np.argmax(global_probs[:N_TYPES]))
        if global_probs[type_id] >= float(params.get("fallback_global_th", 0.80)):
            start = int(np.argmax(start_probs[:, type_id]))
            high = min(
                n_lines - 1, start + int(params.get("candidate_max_len", 12)) - 1
            )
            end = int(start + np.argmax(end_probs[start : high + 1, type_id]))
            if (
                start_probs[start, type_id] >= float(params.get("fallback_start_th", 0.35))
                and end_probs[end, type_id] >= float(params.get("fallback_end_th", 0.35))
            ):
                primary = (start, end, ANOMALY_TYPES[type_id], float(global_probs[type_id]))
    if primary is None:
        return {
            "has_anomaly": 0,
            "primary_start_idx": -1,
            "primary_end_idx": -1,
            "primary_anomaly_type": "none",
            "all_spans": "",
        }
    start, end, anomaly_type, _confidence = maybe_adjust_primary_span(
        primary, n_lines, params, tag_probs, global_probs
    )
    return {
        "has_anomaly": 1,
        "primary_start_idx": int(start),
        "primary_end_idx": int(end),
        "primary_anomaly_type": anomaly_type,
        "all_spans": f"{int(start)}|{int(end)}|{anomaly_type}",
    }


def decode_logits_to_frame(
    samples: list[Sample],
    pred_list: list[dict[str, np.ndarray]],
    params: dict[str, Any],
    length_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample, prediction in zip(samples, pred_list):
        decoded = decode_single_item(prediction, len(sample.lines), params, length_stats)
        decoded["id"] = sample.sid
        rows.append(decoded)
    return pd.DataFrame(rows)[list(SUBMISSION_COLUMNS)]


def samples_to_truth_frame(samples: list[Sample]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.has_anomaly == 0:
            rows.append(
                {
                    "id": sample.sid,
                    "has_anomaly": 0,
                    "primary_start_idx": -1,
                    "primary_end_idx": -1,
                    "primary_anomaly_type": "none",
                    "all_spans": "",
                }
            )
        else:
            if sample.primary is None:
                raise ValueError(f"anomalous sample {sample.sid} is missing its primary span")
            start, end, anomaly_type = sample.primary
            rows.append(
                {
                    "id": sample.sid,
                    "has_anomaly": 1,
                    "primary_start_idx": start,
                    "primary_end_idx": end,
                    "primary_anomaly_type": anomaly_type,
                    "all_spans": f"{start}|{end}|{anomaly_type}",
                }
            )
    return pd.DataFrame(rows)
