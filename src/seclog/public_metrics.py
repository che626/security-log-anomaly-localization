"""Metrics and validation-only calibration for binary public log tasks."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .public_protocol import (
    PreparedSample,
    PublicPrediction,
    PublicProtocolError,
    PublicSpan,
    TaskProfile,
    mask_from_spans,
)


def _safe_auc(labels: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    if len(set(labels.tolist())) != 2:
        return None
    if metric == "roc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def choose_f1_threshold(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Choose a deterministic threshold from validation scores only."""

    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=int)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise PublicProtocolError("threshold scores and labels must be non-empty and aligned")
    if not np.isfinite(score_array).all() or not set(label_array.tolist()).issubset({0, 1}):
        raise PublicProtocolError("threshold inputs must be finite binary-labelled observations")
    if set(label_array.tolist()) != {0, 1}:
        raise PublicProtocolError("validation threshold selection requires both binary classes")
    candidates = np.unique(score_array)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = f1_score(label_array, score_array >= threshold, zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def fit_temperature(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Fit a finite scalar temperature by validation negative log likelihood.

    Scores are probabilities in ``(0, 1)``. A small deterministic logarithmic
    grid avoids adding a numerical optimisation dependency to this portfolio.
    """

    probability = np.asarray(list(scores), dtype=float)
    target = np.asarray(list(labels), dtype=int)
    if len(probability) == 0 or len(probability) != len(target):
        raise PublicProtocolError("calibration scores and labels must be non-empty and aligned")
    if set(target.tolist()) != {0, 1}:
        raise PublicProtocolError("temperature calibration requires both binary classes")
    if not np.isfinite(probability).all() or (probability <= 0).any() or (probability >= 1).any():
        raise PublicProtocolError("temperature calibration requires probabilities strictly between zero and one")
    logits = np.log(probability / (1.0 - probability))
    temperatures = np.geomspace(0.05, 10.0, num=81)
    losses: list[float] = []
    for temperature in temperatures:
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60, 60)))
        calibrated = np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        loss = -np.mean(target * np.log(calibrated) + (1 - target) * np.log(1.0 - calibrated))
        losses.append(float(loss))
    return float(temperatures[int(np.argmin(losses))])


def apply_temperature(scores: Iterable[float], temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise PublicProtocolError("temperature must be finite and positive")
    probability = np.asarray(list(scores), dtype=float)
    if not np.isfinite(probability).all() or (probability <= 0).any() or (probability >= 1).any():
        raise PublicProtocolError("temperature scaling requires probabilities strictly between zero and one")
    logits = np.log(probability / (1.0 - probability))
    return np.clip(
        1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60, 60))),
        1e-6,
        1.0 - 1e-6,
    )


def _aligned_predictions(
    samples: Iterable[PreparedSample],
    predictions: Iterable[PublicPrediction],
    profile: TaskProfile,
) -> tuple[tuple[PreparedSample, ...], tuple[PublicPrediction, ...]]:
    sample_items = tuple(samples)
    prediction_items = tuple(predictions)
    if len(sample_items) != len(prediction_items):
        raise PublicProtocolError("prediction count does not match prepared sample count")
    if [sample.sid for sample in sample_items] != [prediction.sid for prediction in prediction_items]:
        raise PublicProtocolError("public predictions must use the prepared sample order")
    for sample, prediction in zip(sample_items, prediction_items):
        prediction.validate(sample, profile)
    return sample_items, prediction_items


def evaluate_sequence_predictions(
    samples: Iterable[PreparedSample], predictions: Iterable[PublicPrediction]
) -> dict[str, float | int | None]:
    sample_items, prediction_items = _aligned_predictions(
        samples, predictions, TaskProfile.SEQUENCE_BINARY
    )
    gold = np.asarray([sample.has_anomaly for sample in sample_items], dtype=int)
    scores = np.asarray([prediction.score for prediction in prediction_items], dtype=float)
    decisions = np.asarray([prediction.has_anomaly for prediction in prediction_items], dtype=int)
    if not np.isfinite(scores).all():
        raise PublicProtocolError("sequence scores must be finite")
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, decisions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(gold, decisions, labels=[0, 1]).ravel()
    return {
        "sample_count": len(sample_items),
        "anomalous_sample_count": int(gold.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": _safe_auc(gold, scores, "pr"),
        "roc_auc": _safe_auc(gold, scores, "roc"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def inclusive_iou(left: PublicSpan, right: PublicSpan) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start) + 1)
    union = max(left.end, right.end) - min(left.start, right.start) + 1
    return float(intersection / union) if union else 0.0


def _span_matches(gold: tuple[PublicSpan, ...], predicted: tuple[PublicSpan, ...]) -> tuple[int, list[float]]:
    available = set(range(len(predicted)))
    matched = 0
    ious: list[float] = []
    for gold_span in gold:
        candidates = [(inclusive_iou(gold_span, predicted[index]), index) for index in available]
        if not candidates:
            continue
        best_iou, best_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_iou > 0:
            matched += 1
            ious.append(best_iou)
            available.remove(best_index)
    return matched, ious


def evaluate_span_predictions(
    samples: Iterable[PreparedSample], predictions: Iterable[PublicPrediction]
) -> dict[str, float | int | None]:
    sample_items, prediction_items = _aligned_predictions(samples, predictions, TaskProfile.SPAN_BINARY)
    gold_lines: list[int] = []
    predicted_lines: list[int] = []
    matched = 0
    gold_span_count = 0
    predicted_span_count = 0
    exact = 0
    ious: list[float] = []
    for sample, prediction in zip(sample_items, prediction_items):
        gold_mask = mask_from_spans(len(sample.lines), sample.spans)
        prediction_mask = mask_from_spans(len(sample.lines), prediction.spans)
        gold_lines.extend(gold_mask)
        predicted_lines.extend(prediction_mask)
        count, sample_ious = _span_matches(sample.spans, prediction.spans)
        matched += count
        ious.extend(sample_ious)
        gold_span_count += len(sample.spans)
        predicted_span_count += len(prediction.spans)
        exact += sum(1 for span in sample.spans if span in prediction.spans)
    line_precision, line_recall, line_f1, _ = precision_recall_fscore_support(
        gold_lines, predicted_lines, average="binary", zero_division=0
    )
    span_precision = matched / predicted_span_count if predicted_span_count else 0.0
    span_recall = matched / gold_span_count if gold_span_count else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if span_precision + span_recall
        else 0.0
    )
    return {
        "sample_count": len(sample_items),
        "line_precision": float(line_precision),
        "line_recall": float(line_recall),
        "line_f1": float(line_f1),
        "span_precision": float(span_precision),
        "span_recall": float(span_recall),
        "span_f1": float(span_f1),
        "mean_inclusive_iou": float(np.mean(ious)) if ious else 0.0,
        "exact_boundary_accuracy": float(exact / gold_span_count) if gold_span_count else None,
        "gold_span_count": gold_span_count,
        "predicted_span_count": predicted_span_count,
    }


def evaluate_normal_only_predictions(
    samples: Iterable[PreparedSample],
    predictions: Iterable[PublicPrediction],
    profile: TaskProfile,
) -> dict[str, float | int]:
    """Measure cross-system false positives when a target corpus has no anomalies.

    A normal-only target cannot yield recall, F1, ROC-AUC, or PR-AUC. Reporting
    those as zero would be misleading, so this function reports only the
    operational false-positive quantities that are defined.
    """

    sample_items, prediction_items = _aligned_predictions(samples, predictions, profile)
    if any(sample.has_anomaly for sample in sample_items):
        raise PublicProtocolError("normal-only evaluation requires every target sample to be normal")
    false_positive_samples = sum(prediction.has_anomaly for prediction in prediction_items)
    result: dict[str, float | int] = {
        "sample_count": len(sample_items),
        "false_positive_samples": false_positive_samples,
        "sample_false_positive_rate": float(false_positive_samples / len(sample_items)),
    }
    if profile is TaskProfile.SPAN_BINARY:
        total_lines = sum(len(sample.lines) for sample in sample_items)
        false_positive_lines = sum(
            sum(mask_from_spans(len(sample.lines), prediction.spans))
            for sample, prediction in zip(sample_items, prediction_items)
        )
        result.update(
            {
                "line_count": total_lines,
                "false_positive_lines": false_positive_lines,
                "line_false_positive_rate": float(false_positive_lines / total_lines),
                "predicted_span_count": sum(len(prediction.spans) for prediction in prediction_items),
            }
        )
    return result
