"""Fair, train-only baseline runners for public binary log tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .features import clean_log_line
from .public_metrics import choose_f1_threshold
from .public_protocol import (
    PreparedSample,
    PublicPrediction,
    PublicProtocolError,
    TaskProfile,
    mask_from_spans,
    spans_from_mask,
)

BASELINE_NAMES = ("rarity", "tfidf_logistic", "decision_tree", "isolation_forest")


@dataclass(frozen=True)
class BaselineRun:
    name: str
    profile: str
    threshold: float
    validation_predictions: tuple[PublicPrediction, ...]
    test_predictions: tuple[PublicPrediction, ...]
    metadata: dict[str, float | int | str]


def _require_binary_labels(labels: np.ndarray, description: str) -> None:
    if set(labels.tolist()) != {0, 1}:
        raise PublicProtocolError(f"{description} requires both normal and anomalous training labels")


def _line_features(lines: Iterable[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for line in lines:
        raw = str(line)
        cleaned = clean_log_line(raw)
        words = cleaned.split()
        rows.append(
            [
                float(len(raw)),
                float(len(words)),
                float(sum(character.isdigit() for character in raw)),
                float(sum(character.isupper() for character in raw)),
                float(raw.count("=")),
                float(raw.count(":")),
            ]
        )
    return np.asarray(rows, dtype=float)


def _sequence_features(samples: Iterable[PreparedSample]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in samples:
        line = _line_features(sample.lines)
        rows.append(
            np.asarray(
                [
                    float(len(sample.lines)),
                    *line.mean(axis=0).tolist(),
                    float(len({clean_log_line(item) for item in sample.lines})),
                ],
                dtype=float,
            )
        )
    return np.vstack(rows)


def _sequence_text(samples: Iterable[PreparedSample]) -> list[str]:
    return ["\n".join(clean_log_line(line) for line in sample.lines) for sample in samples]


def _line_dataset(samples: Iterable[PreparedSample]) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    texts: list[str] = []
    labels: list[int] = []
    index: list[tuple[int, int]] = []
    for sample_index, sample in enumerate(samples):
        line_labels = mask_from_spans(len(sample.lines), sample.spans)
        for line_index, (line, label) in enumerate(zip(sample.lines, line_labels)):
            texts.append(clean_log_line(line))
            labels.append(label)
            index.append((sample_index, line_index))
    return texts, np.asarray(labels, dtype=int), index


def _rarity_scores(train_items: Iterable[str], target_items: Iterable[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    total = 0
    for item in train_items:
        counts[item] = counts.get(item, 0) + 1
        total += 1
    if total == 0:
        raise PublicProtocolError("rarity baseline cannot fit an empty training set")
    return np.asarray([(total + len(counts)) / (counts.get(item, 0) + 1) for item in target_items])


def _classifier_scores(
    name: str,
    train_text: list[str],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    target_text: list[str],
    target_features: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if name == "tfidf_logistic":
        _require_binary_labels(train_labels, "TF-IDF logistic baseline")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=25_000)
        matrix = vectorizer.fit_transform(train_text)
        model = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed)
        model.fit(matrix, train_labels)
        return model.predict_proba(vectorizer.transform(target_text))[:, 1], {
            "feature_count": int(len(vectorizer.vocabulary_)),
        }
    if name == "decision_tree":
        _require_binary_labels(train_labels, "decision-tree baseline")
        model = DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=2, max_depth=12, random_state=seed
        )
        model.fit(train_features, train_labels)
        return model.predict_proba(target_features)[:, 1], {"feature_count": int(train_features.shape[1])}
    if name == "isolation_forest":
        model = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
        model.fit(train_features)
        return -model.score_samples(target_features), {"feature_count": int(train_features.shape[1])}
    raise PublicProtocolError(f"unknown baseline: {name}")


def _sequence_scores(
    name: str,
    train: tuple[PreparedSample, ...],
    target: tuple[PreparedSample, ...],
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if name == "rarity":
        return _rarity_scores(
            (sample.template_key for sample in train),
            (sample.template_key for sample in target),
        ), {"feature_count": 1}
    return _classifier_scores(
        name,
        _sequence_text(train),
        _sequence_features(train),
        np.asarray([sample.has_anomaly for sample in train], dtype=int),
        _sequence_text(target),
        _sequence_features(target),
        seed,
    )


def _span_scores(
    name: str,
    train: tuple[PreparedSample, ...],
    target: tuple[PreparedSample, ...],
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, float | int | str]]:
    train_text, train_labels, _ = _line_dataset(train)
    target_text, _target_labels, target_index = _line_dataset(target)
    if name == "rarity":
        scores = _rarity_scores(train_text, target_text)
        metadata: dict[str, float | int | str] = {"feature_count": 1}
    else:
        scores, metadata = _classifier_scores(
            name,
            train_text,
            _line_features(train_text),
            train_labels,
            target_text,
            _line_features(target_text),
            seed,
        )
    return scores, target_index, metadata


def _span_predictions(
    samples: tuple[PreparedSample, ...],
    scores: np.ndarray,
    index: list[tuple[int, int]],
    threshold: float,
) -> tuple[PublicPrediction, ...]:
    per_sample: list[list[float]] = [[] for _ in samples]
    for score, (sample_index, _line_index) in zip(scores, index):
        per_sample[sample_index].append(float(score))
    predictions: list[PublicPrediction] = []
    for sample, line_scores in zip(samples, per_sample):
        if len(line_scores) != len(sample.lines):
            raise RuntimeError("span baseline scores are not aligned with prepared lines")
        spans = spans_from_mask(score >= threshold for score in line_scores)
        predictions.append(
            PublicPrediction(
                sid=sample.sid,
                score=float(max(line_scores)),
                has_anomaly=int(bool(spans)),
                spans=spans,
            )
        )
    return tuple(predictions)


def run_baseline(
    name: str,
    profile: TaskProfile,
    train_samples: Iterable[PreparedSample],
    validation_samples: Iterable[PreparedSample],
    test_samples: Iterable[PreparedSample],
    *,
    seed: int = 20260711,
) -> BaselineRun:
    """Fit one baseline and select its sole threshold from validation labels."""

    if name not in BASELINE_NAMES:
        raise PublicProtocolError(f"unknown baseline {name}; expected one of {BASELINE_NAMES}")
    train = tuple(train_samples)
    validation = tuple(validation_samples)
    test = tuple(test_samples)
    if not train or not validation or not test:
        raise PublicProtocolError("baseline train, validation, and test samples must all be non-empty")
    if profile is TaskProfile.SEQUENCE_BINARY:
        validation_scores, metadata = _sequence_scores(name, train, validation, seed)
        threshold = choose_f1_threshold(validation_scores, (sample.has_anomaly for sample in validation))
        test_scores, _ = _sequence_scores(name, train, test, seed)
        validation_predictions = tuple(
            PublicPrediction(sample.sid, float(score), int(score >= threshold))
            for sample, score in zip(validation, validation_scores)
        )
        test_predictions = tuple(
            PublicPrediction(sample.sid, float(score), int(score >= threshold))
            for sample, score in zip(test, test_scores)
        )
    elif profile is TaskProfile.SPAN_BINARY:
        validation_scores, validation_index, metadata = _span_scores(name, train, validation, seed)
        validation_text, validation_labels, _ = _line_dataset(validation)
        if len(validation_text) != len(validation_scores):
            raise RuntimeError("span baseline validation score alignment failed")
        threshold = choose_f1_threshold(validation_scores, validation_labels)
        test_scores, test_index, _ = _span_scores(name, train, test, seed)
        validation_predictions = _span_predictions(
            validation, validation_scores, validation_index, threshold
        )
        test_predictions = _span_predictions(test, test_scores, test_index, threshold)
    else:
        raise PublicProtocolError(f"unsupported baseline profile {profile}")
    return BaselineRun(
        name=name,
        profile=profile.value,
        threshold=float(threshold),
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        metadata={**metadata, "seed": seed},
    )
