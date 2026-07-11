"""Leakage-safe split assignment for prepared public benchmark samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from .public_protocol import PreparedSample, PublicProtocolError


@dataclass(frozen=True)
class PublicSplitAssignment:
    strategy: str
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def validate(self, samples: Iterable[PreparedSample]) -> None:
        items = tuple(samples)
        known = {sample.sid for sample in items}
        partitions = (set(self.train_ids), set(self.validation_ids), set(self.test_ids))
        if any(not part for part in partitions):
            raise PublicProtocolError("every split partition must contain at least one sample")
        if set.union(*partitions) != known:
            raise PublicProtocolError("split assignment does not cover exactly the prepared samples")
        if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
            raise PublicProtocolError("split assignment partitions overlap")
        membership = {
            sid: split
            for split, ids in zip(("train", "validation", "test"), partitions)
            for sid in ids
        }
        source_owner: dict[str, str] = {}
        group_owner: dict[str, str] = {}
        for sample in items:
            split = membership[sample.sid]
            previous_group = group_owner.setdefault(sample.source_group, split)
            if previous_group != split:
                raise PublicProtocolError("a source group appears in more than one split")
            for source_line in sample.source_line_ids:
                previous = source_owner.setdefault(source_line, split)
                if previous != split:
                    raise PublicProtocolError("a source log line appears in more than one split")


def _validate_sizes(train_fraction: float, validation_fraction: float) -> None:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise PublicProtocolError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise PublicProtocolError("train_fraction + validation_fraction must be less than one")


def _require_binary_support(ids: np.ndarray, labels: np.ndarray, name: str) -> None:
    values = set(labels[ids].tolist())
    if values != {0, 1}:
        raise PublicProtocolError(
            f"{name} split requires both normal and anomalous samples; found {sorted(values)}"
        )


def _finalize(
    strategy: str,
    seed: int,
    samples: tuple[PreparedSample, ...],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> PublicSplitAssignment:
    labels = np.asarray([sample.has_anomaly for sample in samples], dtype=int)
    _require_binary_support(train, labels, "train")
    _require_binary_support(validation, labels, "validation")
    _require_binary_support(test, labels, "test")
    assignment = PublicSplitAssignment(
        strategy=strategy,
        seed=seed,
        train_ids=tuple(samples[index].sid for index in train),
        validation_ids=tuple(samples[index].sid for index in validation),
        test_ids=tuple(samples[index].sid for index in test),
    )
    assignment.validate(samples)
    return assignment


def random_split(
    samples: Iterable[PreparedSample],
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> PublicSplitAssignment:
    items = tuple(samples)
    _validate_sizes(train_fraction, validation_fraction)
    labels = np.asarray([sample.has_anomaly for sample in items], dtype=int)
    if len(items) < 10:
        raise PublicProtocolError("random split requires at least ten samples")
    holdout_fraction = 1.0 - train_fraction
    try:
        first = StratifiedShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
        train, holdout = next(first.split(np.arange(len(items)), labels))
        second = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1.0 - train_fraction - validation_fraction) / holdout_fraction,
            random_state=seed + 1,
        )
        validation_relative, test_relative = next(second.split(holdout, labels[holdout]))
    except ValueError as exc:
        raise PublicProtocolError(f"cannot build stratified random split: {exc}") from exc
    return _finalize("random", seed, items, train, holdout[validation_relative], holdout[test_relative])


def chronological_split(
    samples: Iterable[PreparedSample],
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> PublicSplitAssignment:
    items = tuple(samples)
    _validate_sizes(train_fraction, validation_fraction)
    if any(sample.timestamp is None for sample in items):
        raise PublicProtocolError("chronological split requires a timestamp on every prepared sample")
    ordered = np.asarray(sorted(range(len(items)), key=lambda index: (items[index].timestamp, items[index].sid)))
    train_end = int(len(items) * train_fraction)
    validation_end = train_end + int(len(items) * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end >= len(items):
        raise PublicProtocolError("chronological split fractions leave an empty partition")
    return _finalize(
        "chronological",
        seed,
        items,
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


def template_isolated_split(
    samples: Iterable[PreparedSample],
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> PublicSplitAssignment:
    items = tuple(samples)
    _validate_sizes(train_fraction, validation_fraction)
    labels = np.asarray([sample.has_anomaly for sample in items], dtype=int)
    groups = np.asarray([sample.template_key for sample in items])
    if len(set(groups)) < 3:
        raise PublicProtocolError("template-isolated split requires at least three template groups")
    first = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_fraction, random_state=seed)
    train, holdout = next(first.split(np.arange(len(items)), labels, groups))
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=(1.0 - train_fraction - validation_fraction) / (1.0 - train_fraction),
        random_state=seed + 1,
    )
    validation_relative, test_relative = next(
        second.split(holdout, labels[holdout], groups[holdout])
    )
    return _finalize(
        "template_isolated",
        seed,
        items,
        train,
        holdout[validation_relative],
        holdout[test_relative],
    )


def write_split_assignment(path: Path, assignment: PublicSplitAssignment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(assignment), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_split_assignment(path: Path, samples: Iterable[PreparedSample]) -> PublicSplitAssignment:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assignment = PublicSplitAssignment(
            strategy=str(payload["strategy"]),
            seed=int(payload["seed"]),
            train_ids=tuple(str(item) for item in payload["train_ids"]),
            validation_ids=tuple(str(item) for item in payload["validation_ids"]),
            test_ids=tuple(str(item) for item in payload["test_ids"]),
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicProtocolError(f"invalid split assignment: {path}") from exc
    assignment.validate(samples)
    return assignment


def partition_samples(
    samples: Iterable[PreparedSample], assignment: PublicSplitAssignment
) -> tuple[tuple[PreparedSample, ...], tuple[PreparedSample, ...], tuple[PreparedSample, ...]]:
    """Return samples in their stored preparation order for one validated assignment."""

    items = tuple(samples)
    assignment.validate(items)
    lookup = {sample.sid: sample for sample in items}
    return (
        tuple(lookup[sid] for sid in assignment.train_ids),
        tuple(lookup[sid] for sid in assignment.validation_ids),
        tuple(lookup[sid] for sid in assignment.test_ids),
    )
