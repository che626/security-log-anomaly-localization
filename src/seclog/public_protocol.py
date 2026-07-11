"""Data contracts for publicly sourced binary log benchmarks.

The ISCC workflow has a ten-class annotation contract.  Public Loghub datasets
use heterogeneous labels, so they deliberately use this separate binary
protocol instead of pretending that a public anomaly belongs to one of the
competition classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .features import clean_log_line


class PublicProtocolError(ValueError):
    """Raised when public benchmark data violate the documented contract."""


class TaskProfile(str, Enum):
    SEQUENCE_BINARY = "sequence_binary"
    SPAN_BINARY = "span_binary"


@dataclass(frozen=True, order=True)
class PublicSpan:
    """An inclusive, binary anomalous span in one prepared log sample."""

    start: int
    end: int

    def validate(self, length: int) -> None:
        if length <= 0:
            raise PublicProtocolError("a span requires at least one log line")
        if self.start < 0 or self.end < self.start or self.end >= length:
            raise PublicProtocolError(
                f"invalid inclusive span ({self.start}, {self.end}) for length {length}"
            )


@dataclass(frozen=True)
class PreparedSample:
    """A public benchmark record independent of the private competition schema."""

    sid: str
    lines: tuple[str, ...]
    has_anomaly: int
    spans: tuple[PublicSpan, ...]
    source_group: str
    source_line_ids: tuple[str, ...]
    template_key: str
    timestamp: str | None = None

    def validate(self, profile: TaskProfile) -> None:
        if not self.sid.strip():
            raise PublicProtocolError("prepared sample id cannot be empty")
        if not self.lines or any(not line.strip() for line in self.lines):
            raise PublicProtocolError(f"sample {self.sid} must contain non-empty log lines")
        if self.has_anomaly not in {0, 1}:
            raise PublicProtocolError(f"sample {self.sid} has_anomaly must be 0 or 1")
        if not self.source_group.strip():
            raise PublicProtocolError(f"sample {self.sid} source_group cannot be empty")
        if len(self.source_line_ids) != len(self.lines):
            raise PublicProtocolError(
                f"sample {self.sid} source_line_ids must align with log lines"
            )
        if len(set(self.source_line_ids)) != len(self.source_line_ids):
            raise PublicProtocolError(f"sample {self.sid} repeats a source line id")
        if not self.template_key.strip():
            raise PublicProtocolError(f"sample {self.sid} template_key cannot be empty")
        previous_end = -1
        for span in self.spans:
            span.validate(len(self.lines))
            if span.start <= previous_end:
                raise PublicProtocolError(f"sample {self.sid} spans overlap or are unsorted")
            previous_end = span.end
        if self.has_anomaly == 0 and self.spans:
            raise PublicProtocolError(f"normal sample {self.sid} cannot contain spans")
        if self.has_anomaly == 1 and profile is TaskProfile.SPAN_BINARY and not self.spans:
            raise PublicProtocolError(f"anomalous span sample {self.sid} requires a span")


@dataclass(frozen=True)
class PreparedManifest:
    """Stable provenance attached to one locally prepared public dataset."""

    schema_version: int
    dataset: str
    profile: str
    source_sha256: dict[str, str]
    source_metadata: dict[str, str]
    preparation: dict[str, Any]
    sample_count: int
    anomalous_sample_count: int
    source_line_count: int

    def validate(self) -> None:
        if self.schema_version != 1:
            raise PublicProtocolError("unsupported public manifest schema version")
        if not self.dataset.strip():
            raise PublicProtocolError("manifest dataset cannot be empty")
        try:
            TaskProfile(self.profile)
        except ValueError as exc:
            raise PublicProtocolError(f"unsupported public profile: {self.profile}") from exc
        if not self.source_sha256 or any(len(value) != 64 for value in self.source_sha256.values()):
            raise PublicProtocolError("manifest must contain SHA256 values for every source file")
        if self.sample_count <= 0 or self.source_line_count <= 0:
            raise PublicProtocolError("manifest counts must be positive")
        if not 0 <= self.anomalous_sample_count <= self.sample_count:
            raise PublicProtocolError("manifest anomalous_sample_count is invalid")


@dataclass(frozen=True)
class PublicPrediction:
    """A binary public-task prediction without any competition anomaly type."""

    sid: str
    score: float
    has_anomaly: int
    spans: tuple[PublicSpan, ...] = ()

    def validate(self, sample: PreparedSample, profile: TaskProfile) -> None:
        if self.sid != sample.sid:
            raise PublicProtocolError("prediction id does not match prepared sample")
        if self.has_anomaly not in {0, 1}:
            raise PublicProtocolError("public prediction has_anomaly must be 0 or 1")
        if not isinstance(self.score, float):
            raise PublicProtocolError("public prediction score must be a float")
        if self.has_anomaly == 0 and self.spans:
            raise PublicProtocolError("normal public prediction cannot contain spans")
        if profile is TaskProfile.SEQUENCE_BINARY and self.spans:
            raise PublicProtocolError("sequence prediction cannot contain spans")
        previous_end = -1
        for span in self.spans:
            span.validate(len(sample.lines))
            if span.start <= previous_end:
                raise PublicProtocolError("prediction spans overlap or are unsorted")
            previous_end = span.end


def template_signature(lines: Iterable[str]) -> str:
    """Hash normalized lines without exposing raw logs in split artifacts."""

    normalized = "\n".join(clean_log_line(line) for line in lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def spans_from_mask(mask: Iterable[bool | int]) -> tuple[PublicSpan, ...]:
    values = [bool(value) for value in mask]
    spans: list[PublicSpan] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append(PublicSpan(start=start, end=index - 1))
            start = None
    if start is not None:
        spans.append(PublicSpan(start=start, end=len(values) - 1))
    return tuple(spans)


def mask_from_spans(length: int, spans: Iterable[PublicSpan]) -> list[int]:
    if length <= 0:
        raise PublicProtocolError("mask length must be positive")
    values = [0] * length
    for span in spans:
        span.validate(length)
        for index in range(span.start, span.end + 1):
            values[index] = 1
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_to_record(sample: PreparedSample) -> dict[str, object]:
    return {
        "sid": sample.sid,
        "lines": list(sample.lines),
        "has_anomaly": sample.has_anomaly,
        "spans": [asdict(span) for span in sample.spans],
        "source_group": sample.source_group,
        "source_line_ids": list(sample.source_line_ids),
        "template_key": sample.template_key,
        "timestamp": sample.timestamp,
    }


def _sample_from_record(record: object, profile: TaskProfile) -> PreparedSample:
    if not isinstance(record, dict):
        raise PublicProtocolError("prepared sample record must be an object")
    try:
        raw_spans = record.get("spans", [])
        if not isinstance(raw_spans, list):
            raise TypeError("spans is not a list")
        sample = PreparedSample(
            sid=str(record["sid"]),
            lines=tuple(str(line) for line in record["lines"]),
            has_anomaly=int(record["has_anomaly"]),
            spans=tuple(PublicSpan(int(span["start"]), int(span["end"])) for span in raw_spans),
            source_group=str(record["source_group"]),
            source_line_ids=tuple(str(item) for item in record["source_line_ids"]),
            template_key=str(record["template_key"]),
            timestamp=None if record.get("timestamp") is None else str(record["timestamp"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicProtocolError("invalid prepared sample record") from exc
    sample.validate(profile)
    return sample


def write_prepared_dataset(
    path: Path,
    samples: Iterable[PreparedSample],
    profile: TaskProfile,
) -> tuple[PreparedSample, ...]:
    """Write deterministic JSONL after validating IDs and the task contract."""

    items = tuple(samples)
    if not items:
        raise PublicProtocolError("cannot write an empty prepared dataset")
    seen: set[str] = set()
    for sample in items:
        sample.validate(profile)
        if sample.sid in seen:
            raise PublicProtocolError(f"duplicate prepared sample id: {sample.sid}")
        seen.add(sample.sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in items:
            handle.write(json.dumps(_sample_to_record(sample), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return items


def read_prepared_dataset(path: Path, profile: TaskProfile) -> tuple[PreparedSample, ...]:
    if not path.is_file():
        raise PublicProtocolError(f"prepared dataset does not exist: {path}")
    samples: list[PreparedSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PublicProtocolError(f"blank prepared dataset row at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PublicProtocolError(f"invalid JSONL at row {line_number}") from exc
            samples.append(_sample_from_record(record, profile))
    if not samples:
        raise PublicProtocolError("prepared dataset is empty")
    if len({sample.sid for sample in samples}) != len(samples):
        raise PublicProtocolError("prepared dataset contains duplicate ids")
    return tuple(samples)


def write_manifest(path: Path, manifest: PreparedManifest) -> None:
    manifest.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path) -> PreparedManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = PreparedManifest(**payload)
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicProtocolError(f"invalid public manifest: {path}") from exc
    manifest.validate()
    return manifest
