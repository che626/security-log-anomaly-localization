"""Explicit local-source adapters for public log anomaly benchmarks."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .public_protocol import (
    PreparedManifest,
    PreparedSample,
    PublicProtocolError,
    TaskProfile,
    sha256_file,
    spans_from_mask,
    template_signature,
    write_manifest,
    write_prepared_dataset,
)

_HDFS_BLOCK = re.compile(r"\bblk_-?\d+\b")
_LABEL_COLUMNS = ("label", "anomaly", "is_anomaly", "anomalous")
_GROUP_COLUMNS = ("blockid", "block_id", "group", "group_id", "instance_id", "id")
_MESSAGE_COLUMNS = ("message", "content", "log", "log_message", "line")
_NORMAL_LABELS = {"-", "normal", "0", "false", "no", "none"}
_RAW_HDFS_TIMESTAMP = re.compile(r"^\s*(\d{6})\s+(\d{6})\b")


@dataclass(frozen=True)
class PreparedPaths:
    dataset_path: Path
    manifest_path: Path


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PublicProtocolError(f"{description} does not exist or is not a file: {path}")


def _normalise_column(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], description: str) -> str:
    normalized = {_normalise_column(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise PublicProtocolError(
        f"cannot find {description}; expected one of {sorted(candidates)}, "
        f"found {list(frame.columns)}"
    )


def _find_optional_column(frame: pd.DataFrame, candidate: str) -> str | None:
    normalized = {_normalise_column(column): str(column) for column in frame.columns}
    return normalized.get(candidate)


def _is_anomaly(value: object) -> int:
    text = str(value).strip().lower()
    if not text:
        raise PublicProtocolError("empty anomaly label")
    return int(text not in _NORMAL_LABELS)


def _manifest(
    dataset: str,
    profile: TaskProfile,
    sources: dict[str, Path],
    metadata: dict[str, str],
    preparation: dict[str, object],
    samples: tuple[PreparedSample, ...],
) -> PreparedManifest:
    return PreparedManifest(
        schema_version=1,
        dataset=dataset,
        profile=profile.value,
        source_sha256={name: sha256_file(path) for name, path in sources.items()},
        source_metadata=metadata,
        preparation=preparation,
        sample_count=len(samples),
        anomalous_sample_count=sum(sample.has_anomaly for sample in samples),
        source_line_count=sum(len(sample.lines) for sample in samples),
    )


def _write_bundle(
    dataset: str,
    profile: TaskProfile,
    samples: Iterable[PreparedSample],
    sources: dict[str, Path],
    metadata: dict[str, str],
    preparation: dict[str, object],
    output_dir: Path,
) -> PreparedPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{dataset}-{profile.value}.jsonl"
    manifest_path = output_dir / f"{dataset}-{profile.value}.manifest.json"
    items = write_prepared_dataset(dataset_path, samples, profile)
    manifest = _manifest(dataset, profile, sources, metadata, preparation, items)
    write_manifest(manifest_path, manifest)
    return PreparedPaths(dataset_path=dataset_path, manifest_path=manifest_path)


def prepare_hdfs(
    log_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> PreparedPaths:
    """Prepare HDFS block sequences from a raw log and official label CSV."""

    _require_file(log_path, "HDFS log source")
    _require_file(labels_path, "HDFS label source")
    labels = pd.read_csv(labels_path)
    group_column = _find_column(labels, _GROUP_COLUMNS, "HDFS block-id column")
    label_column = _find_column(labels, _LABEL_COLUMNS, "HDFS label column")
    label_map: dict[str, int] = {}
    for _, row in labels.iterrows():
        group = str(row[group_column]).strip()
        if not group:
            raise PublicProtocolError("HDFS labels contain an empty block id")
        label = _is_anomaly(row[label_column])
        if group in label_map and label_map[group] != label:
            raise PublicProtocolError(f"HDFS block {group} has conflicting labels")
        label_map[group] = label
    grouped: OrderedDict[str, list[tuple[str, str, str | None]]] = OrderedDict()
    if log_path.suffix.lower() == ".csv":
        structured = pd.read_csv(log_path, dtype=str)
        content_column = _find_column(structured, ("content", "message", "log"), "HDFS content column")
        date_column = _find_optional_column(structured, "date")
        time_column = _find_optional_column(structured, "time")
        for row_number, row in structured.iterrows():
            line = str(row[content_column]).strip()
            if not line:
                continue
            blocks = set(_HDFS_BLOCK.findall(line))
            if len(blocks) != 1:
                continue
            block = next(iter(blocks))
            timestamp = None
            if date_column is not None and time_column is not None:
                timestamp = f"{str(row[date_column]).strip()} {str(row[time_column]).strip()}"
            grouped.setdefault(block, []).append((str(row_number + 1), line, timestamp))
        input_kind = "structured_csv"
    else:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                blocks = set(_HDFS_BLOCK.findall(line))
                if len(blocks) != 1:
                    continue
                block = next(iter(blocks))
                match = _RAW_HDFS_TIMESTAMP.match(line)
                timestamp = f"{match.group(1)} {match.group(2)}" if match else None
                grouped.setdefault(block, []).append((str(line_number), line, timestamp))
        input_kind = "raw_text"
    if not grouped:
        raise PublicProtocolError("HDFS source contains no lines with exactly one block id")
    missing_labels = sorted(set(grouped) - set(label_map))
    if missing_labels:
        preview = ", ".join(missing_labels[:3])
        raise PublicProtocolError(f"HDFS blocks are missing official labels: {preview}")
    samples = tuple(
        PreparedSample(
            sid=f"hdfs:{block}",
            lines=tuple(line for _source_id, line, _timestamp in records),
            has_anomaly=label_map[block],
            spans=(),
            source_group=block,
            source_line_ids=tuple(
                f"hdfs:{source_id}" for source_id, _line, _timestamp in records
            ),
            template_key=template_signature(line for _source_id, line, _timestamp in records),
            timestamp=records[0][2],
        )
        for block, records in grouped.items()
    )
    return _write_bundle(
        dataset="hdfs",
        profile=TaskProfile.SEQUENCE_BINARY,
        samples=samples,
        sources={"log": log_path, "labels": labels_path},
        metadata={"source": "Loghub HDFS", "label_unit": "block"},
        preparation={
            "block_pattern": _HDFS_BLOCK.pattern,
            "input_kind": input_kind,
            "timestamp": "first block log-line Date+Time when available",
            "unlabelled_lines": "ignored",
        },
        output_dir=output_dir,
    )


def prepare_grouped_csv(
    dataset: str,
    logs_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    group_column: str | None = None,
    message_column: str | None = None,
    label_group_column: str | None = None,
    label_column: str | None = None,
) -> PreparedPaths:
    """Prepare a labelled sequence benchmark from explicit grouped CSV sources.

    This is used for OpenStack because source distributions differ.  It refuses
    to infer a grouping rule from free-form log text; the group/label mapping is
    instead explicit and recorded in the generated manifest.
    """

    _require_file(logs_path, f"{dataset} grouped log source")
    _require_file(labels_path, f"{dataset} label source")
    logs = pd.read_csv(logs_path)
    labels = pd.read_csv(labels_path)
    log_group = group_column or _find_column(logs, _GROUP_COLUMNS, f"{dataset} log group column")
    message = message_column or _find_column(logs, _MESSAGE_COLUMNS, f"{dataset} message column")
    label_group = label_group_column or _find_column(
        labels, _GROUP_COLUMNS, f"{dataset} label group column"
    )
    label_name = label_column or _find_column(labels, _LABEL_COLUMNS, f"{dataset} label column")
    for column, frame_name, frame in ((log_group, "logs", logs), (message, "logs", logs), (label_group, "labels", labels), (label_name, "labels", labels)):
        if column not in frame.columns:
            raise PublicProtocolError(f"{dataset} {frame_name} source is missing requested column {column}")
    label_map: dict[str, int] = {}
    for _, row in labels.iterrows():
        group = str(row[label_group]).strip()
        if not group:
            raise PublicProtocolError(f"{dataset} labels contain an empty group")
        label = _is_anomaly(row[label_name])
        if group in label_map and label_map[group] != label:
            raise PublicProtocolError(f"{dataset} group {group} has conflicting labels")
        label_map[group] = label
    grouped: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for row_number, row in logs.iterrows():
        group = str(row[log_group]).strip()
        line = str(row[message]).strip()
        if not group or not line:
            raise PublicProtocolError(f"{dataset} log row {row_number} has an empty group or message")
        grouped.setdefault(group, []).append((str(row_number), line))
    missing_labels = sorted(set(grouped) - set(label_map))
    if missing_labels:
        raise PublicProtocolError(f"{dataset} log groups are missing labels: {', '.join(missing_labels[:3])}")
    samples = tuple(
        PreparedSample(
            sid=f"{dataset}:{group}",
            lines=tuple(line for _row, line in records),
            has_anomaly=label_map[group],
            spans=(),
            source_group=group,
            source_line_ids=tuple(f"{dataset}:{row}" for row, _line in records),
            template_key=template_signature(line for _row, line in records),
        )
        for group, records in grouped.items()
    )
    return _write_bundle(
        dataset=dataset,
        profile=TaskProfile.SEQUENCE_BINARY,
        samples=samples,
        sources={"logs": logs_path, "labels": labels_path},
        metadata={"source": f"Loghub {dataset}", "label_unit": "explicit group"},
        preparation={
            "log_group_column": log_group,
            "message_column": message,
            "label_group_column": label_group,
            "label_column": label_name,
        },
        output_dir=output_dir,
    )


def prepare_openstack(
    logs_path: Path,
    labels_path: Path,
    output_dir: Path,
    **columns: str,
) -> PreparedPaths:
    return prepare_grouped_csv("openstack", logs_path, labels_path, output_dir, **columns)


def _parse_labelled_lines(path: Path, dataset: str) -> list[tuple[str, str, int, str | None]]:
    _require_file(path, f"{dataset} log source")
    parsed: list[tuple[str, str, int, str | None]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            bits = raw.split(maxsplit=2)
            if len(bits) < 2:
                raise PublicProtocolError(f"{dataset} line {line_number} lacks a label and message")
            label = _is_anomaly(bits[0])
            timestamp = bits[1] if len(bits) >= 3 else None
            message = bits[2].strip() if len(bits) >= 3 else bits[1].strip()
            if not message:
                raise PublicProtocolError(f"{dataset} line {line_number} has an empty message")
            parsed.append((str(line_number), message, label, timestamp))
    if not parsed:
        raise PublicProtocolError(f"{dataset} source contains no labelled log lines")
    return parsed


def prepare_line_windows(
    dataset: str,
    log_path: Path,
    output_dir: Path,
    *,
    window_size: int,
    stride: int,
    max_source_lines: int | None = None,
) -> PreparedPaths:
    """Prepare non-overlapping binary span windows from labelled raw log lines."""

    if window_size <= 0 or stride <= 0:
        raise PublicProtocolError("window_size and stride must be positive")
    if stride < window_size:
        raise PublicProtocolError(
            "stride must be at least window_size so source lines cannot leak across split windows"
        )
    if max_source_lines is not None and max_source_lines <= 0:
        raise PublicProtocolError("max_source_lines must be positive when supplied")
    records = _parse_labelled_lines(log_path, dataset)
    if max_source_lines is not None:
        records = records[:max_source_lines]
    samples: list[PreparedSample] = []
    for start in range(0, len(records), stride):
        window = records[start : start + window_size]
        if not window:
            continue
        lines = tuple(item[1] for item in window)
        mask = [item[2] for item in window]
        spans = spans_from_mask(mask)
        source_ids = tuple(f"{dataset}:{item[0]}" for item in window)
        samples.append(
            PreparedSample(
                sid=f"{dataset}:window:{start}:{start + len(window) - 1}",
                lines=lines,
                has_anomaly=int(any(mask)),
                spans=spans,
                source_group=f"{dataset}:window:{start}:{start + len(window) - 1}",
                source_line_ids=source_ids,
                template_key=template_signature(lines),
                timestamp=window[0][3],
            )
        )
    return _write_bundle(
        dataset=dataset,
        profile=TaskProfile.SPAN_BINARY,
        samples=samples,
        sources={"log": log_path},
        metadata={"source": f"Loghub {dataset}", "label_unit": "line"},
        preparation={
            "window_size": window_size,
            "stride": stride,
            "max_source_lines": max_source_lines,
            "overlap": False,
        },
        output_dir=output_dir,
    )


def prepare_bgl(
    log_path: Path,
    output_dir: Path,
    *,
    window_size: int,
    stride: int,
    max_source_lines: int | None = None,
) -> PreparedPaths:
    return prepare_line_windows(
        "bgl",
        log_path,
        output_dir,
        window_size=window_size,
        stride=stride,
        max_source_lines=max_source_lines,
    )


def prepare_thunderbird(
    log_path: Path,
    output_dir: Path,
    *,
    window_size: int,
    stride: int,
    max_source_lines: int,
) -> PreparedPaths:
    return prepare_line_windows(
        "thunderbird",
        log_path,
        output_dir,
        window_size=window_size,
        stride=stride,
        max_source_lines=max_source_lines,
    )
