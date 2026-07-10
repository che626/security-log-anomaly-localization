import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import GLOBAL_NONE_ID, N_TYPES, TYPE_TO_ID
from .features import encode_log_line, nonempty_log_lines
from .schemas import SchemaError, validate_test_frame, validate_training_frame

Span = tuple[int, int, str]


@dataclass
class Sample:
    sid: int
    lines: list[str]
    token_ids: list[list[int]]
    labels: np.ndarray | None = None
    start_labels: np.ndarray | None = None
    end_labels: np.ndarray | None = None
    global_label: int | None = None
    has_anomaly: int | None = None
    primary: Span | None = None
    spans: list[Span] | None = None


class SpanSequenceDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def parse_annotation_spans(
    all_spans: object,
    has_anomaly: int,
    primary_start: int,
    primary_end: int,
    primary_type: str,
) -> list[Span]:
    spans: list[Span] = []
    if int(has_anomaly) == 0:
        return spans
    if isinstance(all_spans, str) and all_spans.strip():
        for part in all_spans.split(";"):
            bits = part.strip().split("|")
            if len(bits) >= 3:
                try:
                    start, end, anomaly_type = int(bits[0]), int(bits[1]), bits[2]
                except ValueError:
                    continue
                if anomaly_type in TYPE_TO_ID and start <= end:
                    spans.append((start, end, anomaly_type))
    if not spans and primary_type in TYPE_TO_ID:
        spans.append((int(primary_start), int(primary_end), primary_type))
    return sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))


def build_sequence_labels(length: int, spans: list[Span]) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    for start, end, anomaly_type in spans:
        if anomaly_type not in TYPE_TO_ID or length == 0:
            continue
        start = max(0, min(length - 1, int(start)))
        end = max(0, min(length - 1, int(end)))
        if start > end:
            continue
        type_id = TYPE_TO_ID[anomaly_type]
        labels[start] = 1 + type_id
        if end > start:
            labels[start + 1 : end + 1] = 1 + N_TYPES + type_id
    return labels


def build_endpoint_labels(length: int, spans: list[Span]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.zeros((length, N_TYPES), dtype=np.float32)
    ends = np.zeros((length, N_TYPES), dtype=np.float32)
    for start, end, anomaly_type in spans:
        if anomaly_type in TYPE_TO_ID and 0 <= start < length and 0 <= end < length:
            type_id = TYPE_TO_ID[anomaly_type]
            starts[start, type_id] = 1.0
            ends[end, type_id] = 1.0
    return starts, ends


def pack_log_batch(batch: list[Sample]) -> dict[str, Any]:
    if not batch or any(not sample.lines for sample in batch):
        raise ValueError("every batch and sample must contain at least one log line")
    batch_size = len(batch)
    lengths = [len(sample.lines) for sample in batch]
    max_len = max(lengths)
    flat_ids: list[int] = []
    offsets: list[int] = []
    owners: list[tuple[int, int]] = []
    for batch_index, sample in enumerate(batch):
        for line_index, token_ids in enumerate(sample.token_ids):
            offsets.append(len(flat_ids))
            flat_ids.extend(token_ids if token_ids else [0])
            owners.append((batch_index, line_index))
    input_ids = torch.tensor(flat_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    owner = torch.tensor(owners, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    pos_feats = torch.zeros((batch_size, max_len, 10), dtype=torch.float32)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    start_labels = torch.zeros((batch_size, max_len, N_TYPES), dtype=torch.float32)
    end_labels = torch.zeros((batch_size, max_len, N_TYPES), dtype=torch.float32)
    global_labels = torch.full((batch_size,), GLOBAL_NONE_ID, dtype=torch.long)
    for batch_index, sample in enumerate(batch):
        length = len(sample.lines)
        mask[batch_index, :length] = True
        if sample.labels is not None:
            labels[batch_index, :length] = torch.tensor(sample.labels, dtype=torch.long)
        if sample.start_labels is not None:
            start_labels[batch_index, :length] = torch.tensor(
                sample.start_labels, dtype=torch.float32
            )
        if sample.end_labels is not None:
            end_labels[batch_index, :length] = torch.tensor(
                sample.end_labels, dtype=torch.float32
            )
        if sample.global_label is not None:
            global_labels[batch_index] = int(sample.global_label)
        for line_index in range(length):
            position = line_index / max(1, length - 1)
            pos_feats[batch_index, line_index, 0] = position
            pos_feats[batch_index, line_index, 1] = 1.0 - position
            pos_feats[batch_index, line_index, 2] = math.sin(2 * math.pi * position)
            pos_feats[batch_index, line_index, 3] = math.cos(2 * math.pi * position)
            pos_feats[batch_index, line_index, 4] = math.sin(4 * math.pi * position)
            pos_feats[batch_index, line_index, 5] = math.cos(4 * math.pi * position)
            pos_feats[batch_index, line_index, 6] = min(line_index, 24) / 24.0
            pos_feats[batch_index, line_index, 7] = (
                min(length - 1 - line_index, 24) / 24.0
            )
            pos_feats[batch_index, line_index, 8] = math.log1p(line_index) / math.log1p(
                max(1, length)
            )
            pos_feats[batch_index, line_index, 9] = math.log1p(
                length - 1 - line_index
            ) / math.log1p(max(1, length))
    return {
        "input_ids": input_ids,
        "offsets": offsets_tensor,
        "owner": owner,
        "mask": mask,
        "pos_feats": pos_feats,
        "labels": labels,
        "start_labels": start_labels,
        "end_labels": end_labels,
        "global_labels": global_labels,
        "samples": batch,
    }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    train_sha256: str
    test_sha256: str
    train_rows: int
    test_rows: int
    vocab_size: int
    max_tokens: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_to_sample(
    row: pd.Series,
    is_training: bool,
    vocab_size: int,
    max_tokens: int,
) -> Sample:
    lines = nonempty_log_lines(row["log_text"])
    if not lines:
        raise SchemaError(f"id {row['id']} has no non-empty log lines")
    token_ids = [encode_log_line(line, vocab_size, max_tokens) for line in lines]
    if not is_training:
        return Sample(sid=int(row["id"]), lines=lines, token_ids=token_ids)
    has_anomaly = int(row["has_anomaly"])
    spans = parse_annotation_spans(
        row.get("all_spans", ""),
        has_anomaly,
        int(row["primary_start_idx"]),
        int(row["primary_end_idx"]),
        str(row["primary_anomaly_type"]),
    )
    labels = build_sequence_labels(len(lines), spans)
    start_labels, end_labels = build_endpoint_labels(len(lines), spans)
    primary = None
    global_label = GLOBAL_NONE_ID
    if has_anomaly:
        primary_type = str(row["primary_anomaly_type"])
        primary = (
            int(row["primary_start_idx"]),
            int(row["primary_end_idx"]),
            primary_type,
        )
        global_label = TYPE_TO_ID[primary_type]
    return Sample(
        sid=int(row["id"]),
        lines=lines,
        token_ids=token_ids,
        labels=labels,
        start_labels=start_labels,
        end_labels=end_labels,
        global_label=global_label,
        has_anomaly=has_anomaly,
        primary=primary,
        spans=spans,
    )


def load_dataset(
    data_dir: Path,
    cache_dir: Path,
    vocab_size: int,
    max_tokens: int,
    rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Sample], list[Sample], DatasetManifest]:
    """Load validated CSVs and trusted caches generated by this function only.

    Pickle can execute code while loading. Never point ``cache_dir`` at files from
    an untrusted source.
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    validate_training_frame(train_frame)
    validate_test_frame(test_frame)
    manifest = DatasetManifest(
        schema_version=1,
        train_sha256=sha256_file(train_path),
        test_sha256=sha256_file(test_path),
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        vocab_size=vocab_size,
        max_tokens=max_tokens,
    )
    cache_path = cache_dir / f"tokens-v{vocab_size}-m{max_tokens}.pkl"
    manifest_path = cache_path.with_suffix(".manifest.json")
    if cache_path.exists() and manifest_path.exists() and not rebuild_cache:
        cached_manifest = DatasetManifest(
            **json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if cached_manifest == manifest:
            with cache_path.open("rb") as handle:
                train_samples, test_samples = pickle.load(handle)
            return train_frame, test_frame, train_samples, test_samples, manifest
    train_samples = [
        _row_to_sample(row, True, vocab_size, max_tokens)
        for _, row in train_frame.iterrows()
    ]
    test_samples = [
        _row_to_sample(row, False, vocab_size, max_tokens)
        for _, row in test_frame.iterrows()
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump((train_samples, test_samples), handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_frame, test_frame, train_samples, test_samples, manifest
