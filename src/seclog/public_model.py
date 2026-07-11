"""Binary public-task view of the existing CNN + BiGRU log encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import encode_log_line
from .model import LogBoundaryNetwork
from .public_protocol import PreparedSample, PublicProtocolError, TaskProfile, mask_from_spans


@dataclass(frozen=True)
class PublicModelSample:
    prepared: PreparedSample
    token_ids: tuple[tuple[int, ...], ...]
    tag_labels: np.ndarray
    start_labels: np.ndarray
    end_labels: np.ndarray
    global_label: int


class PublicSequenceDataset(Dataset):
    def __init__(self, samples: Iterable[PublicModelSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PublicModelSample:
        return self.samples[index]


def build_public_model_samples(
    samples: Iterable[PreparedSample],
    profile: TaskProfile,
    *,
    vocab_size: int,
    max_tokens: int,
) -> tuple[PublicModelSample, ...]:
    """Tokenise a binary public task without reusing any ISCC type id."""

    items: list[PublicModelSample] = []
    for sample in samples:
        sample.validate(profile)
        length = len(sample.lines)
        tag_labels = np.zeros(length, dtype=np.int64)
        start_labels = np.zeros((length, 1), dtype=np.float32)
        end_labels = np.zeros((length, 1), dtype=np.float32)
        if profile is TaskProfile.SPAN_BINARY:
            for span in sample.spans:
                tag_labels[span.start] = 1
                if span.end > span.start:
                    tag_labels[span.start + 1 : span.end + 1] = 2
                start_labels[span.start, 0] = 1.0
                end_labels[span.end, 0] = 1.0
        items.append(
            PublicModelSample(
                prepared=sample,
                token_ids=tuple(
                    tuple(encode_log_line(line, vocab_size, max_tokens)) for line in sample.lines
                ),
                tag_labels=tag_labels,
                start_labels=start_labels,
                end_labels=end_labels,
                # The reusable global head reserves index n_types for normal.
                global_label=0 if sample.has_anomaly else 1,
            )
        )
    if not items:
        raise PublicProtocolError("cannot build public model samples from an empty dataset")
    return tuple(items)


def pack_public_batch(batch: list[PublicModelSample]) -> dict[str, Any]:
    if not batch:
        raise PublicProtocolError("cannot pack an empty public model batch")
    batch_size = len(batch)
    lengths = [len(item.prepared.lines) for item in batch]
    max_length = max(lengths)
    flat_ids: list[int] = []
    offsets: list[int] = []
    owners: list[tuple[int, int]] = []
    for batch_index, item in enumerate(batch):
        for line_index, token_ids in enumerate(item.token_ids):
            offsets.append(len(flat_ids))
            flat_ids.extend(token_ids if token_ids else (0,))
            owners.append((batch_index, line_index))
    input_ids = torch.tensor(flat_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    owner = torch.tensor(owners, dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    pos_feats = torch.zeros((batch_size, max_length, 10), dtype=torch.float32)
    tags = torch.full((batch_size, max_length), -100, dtype=torch.long)
    starts = torch.zeros((batch_size, max_length, 1), dtype=torch.float32)
    ends = torch.zeros((batch_size, max_length, 1), dtype=torch.float32)
    global_labels = torch.zeros((batch_size,), dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = len(item.prepared.lines)
        mask[batch_index, :length] = True
        tags[batch_index, :length] = torch.from_numpy(item.tag_labels)
        starts[batch_index, :length] = torch.from_numpy(item.start_labels)
        ends[batch_index, :length] = torch.from_numpy(item.end_labels)
        global_labels[batch_index] = item.global_label
        for line_index in range(length):
            position = line_index / max(1, length - 1)
            pos_feats[batch_index, line_index] = torch.tensor(
                (
                    position,
                    1.0 - position,
                    float(np.sin(2 * np.pi * position)),
                    float(np.cos(2 * np.pi * position)),
                    float(np.sin(4 * np.pi * position)),
                    float(np.cos(4 * np.pi * position)),
                    min(line_index, 24) / 24.0,
                    min(length - 1 - line_index, 24) / 24.0,
                    float(np.log1p(line_index) / np.log1p(max(1, length))),
                    float(np.log1p(length - 1 - line_index) / np.log1p(max(1, length))),
                ),
                dtype=torch.float32,
            )
    return {
        "input_ids": input_ids,
        "offsets": offsets_tensor,
        "owner": owner,
        "mask": mask,
        "pos_feats": pos_feats,
        "tags": tags,
        "starts": starts,
        "ends": ends,
        "global_labels": global_labels,
        "samples": batch,
    }


def create_public_model(
    *,
    vocab_size: int,
    emb_dim: int,
    hidden: int,
    layers: int,
    dropout: float,
) -> LogBoundaryNetwork:
    """Reuse the established encoder with a single internal binary anomaly type."""

    return LogBoundaryNetwork(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden=hidden,
        num_layers=layers,
        dropout=dropout,
        num_types=1,
    )


def public_line_labels(sample: PublicModelSample) -> list[int]:
    return mask_from_spans(len(sample.prepared.lines), sample.prepared.spans)
