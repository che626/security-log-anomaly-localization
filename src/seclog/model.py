import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_LABELS, N_TYPES, TYPE_TO_ID
from .data import Sample


class LogBoundaryNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 160,
        hidden: int = 224,
        num_layers: int = 2,
        dropout: float = 0.25,
        num_types: int = N_TYPES,
    ) -> None:
        super().__init__()
        if num_types < 1:
            raise ValueError("num_types must be positive")
        self.num_types = num_types
        self.emb = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean", include_last_offset=False)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim + 10, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(hidden)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden * 2, 1 + 2 * num_types)
        self.start_head = nn.Linear(hidden * 2, num_types)
        self.end_head = nn.Linear(hidden * 2, num_types)
        self.global_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_types + 1),
        )

    def forward(self, input_ids, offsets, owner, mask, pos_feats):
        line_emb = self.emb(input_ids, offsets)
        batch_size, max_len = mask.shape
        dense = torch.zeros(
            (batch_size, max_len, line_emb.shape[-1]),
            device=line_emb.device,
            dtype=line_emb.dtype,
        )
        dense[owner[:, 0], owner[:, 1]] = line_emb
        features = self.proj(torch.cat([dense, pos_feats.to(dense.device)], dim=-1))
        channels = features.transpose(1, 2)
        convolved = 0.5 * (self.conv3(channels) + self.conv5(channels)).transpose(1, 2)
        features = self.conv_norm(features + F.gelu(convolved))
        if getattr(self, "disable_cudnn_rnn", False) and features.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                hidden, _ = self.gru(features.float())
        else:
            hidden, _ = self.gru(features)
        hidden = self.dropout(hidden)
        tag_logits = self.tag_head(hidden)
        start_logits = self.start_head(hidden)
        end_logits = self.end_head(hidden)
        mask_float = mask.to(hidden.dtype).unsqueeze(-1)
        mean_pool = (hidden * mask_float).sum(1) / mask_float.sum(1).clamp_min(1)
        very_negative = torch.finfo(hidden.dtype).min / 4
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), very_negative).max(1).values
        global_logits = self.global_head(torch.cat([mean_pool, max_pool], dim=-1))
        return tag_logits, start_logits, end_logits, global_logits


def sequence_loss_weights(samples: list[Sample], o_weight: float = 0.18) -> torch.Tensor:
    counts = np.ones(N_LABELS, dtype=np.float64)
    for sample in samples:
        if sample.labels is not None:
            for label in sample.labels:
                counts[int(label)] += 1
    inverse = 1.0 / np.sqrt(counts)
    inverse = inverse / inverse.mean()
    inverse[0] *= o_weight
    inverse = np.clip(inverse, 0.05, 8.0)
    return torch.tensor(inverse, dtype=torch.float32)


def boundary_positive_weights(samples: list[Sample]) -> torch.Tensor:
    positive = np.ones(N_TYPES, dtype=np.float64)
    total_lines = 0
    for sample in samples:
        total_lines += len(sample.lines)
        if sample.spans:
            for _start, _end, anomaly_type in sample.spans:
                positive[TYPE_TO_ID[anomaly_type]] += 1
    negative = max(1, total_lines) - positive
    weights = np.sqrt(negative / positive)
    weights = np.clip(weights, 3.0, 25.0)
    return torch.tensor(weights, dtype=torch.float32)
