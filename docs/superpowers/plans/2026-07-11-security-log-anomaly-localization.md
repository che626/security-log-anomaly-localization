# Security Log Anomaly Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and publish a focused, reproducible portfolio repository for multi-task system-log anomaly detection, span localization, and anomaly-type classification, with an honest ISCC 2026 retrospective.

**Architecture:** Refactor the selected single-file competition pipeline into small Python modules with stable interfaces for schemas, preprocessing, data loading, modeling, decoding, evaluation, training, and inference. Keep competition data and checkpoints private; publish deterministic synthetic fixtures, tests, documentation, metric manifests, and a local visualization adapter.

**Tech Stack:** Python 3.10/3.11, PyTorch, NumPy, pandas, scikit-learn, PyYAML, Streamlit, pytest, Ruff, GitHub Actions.

---

## Source Map and File Responsibilities

The implementation starts from the approved design and the private system-log reference pipeline. The private archive is read-only and never committed.

| New file | Responsibility |
|---|---|
| pyproject.toml | Package metadata, runtime dependencies, optional demo/dev dependencies, pytest and Ruff configuration |
| .gitignore | Exclude private data, checkpoints, caches, local outputs, virtual environments, and editor files |
| .gitattributes | Normalize text files to LF |
| configs/final.yaml | Selected historical model and decoder configuration |
| configs/smoke.yaml | Tiny CPU-only configuration used by tests |
| src/seclog/constants.py | Labels, submission columns, and regex patterns |
| src/seclog/schemas.py | Dataframe, prediction, span, and manifest validation |
| src/seclog/features.py | Log normalization and deterministic hashed-token features |
| src/seclog/data.py | Sample construction, annotation parsing, caching, manifests, splits, and batching |
| src/seclog/model.py | EmbeddingBag + CNN + BiGRU multi-task network and loss weights |
| src/seclog/metrics.py | Detection F1, span IoU, type F1, and official composite metric |
| src/seclog/decode.py | Viterbi decoding, endpoint assistance, boundary refinement, and structured output |
| src/seclog/tuning.py | Decoder parameter grids, tuning slices, and tuning reports |
| src/seclog/config.py | Typed YAML configuration loading and validation |
| src/seclog/splitting.py | Duplicate signatures, group-aware splits, and locked evaluation partitions |
| src/seclog/training.py | Device selection, fold training, checkpointing, OOF prediction, and smoke training |
| src/seclog/inference.py | Checkpoint loading, fold ensembling, decoding, and prediction-frame creation |
| src/seclog/cli.py | check-data, train, evaluate, and predict commands |
| src/seclog/presentation.py | UI-independent line annotation objects |
| app/streamlit_app.py | Local log input, model adapter, and anomaly-span visualization |
| scripts/build_private_manifest.py | Local-only dataset checksum and schema manifest generation |
| scripts/run_private_reproduction.py | Reproduce internal metrics without copying private data into the repository |
| scripts/audit_publication.py | Reject private, oversized, credential-like, or competition-identifying files |
| tests/fixtures/*.json | Synthetic and hand-authored regression fixtures |
| tests/test_*.py | Unit, integration, and CPU smoke tests |
| docs/*.md | Architecture, experiments, error analysis, model card, and ISCC retrospective |
| artifacts/metrics/*.json | Small verified metric and run-manifest files only |
| .github/workflows/ci.yml | Linux CPU lint and test checks |

The legacy function groups map as follows:

- constants.py: TYPES, TYPE_TO_ID, N_TYPES, N_LABELS, GLOBAL_NONE_ID, SUBMISSION_COLUMNS, regex constants.
- features.py: hashed_bucket, clean_log_line, encode_log_line, nonempty_log_lines.
- data.py: Sample, SpanSequenceDataset, annotation-label builders, pack_log_batch, dataset loading, span statistics.
- model.py: LogBoundaryNetwork, sequence_loss_weights, boundary_positive_weights.
- metrics.py: row_iou, evaluate_submission_like.
- decode.py: legacy decoding and decoder-grid functions from tag_to_type through tune_decoder.
- training.py: device, dataloader, loss, epoch, fold, checkpoint, and workflow functions.
- inference.py: model-output collection, fold averaging, checkpoint prediction, decoding, and output alignment.

## Task 1: Establish the Safe Package Skeleton

**Files:**
- Create: pyproject.toml
- Create: README.md
- Create: .gitignore
- Create: .gitattributes
- Create: src/seclog/__init__.py
- Create: tests/test_package.py
- Create: .github/workflows/ci.yml

- [ ] **Step 1: Write the package import test**

~~~python
from importlib.metadata import version

import seclog


def test_package_exposes_version() -> None:
    assert seclog.__version__ == version('security-log-anomaly-localization')
~~~

- [ ] **Step 2: Run the test and verify the package is absent**

Run:

~~~powershell
python -m pytest tests/test_package.py -v
~~~

Expected: FAIL because seclog cannot be imported.

- [ ] **Step 3: Create package metadata and the minimal package**

Use this pyproject.toml:

~~~toml
[build-system]
requires = ['hatchling>=1.24']
build-backend = 'hatchling.build'

[project]
name = 'security-log-anomaly-localization'
version = '0.1.0'
description = 'Multi-task system-log anomaly detection, localization, and classification'
readme = 'README.md'
requires-python = '>=3.10,<3.12'
license = { text = 'MIT' }
dependencies = [
  'numpy>=1.26,<3',
  'pandas>=2.0,<3',
  'scikit-learn>=1.3,<2',
  'torch>=2.2,<3',
  'tqdm>=4.66,<5',
  'PyYAML>=6,<7',
]

[project.optional-dependencies]
demo = ['streamlit>=1.36,<2']
dev = ['pytest>=8,<9', 'pytest-cov>=5,<7', 'ruff>=0.5,<1']

[project.scripts]
seclog = 'seclog.cli:main'

[tool.hatch.build.targets.wheel]
packages = ['src/seclog']

[tool.pytest.ini_options]
testpaths = ['tests']
addopts = '-ra'

[tool.ruff]
line-length = 100
target-version = 'py310'
~~~

Use this src/seclog/__init__.py:

~~~python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('security-log-anomaly-localization')
except PackageNotFoundError:
    __version__ = '0.1.0'

__all__ = ['__version__']
~~~

Create README.md with a deliberately minimal planning-stage message:

~~~markdown
# Security Log Anomaly Localization

Multi-task system-log anomaly detection, span localization, and anomaly-type
classification. Full portfolio documentation is added after implementation and
verification.
~~~

- [ ] **Step 4: Add repository safety rules**

Use this .gitignore:

~~~gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.idea/
.vscode/
.private/
data/
checkpoints/
outputs/
artifacts/checkpoints/
*.pt
*.pth
*.pkl
*.joblib
*.npy
*.npz
*.csv
!examples/**/*.csv
!tests/fixtures/**/*.csv
!artifacts/metrics/*.csv
streamlit_secrets.toml
.env
~~~

Use this .gitattributes:

~~~gitattributes
* text=auto
*.py text eol=lf
*.md text eol=lf
*.toml text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.json text eol=lf
~~~

- [ ] **Step 5: Install the editable development package and run the test**

Run:

~~~powershell
python -m pip install -e '.[dev]'
python -m pytest tests/test_package.py -v
~~~

Expected: PASS.

- [ ] **Step 6: Add CPU CI**

Create .github/workflows/ci.yml:

~~~yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip
      - run: python -m pip install -e '.[dev]'
      - run: ruff check src tests app scripts
      - run: pytest -q
~~~

- [ ] **Step 7: Verify no private or large file is staged**

Run:

~~~powershell
git status --short
git ls-files | Select-String -Pattern 'train.csv|test.csv|\.pt$|\.pkl$|\.joblib$'
~~~

Expected: only package skeleton files are staged; the second command returns no matches.

- [ ] **Step 8: Configure the connected GitHub identity locally**

Read the authenticated account name and verified/noreply email through the approved GitHub integration. Set user.name and user.email only in this repository. Do not modify global Git configuration and do not rewrite the two planning commits, which intentionally remain attributed to Codex Planning.

- [ ] **Step 9: Commit the skeleton**

~~~powershell
git add pyproject.toml README.md .gitignore .gitattributes src tests .github
git commit -m 'chore: initialize safe Python package'
~~~

## Task 2: Define Constants and Strict Schemas

**Files:**
- Create: src/seclog/constants.py
- Create: src/seclog/schemas.py
- Create: tests/test_schemas.py

- [ ] **Step 1: Write failing schema tests**

~~~python
import pandas as pd
import pytest

from seclog.schemas import SchemaError, validate_prediction_frame, validate_training_frame


def valid_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'id': [1],
            'log_text': ['line one\nline two'],
            'has_anomaly': [1],
            'primary_start_idx': [0],
            'primary_end_idx': [1],
            'primary_anomaly_type': ['timeout_retry'],
            'all_spans': ['0|1|timeout_retry'],
        }
    )


def test_training_schema_accepts_valid_frame() -> None:
    validate_training_frame(valid_training_frame())


def test_training_schema_rejects_duplicate_ids() -> None:
    frame = pd.concat([valid_training_frame(), valid_training_frame()], ignore_index=True)
    with pytest.raises(SchemaError, match='duplicate'):
        validate_training_frame(frame)


def test_prediction_schema_rejects_invalid_normal_span() -> None:
    frame = pd.DataFrame(
        {
            'id': [1],
            'has_anomaly': [0],
            'primary_start_idx': [0],
            'primary_end_idx': [0],
            'primary_anomaly_type': ['none'],
            'all_spans': [''],
        }
    )
    with pytest.raises(SchemaError, match='-1'):
        validate_prediction_frame(frame, expected_ids=[1])
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~powershell
pytest tests/test_schemas.py -v
~~~

Expected: FAIL because constants.py and schemas.py do not exist.

- [ ] **Step 3: Add stable label constants**

Create src/seclog/constants.py:

~~~python
import re

ANOMALY_TYPES = (
    'timeout_retry',
    'resource_exhaustion',
    'slow_burn_warning',
    'state_conflict',
    'parameter_drift',
    'out_of_order',
    'missing_step',
    'duplicate_event',
    'cross_component_mismatch',
    'partial_recovery_loop',
)
TYPE_TO_ID = {name: index for index, name in enumerate(ANOMALY_TYPES)}
N_TYPES = len(ANOMALY_TYPES)
N_LABELS = 1 + 2 * N_TYPES
GLOBAL_NONE_ID = N_TYPES
SUBMISSION_COLUMNS = (
    'id',
    'has_anomaly',
    'primary_start_idx',
    'primary_end_idx',
    'primary_anomaly_type',
    'all_spans',
)
TRAIN_COLUMNS = ('id', 'log_text', *SUBMISSION_COLUMNS[1:])
TEST_COLUMNS = ('id', 'log_text')

TS_RE = re.compile(r'^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*')
IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')
NUM_RE = re.compile(r'(?<![a-zA-Z_])[-+]?\d+(?:\.\d+)?')
SEG_RE = re.compile(r'\bseg[_-]?[a-zA-Z0-9]+\b', re.I)
PATH_RE = re.compile(r'/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]*')
WORD_RE = re.compile(r'[a-zA-Z_<>][a-zA-Z0-9_<>-]*|[=:/.-]+')
~~~

- [ ] **Step 4: Implement exact validation rules**

Create src/seclog/schemas.py with these public interfaces:

~~~python
from collections.abc import Iterable

import pandas as pd

from .constants import ANOMALY_TYPES, SUBMISSION_COLUMNS, TEST_COLUMNS, TRAIN_COLUMNS


class SchemaError(ValueError):
    pass


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f'missing columns: {missing}')


def _validate_ids(frame: pd.DataFrame) -> None:
    if frame['id'].isna().any():
        raise SchemaError('id contains null values')
    if frame['id'].duplicated().any():
        raise SchemaError('duplicate ids are not allowed')


def validate_training_frame(frame: pd.DataFrame) -> None:
    _require_columns(frame, TRAIN_COLUMNS)
    _validate_ids(frame)
    if frame.empty:
        raise SchemaError('training frame is empty')
    if not set(frame['has_anomaly'].astype(int)).issubset({0, 1}):
        raise SchemaError('has_anomaly must contain only 0 or 1')
    known = set(ANOMALY_TYPES) | {'none'}
    unknown = set(frame['primary_anomaly_type'].astype(str)) - known
    if unknown:
        raise SchemaError(f'unknown anomaly types: {sorted(unknown)}')


def validate_test_frame(frame: pd.DataFrame) -> None:
    _require_columns(frame, TEST_COLUMNS)
    _validate_ids(frame)
    if frame.empty:
        raise SchemaError('test frame is empty')


def validate_prediction_frame(frame: pd.DataFrame, expected_ids: Iterable[int]) -> None:
    _require_columns(frame, SUBMISSION_COLUMNS)
    _validate_ids(frame)
    expected = list(expected_ids)
    if frame['id'].tolist() != expected:
        raise SchemaError('prediction ids or order do not match the expected ids')
    normal = frame['has_anomaly'].astype(int).eq(0)
    if not frame.loc[normal, 'primary_start_idx'].astype(int).eq(-1).all():
        raise SchemaError('normal rows must use -1 for primary_start_idx')
    if not frame.loc[normal, 'primary_end_idx'].astype(int).eq(-1).all():
        raise SchemaError('normal rows must use -1 for primary_end_idx')
    if not frame.loc[normal, 'primary_anomaly_type'].astype(str).eq('none').all():
        raise SchemaError('normal rows must use none as primary_anomaly_type')
    anomalous = ~normal
    unknown = (
        set(frame.loc[anomalous, 'primary_anomaly_type'].astype(str))
        - set(ANOMALY_TYPES)
    )
    if unknown:
        raise SchemaError(f'unknown predicted anomaly types: {sorted(unknown)}')
    if (frame.loc[anomalous, 'primary_start_idx'] < 0).any():
        raise SchemaError('anomalous rows require non-negative start indices')
    if (
        frame.loc[anomalous, 'primary_end_idx'].astype(int)
        < frame.loc[anomalous, 'primary_start_idx'].astype(int)
    ).any():
        raise SchemaError('primary_end_idx must not be less than primary_start_idx')
~~~

- [ ] **Step 5: Run schema tests**

Run:

~~~powershell
pytest tests/test_schemas.py -v
~~~

Expected: PASS.

- [ ] **Step 6: Commit constants and schemas**

~~~powershell
git add src/seclog/constants.py src/seclog/schemas.py tests/test_schemas.py
git commit -m 'feat: add strict log data schemas'
~~~

## Task 3: Preserve Feature-Extraction Behavior

**Files:**
- Create: src/seclog/features.py
- Create: tests/fixtures/feature_cases.json
- Create: tests/test_features.py

- [ ] **Step 1: Add safe golden fixtures**

Create tests/fixtures/feature_cases.json:

~~~json
[
  {
    "raw": "2026-05-01 12:34:56 Host 10.0.0.7 read /var/log/app 0xAF seg-9 count=42",
    "clean": "host <ip> read <path> <hex> <seg> count=<num>"
  },
  {
    "raw": "   ",
    "clean": ""
  }
]
~~~

- [ ] **Step 2: Write failing feature tests**

~~~python
import json
from pathlib import Path

from seclog.features import clean_log_line, encode_log_line, hashed_bucket


def test_clean_log_line_matches_reference_cases() -> None:
    cases = json.loads(Path('tests/fixtures/feature_cases.json').read_text(encoding='utf-8'))
    for case in cases:
        assert clean_log_line(case['raw']) == case['clean']


def test_crc32_bucket_is_stable() -> None:
    assert hashed_bucket('w:host', 1024) == 61
    assert hashed_bucket('w:<ip>', 1024) == 401


def test_encode_empty_line_uses_nonzero_bucket() -> None:
    assert encode_log_line('', vocab_size=1024, max_tokens=8) == [756]


def test_encode_respects_max_tokens() -> None:
    assert len(encode_log_line('alpha beta gamma delta', 1024, max_tokens=3)) == 3
~~~

- [ ] **Step 3: Run the tests and verify failure**

Run:

~~~powershell
pytest tests/test_features.py -v
~~~

Expected: FAIL because features.py does not exist.

- [ ] **Step 4: Port the deterministic reference feature code**

Create src/seclog/features.py:

~~~python
import re
import zlib

from .constants import HEX_RE, IP_RE, NUM_RE, PATH_RE, SEG_RE, TS_RE, WORD_RE


def hashed_bucket(text: str, mod: int) -> int:
    if mod < 2:
        raise ValueError('vocab_size must be at least 2')
    return (zlib.crc32(text.encode('utf-8')) % (mod - 1)) + 1


def clean_log_line(line: str) -> str:
    text = str(line).strip().lower()
    text = TS_RE.sub('', text)
    text = IP_RE.sub('<ip>', text)
    text = HEX_RE.sub('<hex>', text)
    text = PATH_RE.sub('<path>', text)
    text = SEG_RE.sub('<seg>', text)
    text = NUM_RE.sub('<num>', text)
    return re.sub(r'\s+', ' ', text)


def encode_log_line(line: str, vocab_size: int, max_tokens: int = 128) -> list[int]:
    text = clean_log_line(line)
    tokens: list[str] = []
    words = WORD_RE.findall(text)
    tokens.extend('w:' + word for word in words[:48])
    for left, right in zip(words[:32], words[1:33]):
        tokens.append('bg:' + left + '_' + right)
    chars = re.sub(r'\s+', '_', text)
    if len(chars) > 3:
        step = 3 if len(chars) > 80 else 2
        for width in (3, 4):
            for index in range(0, max(0, len(chars) - width + 1), step):
                tokens.append(f'c{width}:' + chars[index : index + width])
                if len(tokens) >= max_tokens:
                    break
            if len(tokens) >= max_tokens:
                break
    if not tokens:
        tokens = ['<empty>']
    return [hashed_bucket(token, vocab_size) for token in tokens[:max_tokens]]


def nonempty_log_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).split('\n') if line.strip()]
~~~

- [ ] **Step 5: Run feature tests and lint**

Run:

~~~powershell
pytest tests/test_features.py -v
ruff check src/seclog/features.py tests/test_features.py
~~~

Expected: PASS with no lint errors.

- [ ] **Step 6: Commit feature extraction**

~~~powershell
git add src/seclog/features.py tests/fixtures/feature_cases.json tests/test_features.py
git commit -m 'feat: preserve deterministic log features'
~~~

## Task 4: Build Samples, Labels, Manifests, and Batches

**Files:**
- Create: src/seclog/data.py
- Create: tests/test_data.py

- [ ] **Step 1: Write failing annotation and batching tests**

~~~python
import pandas as pd
import torch

from seclog.data import (
    Sample,
    build_endpoint_labels,
    build_sequence_labels,
    pack_log_batch,
    parse_annotation_spans,
)


def test_parse_annotation_spans_deduplicates_and_sorts() -> None:
    spans = parse_annotation_spans(
        '3|4|missing_step;1|2|timeout_retry;1|2|timeout_retry',
        has_anomaly=1,
        primary_start=1,
        primary_end=2,
        primary_type='timeout_retry',
    )
    assert spans == [(1, 2, 'timeout_retry'), (3, 4, 'missing_step')]


def test_sequence_and_endpoint_labels() -> None:
    spans = [(1, 2, 'timeout_retry')]
    sequence = build_sequence_labels(4, spans)
    starts, ends = build_endpoint_labels(4, spans)
    assert sequence.tolist() == [0, 1, 11, 0]
    assert starts[1, 0] == 1
    assert ends[2, 0] == 1


def test_pack_log_batch_shapes() -> None:
    sample = Sample(sid=7, lines=['a', 'b'], token_ids=[[1, 2], [3]])
    batch = pack_log_batch([sample])
    assert batch['input_ids'].tolist() == [1, 2, 3]
    assert tuple(batch['mask'].shape) == (1, 2)
    assert tuple(batch['pos_feats'].shape) == (1, 2, 10)
    assert batch['mask'].dtype == torch.bool
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~powershell
pytest tests/test_data.py -v
~~~

Expected: FAIL because data.py does not exist.

- [ ] **Step 3: Implement Sample and label construction**

Create src/seclog/data.py with these public types and functions:

~~~python
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
        for part in all_spans.split(';'):
            bits = part.strip().split('|')
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
~~~

- [ ] **Step 4: Port pack_log_batch exactly and add empty-sequence rejection**

Add pack_log_batch with the legacy tensor keys:

~~~python
def pack_log_batch(batch: list[Sample]) -> dict[str, Any]:
    if not batch or any(not sample.lines for sample in batch):
        raise ValueError('every batch and sample must contain at least one log line')
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
            pos_feats[batch_index, line_index, 8] = (
                math.log1p(line_index) / math.log1p(max(1, length))
            )
            pos_feats[batch_index, line_index, 9] = (
                math.log1p(length - 1 - line_index) / math.log1p(max(1, length))
            )
    return {
        'input_ids': input_ids,
        'offsets': offsets_tensor,
        'owner': owner,
        'mask': mask,
        'pos_feats': pos_feats,
        'labels': labels,
        'start_labels': start_labels,
        'end_labels': end_labels,
        'global_labels': global_labels,
        'samples': batch,
    }
~~~

- [ ] **Step 5: Add manifest-backed dataframe-to-sample loading**

Add:

~~~python
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
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
~~~

Add imports for encode_log_line, nonempty_log_lines, SchemaError, validate_test_frame, and validate_training_frame, then add:

~~~python
def _row_to_sample(
    row: pd.Series,
    is_training: bool,
    vocab_size: int,
    max_tokens: int,
) -> Sample:
    lines = nonempty_log_lines(row['log_text'])
    if not lines:
        raise SchemaError(f"id {row['id']} has no non-empty log lines")
    token_ids = [encode_log_line(line, vocab_size, max_tokens) for line in lines]
    if not is_training:
        return Sample(sid=int(row['id']), lines=lines, token_ids=token_ids)
    has_anomaly = int(row['has_anomaly'])
    spans = parse_annotation_spans(
        row.get('all_spans', ''),
        has_anomaly,
        int(row['primary_start_idx']),
        int(row['primary_end_idx']),
        str(row['primary_anomaly_type']),
    )
    labels = build_sequence_labels(len(lines), spans)
    start_labels, end_labels = build_endpoint_labels(len(lines), spans)
    primary = None
    global_label = GLOBAL_NONE_ID
    if has_anomaly:
        primary_type = str(row['primary_anomaly_type'])
        primary = (
            int(row['primary_start_idx']),
            int(row['primary_end_idx']),
            primary_type,
        )
        global_label = TYPE_TO_ID[primary_type]
    return Sample(
        sid=int(row['id']),
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
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
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
    cache_path = cache_dir / f'tokens-v{vocab_size}-m{max_tokens}.pkl'
    manifest_path = cache_path.with_suffix('.manifest.json')
    if cache_path.exists() and manifest_path.exists() and not rebuild_cache:
        cached_manifest = DatasetManifest(
            **json.loads(manifest_path.read_text(encoding='utf-8'))
        )
        if cached_manifest == manifest:
            with cache_path.open('rb') as handle:
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
    with cache_path.open('wb') as handle:
        pickle.dump((train_samples, test_samples), handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return train_frame, test_frame, train_samples, test_samples, manifest
~~~

Document beside load_dataset that pickle caches are trusted local artifacts and must never be loaded from an untrusted source.

- [ ] **Step 6: Run data tests**

Run:

~~~powershell
pytest tests/test_data.py -v
~~~

Expected: PASS.

- [ ] **Step 7: Commit the data layer**

~~~powershell
git add src/seclog/data.py tests/test_data.py
git commit -m 'feat: add manifest-backed sequence data layer'
~~~

## Task 5: Port the Multi-Task Network

**Files:**
- Create: src/seclog/model.py
- Create: tests/test_model.py

- [ ] **Step 1: Write failing model-shape and persistence tests**

~~~python
import torch

from seclog.data import Sample, pack_log_batch
from seclog.model import LogBoundaryNetwork


def make_batch() -> dict:
    return pack_log_batch(
        [
            Sample(sid=1, lines=['a', 'b'], token_ids=[[1], [2]]),
            Sample(sid=2, lines=['c'], token_ids=[[3]]),
        ]
    )


def test_model_output_shapes() -> None:
    batch = make_batch()
    model = LogBoundaryNetwork(vocab_size=64, emb_dim=8, hidden=12, num_layers=1, dropout=0)
    tag, start, end, global_logits = model(
        batch['input_ids'],
        batch['offsets'],
        batch['owner'],
        batch['mask'],
        batch['pos_feats'],
    )
    assert tuple(tag.shape) == (2, 2, 21)
    assert tuple(start.shape) == (2, 2, 10)
    assert tuple(end.shape) == (2, 2, 10)
    assert tuple(global_logits.shape) == (2, 11)


def test_state_dict_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    original = LogBoundaryNetwork(64, 8, 12, 1, 0)
    path = tmp_path / 'model.pt'
    torch.save(original.state_dict(), path)
    restored = LogBoundaryNetwork(64, 8, 12, 1, 0)
    restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    for left, right in zip(original.parameters(), restored.parameters()):
        assert torch.equal(left, right)
~~~

- [ ] **Step 2: Verify tests fail**

Run:

~~~powershell
pytest tests/test_model.py -v
~~~

Expected: FAIL because model.py does not exist.

- [ ] **Step 3: Port LogBoundaryNetwork without changing layer names**

Create src/seclog/model.py. Preserve these state-dict names exactly:

~~~python
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
    ) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, emb_dim, mode='mean', include_last_offset=False)
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
        self.tag_head = nn.Linear(hidden * 2, N_LABELS)
        self.start_head = nn.Linear(hidden * 2, N_TYPES)
        self.end_head = nn.Linear(hidden * 2, N_TYPES)
        self.global_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_TYPES + 1),
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
        if getattr(self, 'disable_cudnn_rnn', False) and features.is_cuda:
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
~~~

- [ ] **Step 4: Port class and boundary loss-weight helpers**

Add sequence_loss_weights and boundary_positive_weights from the reference, preserving:

- inverse-square-root sequence weighting;
- O-label multiplier;
- clipping to 0.05–8.0;
- positive boundary weights clipped to 3.0–25.0.

- [ ] **Step 5: Run model tests**

Run:

~~~powershell
pytest tests/test_model.py -v
~~~

Expected: PASS.

- [ ] **Step 6: Commit the model**

~~~powershell
git add src/seclog/model.py tests/test_model.py
git commit -m 'feat: port multi-task log boundary network'
~~~

## Task 6: Implement and Lock the Official Metrics

**Files:**
- Create: src/seclog/metrics.py
- Create: tests/test_metrics.py

- [ ] **Step 1: Write exact metric tests**

~~~python
import pandas as pd
import pytest

from seclog.metrics import evaluate_predictions, span_iou


def test_span_iou_is_inclusive() -> None:
    assert span_iou(1, 3, 2, 4) == pytest.approx(2 / 4)
    assert span_iou(-1, -1, 0, 1) == 0


def test_perfect_predictions_score_one() -> None:
    from seclog.constants import ANOMALY_TYPES

    types = ['none', *ANOMALY_TYPES]
    frame = pd.DataFrame(
        {
            'id': list(range(len(types))),
            'has_anomaly': [0, *([1] * len(ANOMALY_TYPES))],
            'primary_start_idx': [-1, *([2] * len(ANOMALY_TYPES))],
            'primary_end_idx': [-1, *([4] * len(ANOMALY_TYPES))],
            'primary_anomaly_type': types,
            'all_spans': ['', *[f'2|4|{name}' for name in ANOMALY_TYPES]],
        }
    )
    result = evaluate_predictions(frame, frame)
    assert result == {
        'detect_f1': 1.0,
        'iou': 1.0,
        'type_f1': 1.0,
        'score': 1.0,
    }
~~~

- [ ] **Step 2: Verify tests fail**

Run:

~~~powershell
pytest tests/test_metrics.py -v
~~~

Expected: FAIL because metrics.py does not exist.

- [ ] **Step 3: Implement the official composite**

Create src/seclog/metrics.py:

~~~python
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .constants import ANOMALY_TYPES


def span_iou(gold_start: int, gold_end: int, pred_start: int, pred_end: int) -> float:
    if min(gold_start, gold_end, pred_start, pred_end) < 0:
        return 0.0
    intersection = max(0, min(gold_end, pred_end) - max(gold_start, pred_start) + 1)
    union = max(gold_end, pred_end) - min(gold_start, pred_start) + 1
    return float(intersection / union) if union else 0.0


def evaluate_predictions(pred: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float]:
    if pred['id'].tolist() != gold['id'].tolist():
        raise ValueError('prediction and gold ids must match in the same order')
    detect_f1 = f1_score(
        gold['has_anomaly'],
        pred['has_anomaly'],
        average='macro',
        labels=[0, 1],
        zero_division=0,
    )
    both = gold['has_anomaly'].eq(1) & pred['has_anomaly'].eq(1)
    ious = [
        span_iou(
            int(gold.loc[index, 'primary_start_idx']),
            int(gold.loc[index, 'primary_end_idx']),
            int(pred.loc[index, 'primary_start_idx']),
            int(pred.loc[index, 'primary_end_idx']),
        )
        for index in gold.index[both]
    ]
    iou = float(np.mean(ious)) if ious else 0.0
    type_f1 = (
        f1_score(
            gold.loc[both, 'primary_anomaly_type'],
            pred.loc[both, 'primary_anomaly_type'],
            average='macro',
            labels=ANOMALY_TYPES,
            zero_division=0,
        )
        if both.any()
        else 0.0
    )
    score = 0.15 * detect_f1 + 0.50 * iou + 0.35 * type_f1
    return {
        'detect_f1': float(detect_f1),
        'iou': iou,
        'type_f1': float(type_f1),
        'score': float(score),
    }
~~~

- [ ] **Step 4: Run metric tests**

Run:

~~~powershell
pytest tests/test_metrics.py -v
~~~

Expected: PASS.

- [ ] **Step 5: Commit metrics**

~~~powershell
git add src/seclog/metrics.py tests/test_metrics.py
git commit -m 'feat: lock official anomaly metrics'
~~~

## Task 7: Port and Regression-Test the Decoder

**Files:**
- Create: src/seclog/decode.py
- Create: src/seclog/tuning.py
- Create: tests/test_decode.py

- [ ] **Step 1: Write core decoder tests**

~~~python
import numpy as np

from seclog.decode import constrained_viterbi, spans_from_bio_path, stable_softmax


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
    assert spans[0][:3] == (1, 2, 'timeout_retry')


def test_softmax_rows_sum_to_one() -> None:
    result = stable_softmax(np.array([[1000.0, 1001.0]], dtype=np.float32))
    assert np.allclose(result.sum(axis=1), 1.0)
~~~

- [ ] **Step 2: Verify decoder tests fail**

Run:

~~~powershell
pytest tests/test_decode.py -v
~~~

Expected: FAIL because decode.py does not exist.

- [ ] **Step 3: Port the decoder in three focused sections**

Create src/seclog/decode.py and port these reference functions without semantic changes:

1. Core math and BIO functions: tag_to_type, build_bio_transition, constrained_viterbi, stable_softmax, bounded_sigmoid, spans_from_bio_path.
2. Span selection and refinement: add_endpoint_candidates, polish_span_boundary, pick_primary_span, parse_type_filter, maybe_adjust_primary_span, decode_single_item.
3. Frame functions in decode.py: decode_logits_to_frame and samples_to_truth_frame.
4. Tuning functions in tuning.py: make_decoder_grid, make_v3_local_grid, make_v4_local_grid, make_stratified_eval_slice, and tune_decoder.

Use explicit imports from constants.py, data.py, and metrics.py. Do not copy training, CLI, file-path, or dataframe-schema code into decode.py.

- [ ] **Step 4: Add structured-output regression cases**

Add these cases to tests/test_decode.py:

~~~python
from seclog.decode import decode_single_item


def prediction_with_tags(tag_ids: list[int]) -> dict[str, np.ndarray]:
    tag = np.full((len(tag_ids), 21), -10.0, dtype=np.float32)
    for index, tag_id in enumerate(tag_ids):
        tag[index, tag_id] = 10.0
    global_logits = np.full(11, -10.0, dtype=np.float32)
    global_logits[0] = 10.0
    return {
        'tag': tag,
        'start': np.zeros((len(tag_ids), 10), dtype=np.float32),
        'end': np.zeros((len(tag_ids), 10), dtype=np.float32),
        'global': global_logits,
    }


def decoder_params() -> dict[str, object]:
    return {
        'min_conf': 0.0,
        'bridge_gap': 0,
        'refine_radius': 0,
        'use_boundary_candidates': False,
        'fallback_global_boundary': False,
        'post_adjust_mode': 'none',
    }


def test_decode_single_item_returns_normal_sentinels() -> None:
    decoded = decode_single_item(
        prediction_with_tags([0, 0, 0]),
        n_lines=3,
        params=decoder_params(),
        length_stats={},
    )
    assert decoded == {
        'has_anomaly': 0,
        'primary_start_idx': -1,
        'primary_end_idx': -1,
        'primary_anomaly_type': 'none',
        'all_spans': '',
    }


def test_decode_single_item_returns_inclusive_timeout_span() -> None:
    decoded = decode_single_item(
        prediction_with_tags([1, 11, 0]),
        n_lines=3,
        params=decoder_params(),
        length_stats={'timeout_retry': {'p95': 3}},
    )
    assert decoded == {
        'has_anomaly': 1,
        'primary_start_idx': 0,
        'primary_end_idx': 1,
        'primary_anomaly_type': 'timeout_retry',
        'all_spans': '0|1|timeout_retry',
    }
~~~

- [ ] **Step 5: Run decoder tests and inspect file size**

Run:

~~~powershell
pytest tests/test_decode.py -v
(Get-Content src/seclog/decode.py).Count
(Get-Content src/seclog/tuning.py).Count
~~~

Expected: PASS. Neither focused file exceeds 700 lines.

- [ ] **Step 6: Commit the decoder**

~~~powershell
git add src/seclog/decode.py src/seclog/tuning.py tests/test_decode.py
git commit -m 'feat: port constrained anomaly decoder'
~~~

## Task 8: Add Typed Configuration and Historical Presets

**Files:**
- Create: src/seclog/config.py
- Create: configs/final.yaml
- Create: configs/smoke.yaml
- Create: tests/test_config.py

- [ ] **Step 1: Write failing configuration tests**

~~~python
from seclog.config import load_config


def test_smoke_config_is_cpu_sized() -> None:
    config = load_config('configs/smoke.yaml')
    assert config.model.hidden == 16
    assert config.training.folds == 2
    assert config.training.epochs == 1


def test_final_config_matches_selected_run() -> None:
    config = load_config('configs/final.yaml')
    assert config.model.emb_dim == 128
    assert config.model.hidden == 176
    assert config.training.seed == 999
    assert config.training.folds == 5
~~~

- [ ] **Step 2: Verify tests fail**

Run:

~~~powershell
pytest tests/test_config.py -v
~~~

Expected: FAIL because config.py and YAML files do not exist.

- [ ] **Step 3: Implement frozen configuration dataclasses**

Create ModelConfig, TrainingConfig, FeatureConfig, and ProjectConfig dataclasses. load_config must:

- reject unknown top-level sections;
- reject vocab_size below 2;
- reject folds below 2;
- reject non-positive epochs and batch sizes;
- retain the raw decoder mapping for parity with the historical parameter file.

Use this complete configuration loader:

~~~python
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    emb_dim: int
    hidden: int
    layers: int
    dropout: float


@dataclass(frozen=True)
class FeatureConfig:
    max_tokens: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    seeds: list[int]
    folds: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    o_weight: float
    boundary_loss_weight: float
    global_loss_weight: float
    patience: int


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    features: FeatureConfig
    training: TrainingConfig
    decoder: dict[str, object]


def _section(cls, name: str, payload: object):
    if not isinstance(payload, dict):
        raise ValueError(f'{name} must be a mapping')
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f'{name} contains unknown fields: {unknown}')
    return cls(**payload)


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('configuration root must be a mapping')
    expected = {'model', 'features', 'training', 'decoder'}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(f'configuration sections invalid; missing={missing}, unknown={unknown}')
    model = _section(ModelConfig, 'model', raw['model'])
    features = _section(FeatureConfig, 'features', raw['features'])
    training = _section(TrainingConfig, 'training', raw['training'])
    decoder = raw['decoder']
    if not isinstance(decoder, dict):
        raise ValueError('decoder must be a mapping')
    if model.vocab_size < 2:
        raise ValueError('vocab_size must be at least 2')
    if training.folds < 2:
        raise ValueError('folds must be at least 2')
    if min(training.epochs, training.batch_size, training.eval_batch_size) <= 0:
        raise ValueError('epochs and batch sizes must be positive')
    return ProjectConfig(model, features, training, dict(decoder))
~~~

- [ ] **Step 4: Add the selected historical configuration**

configs/final.yaml must encode:

~~~yaml
model:
  vocab_size: 262144
  emb_dim: 128
  hidden: 176
  layers: 1
  dropout: 0.24
features:
  max_tokens: 64
training:
  seed: 999
  seeds: [999]
  folds: 5
  epochs: 5
  batch_size: 48
  eval_batch_size: 64
  learning_rate: 0.0018
  weight_decay: 0.0001
  o_weight: 0.16
  boundary_loss_weight: 0.20
  global_loss_weight: 0.12
  patience: 2
decoder:
  anomaly_bias: -0.25
  min_conf: 0.35
  bridge_gap: 1
  edge_prob: 0.10
  switch_penalty: -1.8
  continue_bonus: 0.20
  min_len: 1
  hard_max_len: 18
  long_span_conf: 0.58
  primary_strategy: earliest_score
  refine_radius: 1
  boundary_weight: 0.45
  global_weight: 0.10
  length_weight: 0.08
  use_boundary_candidates: false
  post_adjust_mode: short_expand
  post_short_max_len: 2
  post_short_max_conf: 1.01
  post_neighbor_support: 0.0
  post_expand_left: 0
  post_expand_right: 1
~~~

Use this configs/smoke.yaml:

~~~yaml
model:
  vocab_size: 512
  emb_dim: 8
  hidden: 16
  layers: 1
  dropout: 0.0
features:
  max_tokens: 16
training:
  seed: 7
  seeds: [7]
  folds: 2
  epochs: 1
  batch_size: 4
  eval_batch_size: 4
  learning_rate: 0.001
  weight_decay: 0.0
  o_weight: 0.16
  boundary_loss_weight: 0.20
  global_loss_weight: 0.12
  patience: 1
decoder:
  anomaly_bias: 0.0
  min_conf: 0.0
  bridge_gap: 0
  edge_prob: 0.10
  switch_penalty: -1.8
  continue_bonus: 0.20
  min_len: 1
  hard_max_len: 18
  long_span_conf: 0.60
  primary_strategy: earliest
  refine_radius: 0
  use_boundary_candidates: false
  fallback_global_boundary: false
  post_adjust_mode: none
~~~

- [ ] **Step 5: Run configuration tests**

Run:

~~~powershell
pytest tests/test_config.py -v
~~~

Expected: PASS.

- [ ] **Step 6: Commit configuration**

~~~powershell
git add src/seclog/config.py configs tests/test_config.py
git commit -m 'feat: add typed experiment configuration'
~~~

## Task 9: Refactor Training with a CPU Smoke Path

**Files:**
- Create: src/seclog/training.py
- Create: tests/test_training.py
- Create: tests/fixtures/synthetic_data/train.csv
- Create: tests/fixtures/synthetic_data/test.csv

- [ ] **Step 1: Create a balanced synthetic dataset**

Create tests/fixtures/synthetic_data/train.csv exactly as:

~~~csv
id,log_text,has_anomaly,primary_start_idx,primary_end_idx,primary_anomaly_type,all_spans
1,"service boot completed
worker ready",0,-1,-1,none,""
2,"scheduler initialized
queue empty",0,-1,-1,none,""
3,"storage mounted
health check passed",0,-1,-1,none,""
4,"configuration loaded
listener started",0,-1,-1,none,""
5,"request started
attempt timed out
retry scheduled
request completed",1,1,2,timeout_retry,"1|2|timeout_retry"
6,"job accepted
deadline exceeded
retry queued
job resumed",1,1,2,timeout_retry,"1|2|timeout_retry"
7,"worker started
memory usage critical
allocation failed
worker stopped",1,1,2,resource_exhaustion,"1|2|resource_exhaustion"
8,"batch opened
descriptor pool exhausted
new handles rejected
batch aborted",1,1,2,resource_exhaustion,"1|2|resource_exhaustion"
~~~

Create tests/fixtures/synthetic_data/test.csv exactly as:

~~~csv
id,log_text
101,"service boot completed
worker ready"
102,"request started
attempt timed out
retry scheduled"
~~~

These logs are fabricated for testing and contain no copied competition content.

- [ ] **Step 2: Write a failing one-epoch smoke test**

~~~python
from pathlib import Path

from seclog.config import load_config
from seclog.training import run_training


def test_cpu_smoke_training_writes_two_folds(tmp_path: Path) -> None:
    result = run_training(
        data_dir=Path('tests/fixtures/synthetic_data'),
        output_dir=tmp_path,
        config=load_config('configs/smoke.yaml'),
        device_name='cpu',
    )
    assert len(result.checkpoints) == 2
    assert result.oof_path.exists()
    assert result.metrics_path.exists()
~~~

- [ ] **Step 3: Verify the smoke test fails**

Run:

~~~powershell
pytest tests/test_training.py::test_cpu_smoke_training_writes_two_folds -v
~~~

Expected: FAIL because training.py does not exist.

- [ ] **Step 4: Port training helpers with stable interfaces**

Create src/seclog/training.py and port:

- fix_all_seeds with deterministic mode enabled for tests;
- choose_torch_device and amp_enabled;
- build_dataloader and batch_to_device;
- create_model;
- compute_training_loss;
- train_epoch and validate_epoch_loss;
- collect_model_outputs and average_logits;
- train_fold_and_collect;
- checkpoint save/load with config metadata.

Use these result types:

~~~python
@dataclass(frozen=True)
class FoldCheckpoint:
    fold: int
    seed: int
    path: Path


@dataclass(frozen=True)
class TrainingResult:
    checkpoints: tuple[FoldCheckpoint, ...]
    oof_path: Path
    metrics_path: Path
    manifest_path: Path
~~~

run_training must accept explicit data_dir, output_dir, config, and device_name. It must never search the current working directory for data.

- [ ] **Step 5: Add checkpoint compatibility metadata**

Each checkpoint payload must contain:

~~~python
{
    'state_dict': model.state_dict(),
    'model_config': asdict(config.model),
    'feature_config': asdict(config.features),
    'labels': list(ANOMALY_TYPES),
    'fold': fold,
    'seed': seed,
    'data_manifest_sha256': manifest_hash,
}
~~~

Loading must reject changed labels, feature configuration, or model dimensions before load_state_dict.

- [ ] **Step 6: Run the smoke test and the full fast suite**

Run:

~~~powershell
pytest tests/test_training.py -v
pytest -q -m 'not private'
~~~

Expected: PASS on CPU.

- [ ] **Step 7: Commit training**

~~~powershell
git add src/seclog/training.py tests/test_training.py tests/fixtures/synthetic_data
git commit -m 'feat: add reproducible fold training'
~~~

## Task 10: Add Inference, Ensembling, and Submission Validation

**Files:**
- Create: src/seclog/inference.py
- Create: tests/test_inference.py

- [ ] **Step 1: Write failing inference tests**

~~~python
from pathlib import Path

import pandas as pd

from seclog.config import load_config
from seclog.inference import predict
from seclog.training import run_training


def test_smoke_prediction_preserves_test_ids(tmp_path: Path) -> None:
    config = load_config('configs/smoke.yaml')
    training = run_training(
        Path('tests/fixtures/synthetic_data'),
        tmp_path / 'train',
        config,
        'cpu',
    )
    output = predict(
        test_path=Path('tests/fixtures/synthetic_data/test.csv'),
        checkpoint_paths=[item.path for item in training.checkpoints],
        config=config,
        output_path=tmp_path / 'submission.csv',
        device_name='cpu',
    )
    test_ids = pd.read_csv('tests/fixtures/synthetic_data/test.csv')['id'].tolist()
    assert output['id'].tolist() == test_ids
    assert (tmp_path / 'submission.csv').exists()
~~~

- [ ] **Step 2: Verify inference test fails**

Run:

~~~powershell
pytest tests/test_inference.py -v
~~~

Expected: FAIL because inference.py does not exist.

- [ ] **Step 3: Implement explicit checkpoint loading and fold averaging**

Create src/seclog/inference.py with:

~~~python
def load_checkpoint_model(
    checkpoint_path: Path,
    config: ProjectConfig,
    device: torch.device,
) -> LogBoundaryNetwork:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_metadata(payload, config)
    model = LogBoundaryNetwork(
        vocab_size=config.model.vocab_size,
        emb_dim=config.model.emb_dim,
        hidden=config.model.hidden,
        num_layers=config.model.layers,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model
~~~

Implement predict to:

1. validate and tokenize the explicit test_path;
2. load every requested fold;
3. collect fold logits;
4. average tag, start, end, and global logits;
5. decode using config.decoder;
6. validate output schema, row count, and ID order;
7. reject non-finite logits and degenerate all-one-class results unless allow_degenerate=True;
8. write UTF-8 CSV only after validation passes.

- [ ] **Step 4: Run inference and compatibility tests**

Run:

~~~powershell
pytest tests/test_inference.py -v
~~~

Expected: PASS.

- [ ] **Step 5: Commit inference**

~~~powershell
git add src/seclog/inference.py tests/test_inference.py
git commit -m 'feat: add validated fold inference'
~~~

## Task 11: Add Data Audit, Locked Evaluation, and Private Reproduction

**Files:**
- Create: scripts/build_private_manifest.py
- Create: scripts/run_private_reproduction.py
- Create: src/seclog/splitting.py
- Create: tests/test_splitting.py
- Create: artifacts/metrics/historical-results.json

- [ ] **Step 1: Write split and duplicate-detection tests**

~~~python
import pandas as pd

from seclog.splitting import add_template_groups, make_locked_split


def test_identical_normalized_logs_share_group() -> None:
    frame = pd.DataFrame(
        {
            'id': [1, 2],
            'log_text': [
                '2026-01-01 00:00:00 retry id=123',
                '2026-02-02 00:00:00 retry id=999',
            ],
            'has_anomaly': [1, 1],
            'primary_anomaly_type': ['timeout_retry', 'timeout_retry'],
        }
    )
    grouped = add_template_groups(frame)
    assert grouped.loc[0, 'template_group'] == grouped.loc[1, 'template_group']


def test_locked_split_has_disjoint_groups() -> None:
    frame = pd.DataFrame(
        {
            'id': list(range(8)),
            'has_anomaly': [0, 0, 0, 0, 1, 1, 1, 1],
            'primary_anomaly_type': [
                'none',
                'none',
                'none',
                'none',
                'timeout_retry',
                'timeout_retry',
                'resource_exhaustion',
                'resource_exhaustion',
            ],
            'template_group': [f'group-{index}' for index in range(8)],
        }
    )
    tuning, locked = make_locked_split(frame, test_size=0.25, random_state=20260711)
    assert set(tuning['template_group']).isdisjoint(set(locked['template_group']))
~~~

- [ ] **Step 2: Implement normalized-template grouping**

src/seclog/splitting.py must:

- normalize each log line with clean_log_line;
- concatenate normalized lines;
- SHA256 the result into template_group;
- use StratifiedGroupKFold or GroupShuffleSplit;
- stratify on has_anomaly plus primary_anomaly_type when sample counts allow;
- emit an audit table containing duplicate-group sizes and split membership.

- [ ] **Step 3: Add historical results with explicit labels**

Create artifacts/metrics/historical-results.json:

~~~json
{
  "official_b_leaderboard": {
    "score": 0.87111,
    "source": "ISCC 2026 final score table"
  },
  "historical_competition_oof": {
    "detect_f1": 1.0,
    "iou": 0.9604749303490352,
    "type_f1": 0.9977536895161062,
    "score": 0.9794512565051547,
    "warning": "Decoder parameters were selected using OOF predictions; this is not a locked estimate."
  }
}
~~~

- [ ] **Step 4: Add the private manifest script**

build_private_manifest.py accepts --train, --test, and --output. It calls schema validation, computes SHA256, records row counts and label distributions, and writes JSON. It must not print log_text values.

- [ ] **Step 5: Add the private reproduction script**

run_private_reproduction.py accepts:

~~~text
--data-dir
--config configs/final.yaml
--output-dir
--device auto
--locked-seed 20260711
~~~

It must:

1. create template groups;
2. record duplicate statistics;
3. construct decoder-tuning and locked groups;
4. run or load fold predictions;
5. tune only on the tuning group;
6. score the locked group once;
7. write reproduction-metrics.json and reproduction-manifest.json;
8. never copy source CSV files into the repository.

- [ ] **Step 6: Run split tests and a manifest smoke command**

Run:

~~~powershell
pytest tests/test_splitting.py -v
python scripts/build_private_manifest.py --train tests/fixtures/synthetic_data/train.csv --test tests/fixtures/synthetic_data/test.csv --output .private/smoke-manifest.json
~~~

Expected: PASS and a private ignored manifest.

- [ ] **Step 7: Commit audit and reproduction code**

~~~powershell
git add src/seclog/splitting.py scripts tests/test_splitting.py artifacts/metrics/historical-results.json
git commit -m 'feat: add locked evaluation and data audit'
~~~

## Task 12: Add the Command-Line Interface

**Files:**
- Create: src/seclog/cli.py
- Create: tests/test_cli.py

- [ ] **Step 1: Write failing CLI help and check-data tests**

~~~python
import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'seclog.cli', *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_stable_commands() -> None:
    result = run_cli('--help')
    assert result.returncode == 0
    assert 'check-data' in result.stdout
    assert 'train' in result.stdout
    assert 'predict' in result.stdout
    assert 'evaluate' in result.stdout


def test_check_data_accepts_synthetic_fixture() -> None:
    result = run_cli(
        'check-data',
        '--train',
        'tests/fixtures/synthetic_train.csv',
        '--test',
        'tests/fixtures/synthetic_test.csv',
    )
    assert result.returncode == 0
    assert 'schema: OK' in result.stdout
~~~

- [ ] **Step 2: Implement argparse subcommands**

Create CLI handlers:

- check-data: validate schemas and print hashes, row counts, and distributions.
- train: require --data-dir, --config, --output-dir, and optional --device.
- predict: require --test, --config, repeated --checkpoint, and --output.
- evaluate: require --prediction and --gold, print JSON metrics.

All paths must be explicit. No command may search parent folders for train.csv or test.csv.

- [ ] **Step 3: Run CLI tests**

Run:

~~~powershell
pytest tests/test_cli.py -v
seclog --help
~~~

Expected: PASS and the four commands appear.

- [ ] **Step 4: Commit the CLI**

~~~powershell
git add src/seclog/cli.py tests/test_cli.py
git commit -m 'feat: add explicit training and inference CLI'
~~~

## Task 13: Build the Local Streamlit Demonstration

**Files:**
- Create: app/streamlit_app.py
- Create: src/seclog/presentation.py
- Create: tests/test_presentation.py
- Create: examples/synthetic_logs/normal.json
- Create: examples/synthetic_logs/timeout_retry.json

- [ ] **Step 1: Write failing presentation tests**

~~~python
from seclog.presentation import annotate_lines


def test_annotate_lines_marks_only_inclusive_span() -> None:
    result = annotate_lines(['a', 'b', 'c'], start=1, end=2)
    assert [item.is_anomalous for item in result] == [False, True, True]
    assert [item.line_number for item in result] == [0, 1, 2]
~~~

- [ ] **Step 2: Implement UI-independent presentation objects**

~~~python
from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayLine:
    line_number: int
    text: str
    is_anomalous: bool


def annotate_lines(lines: list[str], start: int, end: int) -> list[DisplayLine]:
    return [
        DisplayLine(index, text, start <= index <= end)
        for index, text in enumerate(lines)
    ]
~~~

- [ ] **Step 3: Add fabricated examples**

Create examples/synthetic_logs/normal.json:

~~~json
{
  "name": "Normal worker lifecycle",
  "lines": [
    "service boot completed",
    "configuration loaded",
    "worker registered",
    "health check passed"
  ]
}
~~~

Create examples/synthetic_logs/timeout_retry.json:

~~~json
{
  "name": "Timeout followed by retry",
  "lines": [
    "request accepted",
    "remote operation timed out",
    "retry scheduled after backoff",
    "request completed"
  ]
}
~~~

These examples are fabricated and contain no copied competition logs, real IPs, credentials, personal names, or institution names.

- [ ] **Step 4: Implement the Streamlit shell**

The app must:

- show a prototype/non-production notice;
- allow a synthetic-example selection or pasted text;
- limit input to 400 lines and 200,000 characters;
- never execute input;
- accept checkpoint paths from a local environment variable;
- show a clear unavailable-model message when no private checkpoint is configured;
- display line numbers, highlighted predicted span, anomaly type, and confidence;
- keep inference logic in seclog.inference rather than the UI file.

- [ ] **Step 5: Run tests and manually launch the app**

Run:

~~~powershell
pytest tests/test_presentation.py -v
streamlit run app/streamlit_app.py
~~~

Expected: tests pass; the app opens with synthetic examples and a clear checkpoint status.

- [ ] **Step 6: Commit the demo**

~~~powershell
git add app src/seclog/presentation.py tests/test_presentation.py examples
git commit -m 'feat: add local anomaly visualization demo'
~~~

## Task 14: Write Portfolio Documentation and Resume Material

**Files:**
- Create: README.md
- Create: LICENSE
- Create: docs/architecture.md
- Create: docs/experiments.md
- Create: docs/error-analysis.md
- Create: docs/model-card.md
- Create: docs/iscc-2026-retrospective.md
- Create: docs/resume-and-interview.md

- [ ] **Step 1: Write README result language before adding visuals**

README must state:

- individual ISCC 2026 entry;
- official final rank 277 without an award claim;
- official system-log B-score 0.87111;
- historical OOF 0.9794512565 labeled as non-locked;
- problem, architecture, installation, synthetic smoke run, local demo, results, limitations, and repository map;
- competition data and checkpoints are not distributed.

- [ ] **Step 2: Document architecture and data flow**

architecture.md must describe each module, tensor shapes, the three prediction heads, fold ensembling, decoder flow, and explicit private/public boundary.

- [ ] **Step 3: Document experiments and error analysis**

experiments.md must include the verified historical run table:

- baseline/early 0.9708086 OOF;
- ensemble experiment 0.9800900 OOF;
- seed 3407 experiment 0.9790349 OOF;
- selected seed 999 experiment 0.9794513 OOF;
- official B score 0.87111.

error-analysis.md must summarize the retained OOF categories:

- exact/normal successes;
- boundary shifts;
- bad boundaries;
- wrong types;
- false positives.

Explain that boundary localization is the dominant remaining error.

- [ ] **Step 4: Write the model card**

model-card.md must cover intended use, non-production status, training-data non-distribution, architecture, metrics, limitations, data drift, possible near-duplicate leakage, security considerations, and no warranty.

- [ ] **Step 5: Write the four-task retrospective**

Use the verified table:

| Task | Official B-score | Portfolio treatment |
|---|---:|---|
| Binary vulnerability detection | 0.13407 | experiment and failure analysis |
| System-log anomaly detection | 0.87111 | flagship project |
| PowerShell script detection | 0.66585 | structured-feature ensemble case |
| Network-event classification | 0.71049 | multi-seed and pseudo-label case |

Describe A/B dataset version confusion and the lessons that led to checksums, manifests, fail-fast validation, and output-distribution checks.

- [ ] **Step 6: Write verified resume bullets and an interview script**

resume-and-interview.md must provide:

- a two-bullet concise Chinese resume version;
- a four-bullet detailed project version;
- a 90-second interview explanation;
- likely questions about rank, non-award status, OOF/B-score gap, model choice, data leakage, and failure lessons;
- answers that do not exaggerate ownership, generalization, or production readiness.

- [ ] **Step 7: Add the MIT code license with data exclusion**

LICENSE contains the standard MIT text. README and model-card must state that the license covers repository code only and grants no rights to competition data.

- [ ] **Step 8: Check documentation claims**

Run:

~~~powershell
Get-ChildItem README.md,docs -Recurse -File | Select-String -Pattern '获奖|国三|top\s*\d+%|0\.97945' -CaseSensitive:$false
~~~

Expected: every 0.97945 reference says historical/non-locked OOF; no text claims an award, national third prize, or unsupported top percentage.

- [ ] **Step 9: Commit documentation**

~~~powershell
git add README.md LICENSE docs
git commit -m 'docs: add portfolio narrative and model card'
~~~

## Task 15: Run Full Verification and Publication Audit

**Files:**
- Create: scripts/audit_publication.py
- Create: tests/test_publication_audit.py
- Modify: README.md

- [ ] **Step 1: Write publication-audit tests**

~~~python
from pathlib import Path

from scripts.audit_publication import audit_paths


def test_audit_rejects_private_dataset_name(tmp_path: Path) -> None:
    bad = tmp_path / 'train.csv'
    bad.write_text('id,log_text\n1,secret\n', encoding='utf-8')
    findings = audit_paths(tmp_path)
    assert any('train.csv' in finding for finding in findings)
~~~

- [ ] **Step 2: Implement the audit**

audit_publication.py must fail for:

- files larger than 10 MiB;
- .pt, .pth, .pkl, .joblib, .npy, and .npz files;
- train.csv or test.csv outside tests/fixtures;
- absolute Windows user paths;
- email-like secrets outside documentation examples;
- ISCC participant IDs matching ISCC2026-XSTZ;
- common credential names and private keys;
- staged files ignored by the approved public boundary.

It must print file paths and reasons, not file contents.

- [ ] **Step 3: Run the complete local verification**

Run:

~~~powershell
ruff check src tests app scripts
pytest --cov=seclog --cov-report=term-missing
python scripts/audit_publication.py .
git diff --check
git status --short
~~~

Expected:

- Ruff passes.
- All tests pass.
- Coverage is at least 80% for schemas, features, metrics, and presentation modules.
- Publication audit reports zero findings.
- Git diff check reports no whitespace errors.
- Worktree is clean after the verification commit.

- [ ] **Step 4: Run the private reproduction checkpoint**

Run the approved reproduction command against the private data path, writing outputs under .private/reproduction. Review:

- dataset hashes and row counts;
- duplicate/template-group statistics;
- tuning/locked separation;
- reproduced metrics;
- prediction distributions;
- no raw data copied into tracked paths.

If full GPU training is impractical, reuse verified private checkpoints only after checkpoint metadata and data manifest match. Record the limitation in experiments.md.

- [ ] **Step 5: Add only small verified metric artifacts**

Copy reproduction-metrics.json and a redacted reproduction-manifest.json into artifacts/metrics. The redacted manifest may contain hashes, row counts, configuration, code revision, and metrics; it must not contain local paths, IDs, or raw logs.

- [ ] **Step 6: Re-run verification and commit**

~~~powershell
git add scripts/audit_publication.py tests/test_publication_audit.py artifacts/metrics README.md docs
git commit -m 'test: add publication safety audit'
ruff check src tests app scripts
pytest -q
python scripts/audit_publication.py .
git status --short
~~~

Expected: all checks pass and the worktree is clean.

## Task 16: Create, Review, and Publish the GitHub Repository

**Files:**
- No new local files unless the remote audit identifies a missing repository setting.

- [ ] **Step 1: Confirm final local history and repository size**

Run:

~~~powershell
git log --oneline --decorate --max-count=30
git count-objects -vH
git status --short
~~~

Expected: coherent task-sized commits, repository size well below 50 MiB, clean worktree.

- [ ] **Step 2: Verify the repository-local GitHub identity**

Confirm that repository-local user.name and user.email still match the connected GitHub identity configured in Task 1. Do not change global Git configuration. The two planning commits remain transparently attributed to Codex Planning.

- [ ] **Step 3: Create a private GitHub repository**

Create security-log-anomaly-localization as private with no auto-generated README, license, or .gitignore because those files already exist locally. Add the remote and push main.

- [ ] **Step 4: Verify the private remote**

Confirm:

- repository visibility is private;
- main is the default branch;
- CI runs and passes;
- no large files, datasets, participant IDs, local paths, or secrets appear in the web UI or Git history.

- [ ] **Step 5: Perform a private repository review**

Read the repository as a first-time internship reviewer. Confirm that the README explains the problem, results, architecture, demo, limitations, and setup in under three minutes. Fix only clarity or safety issues, re-run tests, and push.

- [ ] **Step 6: Make the repository public**

Change visibility to public only after CI and the private audit pass. Verify the public URL in a logged-out browser session.

- [ ] **Step 7: Final release check**

Create a v0.1.0 release only if README, tests, demo screenshots, metric artifacts, model card, and retrospective are present. The release must not attach competition-trained checkpoints or datasets.

- [ ] **Step 8: Record the final handoff**

Report:

- public repository URL;
- verified test and CI status;
- official and local metrics with labels;
- exact resume bullets;
- limitations and any deferred work;
- recommended next iteration: binary-model failure study or a public-dataset generalization experiment.

---

## Execution Order and Checkpoints

Implementation uses inline execution with the superpowers:executing-plans skill unless the user explicitly requests delegated subagent execution.

Checkpoint A after Task 6:

- package installs;
- schemas, features, batching, model shapes, and official metrics pass.

Checkpoint B after Task 10:

- decoder, configuration, CPU smoke training, and fold inference pass.

Checkpoint C after Task 13:

- locked-evaluation tooling, CLI, and local demo work with synthetic data.

Checkpoint D after Task 15:

- documentation, private reproduction record, and publication audit pass.

External publication begins only at Task 16.

## Spec Coverage Index

| Approved design requirement | Implemented by plan tasks |
|---|---|
| Focused system-log flagship and explicit non-goals | Tasks 1, 14 |
| Modular package architecture | Tasks 1–12 |
| Legacy feature and model behavior preservation | Tasks 3–10 |
| Explicit private/public data boundary | Tasks 1, 4, 11, 15, 16 |
| Official, historical OOF, and reproduced metric separation | Tasks 6, 11, 14, 15 |
| Duplicate and near-duplicate leakage controls | Task 11 |
| Schema, hash, A/B version, checkpoint, and output validation | Tasks 2, 4, 8, 10, 11, 12 |
| Unit, integration, smoke-training, and CI tests | Tasks 1–13, 15 |
| Local non-production visualization | Task 13 |
| Four-task retrospective and honest non-award language | Task 14 |
| Model card, experiment report, error analysis, resume, and interview material | Task 14 |
| Private-first GitHub publication and public-history audit | Tasks 15, 16 |
| Version 1 acceptance criteria and release handoff | Tasks 15, 16 |

No approved Version 1 requirement is deferred outside this plan. Binary-model repair and a hosted public inference service remain explicit non-goals.
