# Security Log Anomaly Localization — Portfolio Project Design

Date: 2026-07-10
Status: Approved 2026-07-11; implementation not started

## 1. Purpose

Turn the strongest part of the ISCC 2026 data-security submission into a focused, reproducible portfolio project for data-science and machine-learning internships.

The public-facing project will be titled **Security Log Anomaly Localization**. It will detect whether a system-log sequence is anomalous, locate the primary anomalous span, and classify the anomaly type. The repository will also contain a concise retrospective covering the other three ISCC tasks.

The project must demonstrate modeling depth, experimental discipline, engineering reliability, and honest reporting. It must not imply that the competition entry won an award.

## 2. Evidence and Source Assets

The design is based on read-only inspection of the following private local assets. Absolute paths are intentionally omitted so they cannot enter public Git history:

- System-log experiment archive: checkpoints, OOF predictions, decoder searches, and B-test submission.
- Network-event classification archive: model and submission experiments.
- PowerShell experiment archive: model experiments and score manifests.
- Binary-analysis archives: static-analysis experiments and separate A/B data.
- Final model-submission archive: the packaged code, models, and generated submissions.

Verified competition facts to report:

| Task | Official B-score |
|---|---:|
| Binary vulnerability detection | 0.13407 |
| System-log anomaly detection | 0.87111 |
| PowerShell malicious-script detection | 0.66585 |
| Network-security event classification | 0.71049 |

The B leaderboard showed rank 278. The official results list later showed rank 277, with third prize ending at rank 233. Public materials may state `ISCC 2026 Data Security Contest, individual entry, official final rank 277`; they must not state or imply an award.

The system-log experiments record historical OOF composite scores from approximately 0.97081 to 0.98009. The selected final configuration records 0.9794512565. These are local validation results and must always be labeled separately from the official B-score of 0.87111.

## 3. Intended Audience

Primary audience:

- Data-science internship reviewers.
- Machine-learning engineering internship reviewers.
- Faculty or laboratory mentors evaluating undergraduate project ability.

The README should be understandable in under three minutes. Deeper implementation and experimental details will live in separate documentation.

## 4. Scope

### 4.1 Version 1 deliverables

1. A clean Python package for system-log anomaly detection, span localization, and anomaly-type classification.
2. A reproducible configuration for the selected competition model.
3. A strict evaluation pipeline that separates official B-score, historical OOF score, and newly reproduced validation results.
4. Unit and integration tests, including data-version and schema validation.
5. A local Streamlit demonstration that highlights predicted anomalous log lines and displays anomaly type and confidence.
6. Synthetic log examples that are safe to publish.
7. An experiment report, error analysis, model card, and four-task ISCC retrospective.
8. Resume bullets and an interview explanation derived from verified project results.
9. A clean GitHub repository created privately first and made public only after a publication audit.

### 4.2 Explicit non-goals for Version 1

- Do not refactor all four competition solutions into production packages.
- Do not retrain or repair the binary-vulnerability model in Version 1.
- Do not publish ISCC datasets, participant lists, raw leaderboard files, large caches, virtual environments, or original model checkpoints.
- Do not claim production security effectiveness from competition-only data.
- Do not optimize for another leaderboard submission.
- Do not create a hosted public inference service that requires publishing competition-trained weights.

## 5. Repository Architecture

```text
security-log-anomaly-localization/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── final.yaml
│   └── smoke.yaml
├── src/seclog/
│   ├── __init__.py
│   ├── schemas.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── infer.py
│   └── cli.py
├── app/
│   └── streamlit_app.py
├── examples/
│   └── synthetic_logs/
├── tests/
│   ├── test_schemas.py
│   ├── test_features.py
│   ├── test_decode.py
│   ├── test_metrics.py
│   ├── test_inference.py
│   └── test_smoke_training.py
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   ├── error-analysis.md
│   ├── model-card.md
│   └── iscc-2026-retrospective.md
└── artifacts/
    └── metrics/
```

Each module has one responsibility:

- `schemas.py`: input/output contracts and validation.
- `data.py`: dataset loading, version manifests, split construction, and batching.
- `features.py`: normalization and hashed token construction.
- `model.py`: PyTorch architecture only.
- `train.py`: cross-validation, optimization, checkpointing, and OOF generation.
- `decode.py`: Viterbi decoding, boundary refinement, and span construction.
- `evaluate.py`: official metric implementation and evaluation reports.
- `infer.py`: checkpoint loading, fold ensembling, and prediction.
- `cli.py`: stable user-facing commands.

## 6. Model Design

The refactor will preserve the selected competition model before considering any algorithmic change:

1. Normalize timestamps, IP addresses, paths, numbers, hexadecimal values, and other volatile tokens.
2. Construct word, bigram, and character n-gram tokens with deterministic hashing.
3. Aggregate tokens into line representations with `EmbeddingBag`.
4. Apply convolutional layers for local context.
5. Apply a bidirectional GRU for sequence context.
6. Produce three outputs:
   - BIO-style line tags for anomalous spans.
   - Start/end boundary probabilities.
   - Global anomaly-type probabilities.
7. Ensemble fold predictions.
8. Decode spans with Viterbi and boundary refinement.
9. Return anomaly presence, primary span, anomaly type, all spans, and confidence values.

The final competition configuration is the reference behavior. Refactoring must not silently change preprocessing, label ordering, tensor shapes, decoder semantics, or submission columns.

## 7. Data Flow

```text
CSV or in-memory log samples
  -> schema and version validation
  -> normalized log lines
  -> deterministic hashed tokens
  -> padded/packed sequence batches
  -> EmbeddingBag + CNN + BiGRU
  -> tag, boundary, and global-type probabilities
  -> fold ensemble
  -> Viterbi and boundary refinement
  -> structured predictions
  -> metrics, CSV output, or local visualization
```

Training data and private checkpoints stay outside the Git repository. Their locations are supplied through explicit local configuration or environment variables.

## 8. Evaluation Design

### 8.1 Metrics

The official composite metric is:

```text
score = 0.15 * anomaly-detection Macro-F1
      + 0.50 * primary-span IoU
      + 0.35 * anomaly-type Macro-F1
```

The project will report all three components and the composite score.

### 8.2 Reporting categories

Results must be grouped into three clearly labeled categories:

1. **Official B leaderboard:** 0.87111.
2. **Historical competition OOF:** selected record 0.9794512565.
3. **Portfolio reproduction:** produced by the cleaned pipeline and never filled in from historical assumptions.

### 8.3 Leakage controls

Historical decoder parameters were tuned using OOF predictions, so the historical OOF score may be optimistic. The cleaned project will:

- Detect exact duplicate samples using content hashes.
- Derive normalized-template signatures to identify near-duplicate log families.
- Use group-aware stratified splitting when meaningful groups exist.
- Reserve a deterministic locked subset for decoder selection assessment.
- Keep model fitting, decoder tuning, and locked evaluation records separate.
- Record seeds, row IDs, data hashes, configuration hashes, and code revision for every reported run.

The project will not promise that the new validation score matches the historical OOF score. A lower but more defensible score is acceptable.

## 9. Reliability and Error Handling

The pipeline must fail fast for:

- Missing required columns.
- Duplicate or malformed IDs.
- Empty datasets or empty log sequences.
- Unknown labels or changed label order.
- A/B dataset checksum mismatch.
- Missing checkpoint folds.
- Checkpoint/config incompatibility.
- Invalid span boundaries.
- Non-finite probabilities or losses.
- Output row-count or ID-order mismatch.

Before any submission-style CSV is written, validation will check:

- Exact schema and encoding.
- Row count and ID order.
- Allowed anomaly types.
- Span bounds and normal-sample sentinel values.
- Prediction distributions and degenerate all-one-class output.

The demo will treat logs as plain text. It will not execute commands or scripts embedded in log content. Input size and line count will be bounded.

## 10. Testing Strategy

### 10.1 Unit tests

- Normalization of timestamps, IPs, paths, numbers, and hexadecimal values.
- Stable hashing across processes.
- Label encoding and padding masks.
- Viterbi transitions and boundary refinement.
- Span IoU and composite metrics.
- Schema and dataset-manifest failures.

### 10.2 Integration tests

- End-to-end inference with deterministic synthetic inputs.
- Save/load parity for a small checkpoint.
- Fold-ensemble shape and label-order consistency.
- CLI output matching the declared schema.
- Streamlit inference adapter using synthetic data.

### 10.3 Smoke training

A tiny synthetic dataset and reduced model configuration will complete at least one training epoch on CPU. This validates installation and control flow without distributing competition data.

### 10.4 Internal reproduction

Full-data reproduction will run only against the private local dataset. The resulting metrics and configuration manifest may be published; the underlying data and private checkpoints will not be committed.

## 11. Demo Design

The local Streamlit demo will provide:

- A text area or synthetic-example selector.
- Line-numbered log display.
- Highlighting for the predicted primary anomaly span.
- Anomaly presence, type, and confidence summary.
- A compact probability view.
- A notice that the model is a competition research prototype, not a production security control.

The public repository will support the interface and include screenshots or a short GIF. It will not include the original competition-trained checkpoints. A synthetic smoke checkpoint may be used only to verify the interface and must be labeled as non-representative.

## 12. Four-Task Retrospective

`docs/iscc-2026-retrospective.md` will cover:

- The four tasks, datasets, metrics, and verified B-scores.
- The final rank and non-award status stated accurately.
- The strongest methods tried for each task.
- A/B data-version confusion and the need for manifests.
- Why the system-log task became the flagship.
- Why the binary task is presented as an experiment/failure analysis rather than a performance claim.
- Concrete lessons about validation, reproducibility, leaderboard overfitting, and submission safety.

The retrospective will not contain participant personal information or raw competition data.

## 13. Publication and Licensing

1. Develop locally in a new clean repository; never initialize Git in the multi-gigabyte submission archive.
2. Create the GitHub repository as private.
3. Commit only refactored code, tests, synthetic examples, configuration, documentation, and small metric artifacts.
4. Run secret, personal-information, large-file, data-license, and history audits.
5. Add an MIT license for the original/refactored code only, with an explicit statement that competition data is excluded.
6. Make the repository public only after all acceptance checks pass.

GitHub access will be used after the implementation plan reaches its publication step. No remote repository is created during design or planning.

## 14. Acceptance Criteria

Version 1 is complete only when all of the following are true:

- A fresh Python 3.10 or 3.11 environment can install the package.
- Unit, integration, and CPU smoke-training tests pass.
- Synthetic examples run through CLI inference and the local demo.
- Internal private-data reproduction produces a recorded, labeled metric report.
- Historical OOF and official B scores are never conflated.
- README, architecture, experiments, error analysis, model card, and retrospective are complete.
- The public Git history contains no competition dataset, participant information, secrets, virtual environments, cache files, or large checkpoints.
- The README communicates the problem, approach, results, limitations, and demo in under three minutes.
- Resume bullets and an interview explanation use only verified claims.

## 15. Implementation Boundary

Approval of this design authorizes creation of a detailed implementation plan only. It does not yet authorize code refactoring, model training, GitHub repository creation, or publication. Those actions begin after the implementation plan is written and reviewed.
