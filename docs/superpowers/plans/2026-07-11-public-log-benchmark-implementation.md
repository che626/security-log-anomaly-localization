# Public Log Benchmark — Implementation Plan

## Execution contract

This plan implements the approved public-log benchmark design in a way that leaves the existing ISCC competition path unchanged. Raw third-party data stays outside Git. Every public run is represented by a local manifest, a configuration, aggregate metrics, and reproducible report inputs.

## Milestone 1 — Public data protocol and deterministic preparation

### 1. Add a public task contract

Create `src/seclog/public_protocol.py` with:

- `TaskProfile` values `sequence_binary` and `span_binary`;
- immutable prepared-sample and span structures;
- profile validation that rejects type labels for binary public tasks;
- serialisation helpers for aggregate prepared CSV/JSONL artifacts;
- a manifest structure recording source paths, SHA256 values, source metadata, task profile, subset rules, preparation options, and record counts.

Tests in `tests/test_public_protocol.py` must cover valid binary tasks, invalid spans, missing source metadata, deterministic serialisation, and rejection of public multiclass labels.

### 2. Add Loghub adapters

Create `src/seclog/public_data.py` with explicit adapters for HDFS, OpenStack, BGL, and Thunderbird. Each adapter accepts local source paths only, validates the documented raw schema before parsing, and produces a profile-specific prepared dataset.

- HDFS: groups log lines by the source block/session identifier and joins official anomaly labels.
- OpenStack: groups records according to the official supplied grouping/label artefacts and records the exact mapping in the manifest.
- BGL and Thunderbird: preserve chronological raw-line order, parse source anomaly flags, and produce windows with binary contiguous spans.
- Thunderbird: requires an explicit, deterministic start/end or count subset rule; it refuses implicit sampling.

Use the existing normalisation primitives to calculate a stable template key, but retain raw text only in local prepared data. The adapters must fail rather than guess when official labels or group keys are missing.

Commit only synthetic raw fixtures under `tests/fixtures/public_logs/`. Tests must prove HDFS grouping, span construction, multi-span boundaries, deterministic Thunderbird subset selection, malformed-label failures, and no source-line reuse across emitted windows.

### 3. Add leakage-safe public splits

Create `src/seclog/public_splitting.py` for fixed random, chronological, and template-isolated train/validation/test partitions. Split assignment occurs at source-group or source-line level before any overlapping windows are emitted. The module must serialize an assignment file and report partition sizes and positive rates.

Tests must verify determinism, group disjointness, chronological ordering, template isolation, and explicit errors when a requested split has insufficient class support.

## Milestone 2 — Comparable models and evaluation

### 4. Implement public baseline runners

Create `src/seclog/public_baselines.py` containing a shared train/validation/test runner for:

- template-frequency / rarity scoring;
- TF-IDF plus logistic regression;
- template-statistics plus decision tree;
- Isolation Forest.

All fitting, vocabulary generation, score normalisation, and threshold selection are restricted to the training/validation partitions. Span baselines derive line scores, select a validation threshold, and convert contiguous positives to inclusive spans. Persist only compact model metadata and aggregate predictions/metrics outside Git.

Tests must show that threshold selection cannot read test labels and that normal-only or anomaly-sparse data produces a clear error or defined zero-division-safe result.

### 5. Generalise the neural architecture without changing the ISCC path

Keep `LogBoundaryNetwork` as the encoder implementation. Add a public-profile wrapper, `src/seclog/public_model.py`, that supplies a binary global head for `sequence_binary` and binary BIO/start/end heads for `span_binary`. The wrapper reuses the existing line feature path, CNN, BiGRU, pooling, and constrained-span logic, but does not reuse the ten ISCC anomaly classes as a public label.

Create `src/seclog/public_training.py` for profile-masked losses, train-only class weighting, early stopping, checkpoint metadata, deterministic seeds, and batch prediction. Checkpoint metadata must declare `public_profile`, label vocabulary, model dimensions, and dataset-manifest hash so an ISCC checkpoint and a public checkpoint cannot be mixed.

Tests must prove that disabled heads cannot contribute loss or metrics, output shapes match each profile, public and ISCC checkpoint metadata are incompatible, and a small synthetic profile trains end-to-end on CPU.

### 6. Add metrics, calibration, and report aggregation

Create `src/seclog/public_metrics.py` and `src/seclog/public_reporting.py`.

- Sequence metrics: precision, recall, F1, PR-AUC, ROC-AUC when defined, false-positive rate, and confusion matrix fields.
- Span metrics: line/token PRF, span PRF, inclusive IoU, and exact-boundary accuracy.
- Calibration: validation-only temperature scaling or a documented no-op when calibration cannot be fit safely.
- Resource fields: device, elapsed seconds, dataset counts, and peak GPU memory when available.
- Report aggregation: reject rows with mixed profiles, manifests, or split definitions; generate compact CSV/JSON tables and static figures from aggregate results.

Tests must cover edge cases including no positives, no negatives, empty predicted spans, incompatible report rows, and calibration leakage.

## Milestone 3 — CLI, configurations, documentation, and quality gates

### 7. Extend the CLI with explicit public commands

Add commands to `src/seclog/cli.py`:

- `public-prepare` — validates local raw paths and writes prepared data plus manifest;
- `public-split` — writes a named split assignment;
- `public-run-baseline` — trains/evaluates one baseline under one fixed experiment configuration;
- `public-train` — trains/evaluates one neural profile;
- `public-report` — aggregates result records and writes tables/figures.

Every command must require explicit input/output paths and fail if an output would silently combine incompatible runs. No command downloads external data automatically.

### 8. Add checked-in configurations and smoke fixtures

Add `configs/public/` configurations for HDFS sequence detection, OpenStack sequence detection, BGL span localisation, and Thunderbird subset transfer. Add CPU smoke configurations with very small epochs and fixtures. Configurations name a profile, split strategy, seed, feature/model dimensions, calibration setting, and resource limits.

### 9. Update portfolio surfaces

Update `README.md`, `docs/architecture.md`, `docs/model-card.md`, `docs/resume-and-interview.md`, and add `docs/public-benchmark.md`.

- Keep the official ISCC score/ranking visually separate from public results.
- Explain dataset licences/citations and non-redistribution.
- Add a public-result table only after runs are completed; before then state that figures are pending rather than inventing values.
- Include reproducibility commands, task-profile explanation, leakage controls, limitations, and evidence-backed interview material.
- Restrict the Streamlit change to an aggregate-results and synthetic-example view.

### 10. Run quality gates

Run formatting, static checks, full tests, public-data fixture tests, CPU end-to-end smoke commands, and the publication-safety audit. Review `git diff`, tracked files, and `.gitignore` to ensure no raw data, cache, checkpoints, or prediction detail has entered the repository.

## Milestone 4 — Real benchmark runs and publication

### 11. Obtain source data locally and record provenance

Download or clone the allowed Loghub source data into the ignored local `data/` directory. Verify each source format against the adapter and record SHA256 data manifests. Do not commit raw data.

### 12. Produce Phase 1 evidence

Prepare HDFS and OpenStack, generate all supported split assignments, run the four baselines and neural model, calibrate only from validation data, and generate reports. Re-run the fixed configuration once for the final test metric. Record exact commands, device, runtime, and limitations.

### 13. Produce Phase 2 evidence

Prepare BGL and a deterministic Thunderbird subset, run binary span localisation and the BGL-to-Thunderbird transfer matrix. If the complete Thunderbird source cannot be obtained or parsed locally, retain the fully tested adapter/configuration and clearly label Phase 2 as pending rather than fabricating results.

### 14. Final portfolio handoff

Populate only verified result tables, produce the final recruiter-friendly README/report/Streamlit view, run every quality gate again, audit the Git status and publication boundary, make local commits, and push only after the repository contains no restricted files. Prepare final resume bullets and interview scripts using real result values.

## Verification commands

At the end of each milestone, run the applicable subset of:

```powershell
pytest -q
ruff check .
seclog --help
seclog public-prepare --help
seclog public-split --help
seclog public-run-baseline --help
seclog public-train --help
seclog public-report --help
python scripts/publication_audit.py
git status --short
```

The final audit must also prove that result rows have manifests, configurations, and real files behind every public number; tests alone are insufficient evidence for a portfolio claim.
