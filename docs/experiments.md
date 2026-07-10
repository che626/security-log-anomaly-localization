# Experiments

## Verified historical records

These values were recovered from the private competition archive. They describe experiments from the competition period, not fresh public-data benchmarks.

| Run | Score | Label |
|---|---:|---|
| Baseline / early run | 0.9708086 | Historical OOF, non-locked |
| Ensemble experiment | 0.9800900 | Historical OOF, non-locked |
| Seed 3407 experiment | 0.9790349 | Historical OOF, non-locked |
| Selected seed 999 experiment | 0.9794513 | Historical OOF, non-locked; selected configuration |
| ISCC 2026 system-log B leaderboard | 0.87111 | Official external leaderboard |
| Locked decoder evaluation | 0.9783049 | Verified legacy OOF; decoder selection locked only |

The selected run's retained components were `detect_f1=1.0`, `iou=0.9604749303490352`, `type_f1=0.9977536895161062`, producing historical non-locked OOF `0.9794512565051547`. Decoder parameters were selected using OOF predictions, so this value is optimistic as an estimate of unseen performance.

## Selected configuration

`configs/final.yaml` records the selected seed 999 setup: vocabulary 262,144; 128-dimensional embeddings; hidden size 176; one bidirectional GRU layer; five folds; five epochs; batch size 48; and decoder boundary refinement. The file preserves the historical settings; it does not claim they are universally optimal.

## Why OOF and B leaderboard differ

The evidence supports several plausible contributors rather than one proven cause:

- decoder choices were evaluated on the same OOF predictions used for selection;
- normalized near-duplicate logs may cross ordinary random folds;
- A/B dataset versions and caches were mixed in the original workspace;
- the external test distribution may contain different templates, lengths or type frequencies;
- a high type score conditional on detected anomalies does not guarantee robust boundary localization.

The refactor responds with SHA256 manifests, cache/config coupling, checkpoint compatibility metadata, normalized-template groups and an explicit decoder-locked partition.

## Reproduction protocol

Public CI uses fabricated logs and validates code behavior, including real CPU backpropagation. A private reproduction requires authorized `train.csv` and `test.csv` files:

```bash
python scripts/run_private_reproduction.py \
  --data-dir /explicit/private/data \
  --config configs/final.yaml \
  --output-dir .private/reproduction \
  --device auto \
  --locked-seed 20260711
```

The process hashes the data, groups normalized templates, creates decoder-tuning and locked groups, produces OOF logits, tunes only on the tuning group and scores the locked group once. Outputs remain under the ignored `.private` directory until a redacted, manually reviewed metric artifact is intentionally added.

## Verified private reproduction checkpoint

The July 2026 checkpoint reused the trusted historical seed 999 OOF artifact after matching the full run arguments to `configs/final.yaml` and validating sample ID coverage, log-line counts, head names, tensor shapes and finite values. It did not copy competition logs or retrain the five folds.

| Item | Value |
|---|---:|
| Training rows / test rows | 20,000 / 5,000 |
| Normalized template groups | 20,000 |
| Exact duplicate template groups | 0 |
| Decoder-tuning rows | 15,999 |
| Locked decoder-evaluation rows | 4,001 |
| Locked detect F1 | 1.0 |
| Locked inclusive span IoU | 0.9588109329548771 |
| Locked type F1 | 0.9968556634242637 |
| Locked composite | 0.9783049486759308 |

This is a locked **decoder-selection** estimate, not a fully locked end-to-end model estimate: each row's OOF prediction excludes that row from its fold training, but the fold model may still train on other rows assigned to the decoder-locked partition. The result is therefore reported below the official B score in evidential strength, despite its larger numerical value.

## Current limitation

A full fresh private-data training run remains hardware- and data-dependent and is not represented as completed. The verified checkpoint reused historical OOF rather than regenerating the five models. Any future end-to-end result must include dataset hashes, code revision, configuration, split audit and the precise distinction between official, historical OOF, decoder-locked and fully locked estimates.
