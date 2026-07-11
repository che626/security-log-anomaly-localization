# HDFS 100k public benchmark results

This directory contains aggregate results only—no raw logs, prepared samples, checkpoints, or detailed predictions.

## Provenance

- Dataset: LogPai Loglizer `HDFS_100k.log_structured.csv` with the paired HDFS `anomaly_label.csv`.
- Public source lineage: Loghub HDFS v1 labels; see `docs/public-benchmark.md` for citation and licence notes.
- Raw source SHA256:
  - structured log: `0b18e7d3e0bd991d3fc5d1c8b6a2a172fa268fd533d93a06c8ffe90130f4f213`
  - labels: `78de297d3862e37295951484028c1749bee4e3f06c2d563242c9d1999ff1abb0`
- Prepared task: `sequence_binary`; 104,815 log lines grouped into 7,940 block sequences, of which 313 are anomalous.
- Seed: `20260711`. Features, thresholds, temperature calibration, and early stopping use train/validation data only.

## Interpretation

These are HDFS 100k-subset results, not a claim about the full 11M-line HDFS corpus. They show why split choice matters: the neural model has strong ranking metrics under random and template-isolated splits, but its fixed validation-selected threshold loses F1 under unseen templates. The linear baseline has the highest random F1. This is retained as an explicit limitation and interview discussion point rather than hidden.

| Split | Neural F1 | Neural PR-AUC | Neural ROC-AUC | Best F1 baseline/model |
| --- | ---: | ---: | ---: | --- |
| Random | 0.6383 | 0.6570 | 0.9044 | TF-IDF + logistic regression (0.6598) |
| Chronological | 0.5693 | 0.5536 | 0.7767 | Isolation Forest (0.6713) |
| Template-isolated | 0.4138 | 0.4978 | 0.8886 | Decision tree (0.5075) |

The exact per-model measurements are in the three CSV files. Each SVG is generated from the matching CSV by `seclog public-report`.
