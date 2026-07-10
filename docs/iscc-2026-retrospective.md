# ISCC 2026 Data Security Retrospective

## Contest outcome

I entered the four-task data-security track as the individual team `che`. More than four thousand registrations were shown during the event; the B leaderboard later contained over four hundred active entries. My B leaderboard rank was 278, and the official final list recorded rank 277. The national third-prize cutoff shown in the published list ended at rank 233, so this entry did not receive an award.

The value of the work is therefore not a prize claim. It is the experiment history, the engineering lessons and the ability to explain why local validation did not fully transfer to the official leaderboard.

## Four-task results

| Task | Official B score | Portfolio treatment |
|---|---:|---|
| Binary vulnerability detection | 0.13407 | Experiment and failure analysis |
| System-log anomaly detection | 0.87111 | Flagship reproducible project |
| PowerShell malicious-script detection | 0.66585 | Structured-feature ensemble case |
| Network-security event classification | 0.71049 | Multi-seed and pseudo-label case |

## 1. Binary vulnerability detection

The archive contains PE/opcode/CFG, TF-IDF, rule, GRU, BiGRU and graph-neural-network experiments. The official score of 0.13407 shows that breadth of experimentation did not become a robust solution. The likely lessons are more valuable than presenting a weak number as success: establish a reliable baseline first, verify file/sample alignment, isolate representation changes, and use error analysis before increasing model complexity.

Portfolio decision: preserve this task as a failure study rather than the repository headline. A future iteration should use a legal public binary dataset and compare static representations under one locked split.

## 2. System-log anomaly detection

This was the strongest official result and the clearest match to a data-science/ML internship. The solution combines deterministic feature hashing, multi-task sequence learning and structured boundary decoding. Historical OOF was much higher than the official B result, making the task useful for discussing leakage, distribution shift and evaluation design—not only architecture.

Portfolio decision: refactor it into this focused repository with tests, manifests, synthetic examples, a local demo and explicit official/non-locked metric labels.

## 3. PowerShell malicious-script detection

The private archive records a best local OOF around 0.74983 and an official B score of 0.66585. The work explored structured/script features and ensembles. The transfer gap is smaller than in the system-log task but still argues for stronger split design and calibration.

Portfolio decision: describe it as a secondary ensemble case in interviews; do not dilute the main GitHub repository with unrelated code until it has its own clean data and reproducibility story.

## 4. Network-security event classification

The training table contained 53,477 rows and 12 classes. Experiments included LightGBM, multiple seeds and pseudo labels; the official B score was 0.71049. This task demonstrates tabular multiclass modeling, but pseudo labeling creates additional confirmation-bias and distribution-check requirements.

Portfolio decision: retain it as evidence of tabular modeling experience and a possible next standalone project using a public dataset.

## What went wrong operationally

The original workspace mixed A/B dataset versions, multiple caches, outputs and parameter files. That makes a good score difficult to attribute and a failed score difficult to diagnose. The refactor turns those mistakes into explicit controls:

| Competition lesson | Repository control |
|---|---|
| A/B files can be confused | SHA256 data manifests and explicit paths |
| Token cache can outlive its data | Cache key plus matching manifest |
| Checkpoint can use another shape/config | Model, feature and label metadata validation |
| Random folds can share templates | Normalized-template group audit |
| Decoder can overfit OOF | Separate decoder-tuning and locked groups |
| Submission can be misaligned | ID order, row count and sentinel validation |
| Broken model can output one class | Degenerate-output guard |

## What I would do differently

1. Freeze one data manifest before the first serious experiment.
2. Establish one reproducible baseline and one locked validation set.
3. Track every run with code revision, seed, configuration and prediction distribution.
4. Inspect errors by template novelty and span length before adding architectures.
5. Stop model expansion when validation evidence is not trustworthy.

This retrospective intentionally reports the non-award outcome and all four official B scores. It does not infer an unsupported percentile or equate registration count with final competitive rank.
