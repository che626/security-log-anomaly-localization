# Model Card

## Model summary

Security Log Anomaly Localization is a prototype multi-task neural model for ordered system-log sequences. It detects whether a sequence is anomalous, predicts an inclusive primary span and assigns one of ten competition-defined anomaly types.

## Intended use

- educational analysis of sequence classification and span localization;
- reproducible portfolio demonstrations on fabricated or properly authorized logs;
- offline experimentation with explicit human review.

It is not intended for production monitoring, automatic incident response, malware verdicts, employee surveillance or decisions affecting access and safety.

## Architecture

Deterministically hashed word, bigram and character features feed an EmbeddingBag, convolutional context layers and a bidirectional GRU. BIO, start, end and global type heads are trained jointly. Fold logits are averaged and passed to a constrained structured decoder.

## Data

The original training data came from the ISCC 2026 competition and is not distributed. This repository contains only fabricated fixtures. The code license grants no right to competition data or separately obtained model weights.

The repository does not establish that the private competition data represents real operational environments. Unknown collection processes, template imbalance and near duplicates may affect reported results.

## Metrics

- Official system-log B leaderboard: `0.87111`.
- Selected historical, non-locked OOF: `0.9794512565`; decoder parameters were selected using OOF predictions.
- Locked decoder evaluation on verified historical OOF: `0.9783049487`; this isolates decoder selection only and is not a fully locked end-to-end estimate.
- Public synthetic tests: behavioral correctness and CPU pipeline execution, not a meaningful accuracy benchmark.

Detection macro F1 receives 15%, inclusive span IoU 50%, and anomaly-type macro F1 35% in the composite implementation.

## Limitations

- Logs from unseen software, languages or formats may tokenize and behave differently.
- Hash collisions discard lexical identity and may merge unrelated tokens.
- Timestamps, numbers, paths and IPs are normalized; useful magnitude or identity signals may be lost.
- Confidence is not calibrated and should not be read as a real-world probability.
- Template grouping reduces obvious duplication but cannot detect all semantic near duplicates.
- Historical OOF may be optimistic due to decoder selection and data-version issues.
- An attacker could craft text that triggers known templates or exploits hash collisions.

## Security and privacy considerations

Input is treated as text and never executed. The demo escapes rendered content and limits input size. Operators must still remove secrets and personal data before using logs. Checkpoints and pickle caches are local trusted artifacts: never load a checkpoint or pickle from an untrusted source. Model output can expose properties of supplied logs, so inference results should follow the same access controls as inputs.

## Evaluation and drift

Before any operational trial, create an environment-specific, time-separated test set; audit template overlap; measure false positives; calibrate confidence; and monitor log-format and type-frequency drift. Re-evaluate after parser, service or deployment changes.

## Warranty

The software is provided under the MIT license without warranty. The license applies to repository code only and grants no rights to competition data, third-party datasets or separately trained checkpoints.
