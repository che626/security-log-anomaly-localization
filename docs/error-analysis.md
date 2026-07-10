# Error Analysis

## Retained OOF categories

The private competition workspace retained examples and summaries that can be organized into five categories. Raw competition logs are not reproduced here.

### Exact and normal successes

Many normal samples and correctly localized anomaly sequences were classified exactly. Historical OOF detection and type metrics were close to saturation, showing that the shared sequence representation learned strong signals within the competition folds.

### Boundary shifts

The most common remaining pattern was a start or end line shifted by a small amount. Typical cases included omitting the recovery line after a timeout, starting at the warning immediately before the actual failure, or including one contextual line too many. Because span IoU has 50% weight in the composite score, these small shifts matter disproportionately.

### Bad boundaries

Some spans had substantial overlap errors even when the anomaly type was correct. Long, repetitive sequences and partial recovery loops were especially difficult: local tag evidence could stay elevated after the causal event, while endpoint heads preferred a later line.

### Wrong types

Type confusion was less frequent in the retained OOF results but remained possible for behaviorally similar patterns, such as timeout/retry versus partial recovery loop, or resource exhaustion versus slow-burn warning. Global type logits and local BIO evidence can disagree; the decoder currently combines them heuristically rather than through calibrated joint inference.

### False positives

Repetitive but legitimate retries, warnings followed by successful recovery, and uncommon normal templates can resemble anomalies. The anomaly bias, minimum confidence and global fallback parameters trade false positives against missed anomalies.

## Main conclusion

Boundary localization is the dominant observed weakness. This is why the project keeps separate BIO, start and end heads, retains inclusive-index regression tests, and exposes decoder tuning as a distinct stage. However, decoder complexity can itself overfit OOF predictions, so better boundaries on historical OOF do not automatically imply better external performance.

## Recommended next experiments

1. Evaluate on a public log-anomaly dataset with a separately locked template split.
2. Calibrate detection and type confidence on a validation partition not used for decoder selection.
3. Compare the heuristic decoder with a learned semi-Markov or span-ranking objective.
4. Report metrics by anomaly type, span length and template novelty rather than only one composite score.
5. Add adversarial normal examples containing retries and warnings without failures.
