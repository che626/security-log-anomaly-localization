# BGL 2k public integration result

This is a small official `BGL_2k.log` sample committed by Loghub, not the full BGL corpus. It is retained as an end-to-end integration and span-localisation sanity check only.

- Source SHA256: `2a819ea540909db682005c9cf948387a40729b5c2e9f19d430e29ce704825496`
- 2,000 source lines → 63 non-overlapping 32-line windows; 21 anomalous windows.
- Profile: `span_binary`; fixed random split, seed `20260711`; test partition has 13 windows and 11 gold spans.
- The small data size and simple alert-prefix labels explain why TF-IDF is the strongest line-F1 baseline. These numbers are not used as a generalisation claim.

| Model | Line F1 | Span F1 | Mean inclusive IoU |
| --- | ---: | ---: | ---: |
| Rarity | 0.0958 | 0.1091 | 0.2153 |
| TF-IDF + logistic regression | 0.8571 | 0.5333 | 0.8750 |
| Decision tree | 0.0000 | 0.0000 | 0.0000 |
| Isolation Forest | 0.3077 | 0.5806 | 0.7704 |
| CNN + BiGRU | 0.2936 | 0.1622 | 0.3583 |

The CNN + BiGRU row uses the current train-only global class weighting implementation. Its fixed validation threshold is intentionally reused in the Thunderbird normal-only transfer audit rather than retuned on the target data.
