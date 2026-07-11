# BGL → Thunderbird normal-only transfer audit

This is a zero-shot **false-positive audit**, not an anomaly-detection leaderboard. Loghub's fixed `Thunderbird_2k.log` sample contains 2,000 normal lines only, so recall, F1, PR-AUC, and ROC-AUC are undefined and intentionally omitted.

## Protocol

- Source model: `bgl-2k-random-neural-weighted`, trained on the official BGL 2k sample with a fixed random split.
- Target: Loghub `Thunderbird_2k.log`, SHA256 `903bbfa61c34d4803e4adcb0d726ff2eeb9a2e11971243269a2035fa6c3bbeb0`.
- Target preparation: first 2,000 source lines, 63 non-overlapping 32-line span windows, all officially labelled non-alert/normal.
- The target data were not used for model fitting, temperature scaling, or threshold selection. The BGL validation temperature (`0.3646332368608554`) and threshold (`0.07188967956263675`) were reused unchanged.

## Result

| Target windows | Target lines | False-positive windows | Window FPR | False-positive lines | Line FPR | Predicted spans |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 63 | 2,000 | 63 | 1.0000 | 340 | 0.1700 | 131 |

The result is deliberately retained because it demonstrates that a threshold learned on BGL does not transfer safely to a different supercomputer log format. It motivates target-domain calibration or adaptation; it is not evidence that the model generalises across systems.
