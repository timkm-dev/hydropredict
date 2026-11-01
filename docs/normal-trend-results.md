# Normal-trained trend results

22 October 2026. **The fixed trend experiment did not improve fault detection.**
It ran locally against the original checksum-verified Colab inputs; another Colab
run is unnecessary. The frozen LSTM and dashboard remain unchanged. GitHub access
was unavailable, so no issue was read, updated or closed.

## Result

All three methods share 4,638 score endpoints, including 4,251 publisher-selected
nominal-reference endpoints. Two faults meet matched coverage: February 3 and
April 2 at 09:06:14 UTC. Neither was warned about by any method.

| Method | Eligible faults warned | Nominal points flagged | Nominal episodes |
| --- | ---: | ---: | ---: |
| Frozen LSTM on matching endpoints | 0 / 2 | 33 / 4,251 = 0.78% | 11 |
| Normal-trained trends | 0 / 2 | 0 / 4,251 = 0% | 0 |
| Combined, with persistence | 0 / 2 | 4 / 4,251 = 0.09% | 3 |

Lower alert burden without a recovered fault does not meet the declared success
criterion. Neither trend nor combined scoring is retained as a promising model.
In particular, this experiment does not reproduce the retrospective February 3
hybrid warning under its fixed normal-calibration rule.

On its native endpoints, the frozen LSTM warns before 1/3 adequately scored
faults, with 80/5,904 nominal-reference readings flagged (1.36%). Its February 2
warning is unavailable to the matched comparison: the strict six-hour trend
warm-up leaves only 6.94% score-slot coverage before that event. Do not interpret
the matched LSTM 0/2 as a regression in its unchanged native predictions.

## Integrity and reproduction

The [protocol](normal-trend-protocol.md) was written before scoring. Fitting uses
17,744 earlier normal trend rows; separate calibration uses 4,731 rows. Maximum
absolute standardized trend deviation is calibrated at the normal 99th percentile,
not tuned to any fault. Combined scoring and three-reading persistence follow
the same fixed protocol. All nine event outcomes and availability exclusions are
retained. No final-test observations were evaluated.

The canonical run is `models/normal-trend-2026-10-22-v2`. The first local run is
preserved in `models/normal-trend-2026-10-22-v1`; v2 reruns the same protocol after
formatting and input-validation/type improvements, with identical summary results.
Four targeted tests pass: irregular-time slopes and prefix invariance, outage
warm-up, rejection of unordered data, and persistence resets. Lint, formatting
and strict type checking of the new feature module pass. Runtime checks verify
input hashes, prefix invariance and replay of serialized fitting statistics.

```powershell
.venv/Scripts/python.exe scripts/evaluate_normal_trends.py --output models/normal-trend-new-run
```

Run artifacts include model statistics and thresholds, input/source identities,
calibration and development scores, summaries, every event, and checksums.
Only aggregate summary/event evidence is copied into `docs/results`; telemetry
and local model artifacts remain outside committed documentation.

## Decision

Stop this trend experiment. Keep the LSTM as the development baseline and do not
replace its alerts with the trend or combined rules. Further supervised detection
work needs more usable earlier fault examples and better operating/maintenance
context; repeating threshold searches on February–April is not confirmation.
The application can continue with baseline scoring, saved alerts and explicit
insufficient-history states while its limited detection performance is reported.
