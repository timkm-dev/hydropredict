# CARE score of the existing A/B models

1 November 2026. Related issue #21. Replaces the project's own screening gate
with the benchmark's published metric, as agreed with the project owner: no
external performance target exists, so the yardstick is the CARE score.

**Result: both existing models score about 0.48 on Farms A+B, below the 0.5 a
random detector receives and well below the published autoencoder's 0.66.** They
are too quiet. Earlier "detected" counts used any 50-minute alert; the benchmark
requires sustained evidence, and these models rarely provide it.

## What the CARE score measures

Implemented in `src/hydropredict/analysis/care_score.py`, following the reference
`CAREScore` in [AEFDI/EnergyFaultDetector](https://github.com/AEFDI/EnergyFaultDetector)
and the paper [CARE to Compare (Gück et al., 2024)](https://arxiv.org/abs/2404.10320).

| Part | Question it answers | Computed on |
|---|---|---|
| Coverage | How much of the fault period is flagged, favouring precision (F0.5)? | Anomaly datasets, pointwise |
| Earliness | Are flags early in the fault period? Weight 1, then linearly to 0 at the end | Anomaly event windows |
| Reliability | Is each dataset correctly alarmed or quiet (event-level F0.5)? | All datasets |
| Accuracy | How rarely are normal periods flagged? Counts double | Normal datasets |

A dataset is alarmed when a criticality counter reaches 72: +1 per flagged
normal-status reading, -1 per unflagged one, never below 0. That is 12 hours of
net anomaly evidence. CARE = (coverage + earliness + reliability + 2 x accuracy) / 5,
except 0 if nothing is alarmed and mean accuracy if that is <= 0.5. This rule
is why a random detector scores 0.5.

Settings follow the reference code: status 0 is normal, earliness descends after
a quarter of the event, alarm at criticality >= 72. A paper-wording variant
(statuses 0 and 2 normal, descent after half, alarm above 72) changes no score by
more than 0.003. Farm A status is ignored, per the publisher's note. The scorer is
unit-tested against scikit-learn F-beta and a line-for-line copy of the reference
earliness arithmetic.

## Results (reference settings)

No model was refit and no threshold changed; saved per-reading scores were rescored
by `scripts/score_care_benchmark.py`. "Pointwise" uses each threshold exceedance,
the benchmark's intended input; "persistent" uses the original six-reading alerts.

| Model | Input | Farm A | Farm B | A+B | Faults alarmed | Normal datasets alarmed |
|---|---|---:|---:|---:|---:|---:|
| Isolation Forest (v1) | pointwise | 0.483 | 0.000 | 0.462 | 1/18 | 0/19 |
| Isolation Forest (v1) | persistent | 0.474 | 0.000 | 0.455 | 1/18 | 0/19 |
| Extra Trees residuals (v2) | pointwise | 0.486 | 0.477 | **0.484** | 2/18 | 1/19 |
| Extra Trees residuals (v2) | persistent | 0.481 | 0.474 | 0.479 | 2/18 | 1/19 |

Components for the best row (v2 pointwise, A+B): coverage 0.110, earliness 0.024,
reliability 0.333, accuracy 0.977. Full tables: `docs/results/care-score-v1/`.

The published reference points (paper Section 4.2.3) are 0.66 for the autoencoder
normal-behaviour model, 0.5 for random, below 0.5 for Isolation Forest and 0 for
all-normal or all-anomaly. They cover all 95 datasets including Farm C, so they are
indicative rather than a like-for-like comparison with A+B only.

## Why the scores are low

Thresholds were set at the 99th percentile of normal calibration scores to keep
false alarms rare. Accuracy is therefore excellent (0.98), but only about 3% of
readings inside fault periods are flagged, so criticality seldom reaches 72.

There is signal: anomaly datasets accumulate more criticality than normal ones.
For v2 the median maximum criticality is 16 on anomaly datasets versus 8 on normal
ones; three faults exceed 60. The evidence is too weak and too intermittent to
cross the benchmark's alarm level. This is a diagnosis, not a tuned result.

## Next step

Build a detector that flags degradation consistently rather than rarely: an
autoencoder normal-behaviour model as in the paper, smoothed residual scores and
a threshold chosen for the CARE score. Choose settings on Farms A and B only, then
score Farm C once as the untouched confirmation set.
