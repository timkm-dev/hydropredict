# CARE autoencoder: results

1 November 2026; related issue #21. Method frozen in
[care-autoencoder-protocol.md](care-autoencoder-protocol.md); metric in
[care-score-results.md](care-score-results.md).

**Result: the autoencoder early warning detector scores 0.583 on all 95 CARE
datasets and 0.578 on held-out Farm C.** That beats a random detector (0.5), the
published Isolation Forest (below 0.5) and our earlier models (about 0.48), but
stays below the published autoencoder (0.66). The settings chosen on Farms A and
B transferred to an unseen farm with only a small loss (0.606 to 0.578).

## Results

| Farms | Role | Datasets | CARE | Faults alarmed | Normal datasets alarmed | Coverage | Earliness | Reliability | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | development | 22 | 0.620 | 5/12 | 0/10 | 0.247 | 0.104 | 0.781 | 0.984 |
| B | development | 15 | 0.580 | 3/6 | 0/9 | 0.090 | 0.026 | 0.833 | 0.975 |
| A+B | selection | 37 | 0.606 | 8/18 | 0/19 | 0.195 | 0.078 | 0.800 | 0.980 |
| **C** | **held-out confirmation** | **58** | **0.578** | **9/27** | **10/31** | 0.341 | 0.394 | 0.437 | 0.858 |
| All | benchmark comparison | 95 | 0.583 | 17/45 | 10/50 | 0.283 | 0.267 | 0.556 | 0.905 |

| Reference (paper Section 4.2.3, all 95 datasets) | CARE |
|---|---:|
| Published autoencoder normal-behaviour model | 0.66 |
| Random detector | 0.5 |
| Published Isolation Forest | below 0.5 |
| All-normal or all-anomaly | 0 |
| Our Isolation Forest / residual models (A+B only) | 0.46 / 0.48 |

## How it was done

One small autoencoder per dataset learns that turbine's normal 10-minute
behaviour from its own training year. Its reconstruction error is smoothed over
6 hours and flagged above the 97.5th percentile of held-back normal readings.
These two settings were chosen from a 24-setting grid on Farms A and B and
frozen in `docs/results/care-autoencoder-v1/selection.json` before Farm C was
scored once. Farm C was trained on Colab (Tesla T4, 12 minutes). A Farm A model
retrained there correlated 0.99999997 with the laptop version, and all 58 Farm C
source files matched the local copies byte for byte.

## What the results show

- **It generalises.** The held-out farm, with different turbines, about 220
  sensors instead of 42 to 56, and different fault types, scored close to the
  development farms, so the result does not rest mainly on tuning to A and B.
- **Farm C's weakness is false alarms.** All 10 alarmed normal datasets are in
  Farm C. Three of them were flagged for almost their whole prediction period
  (accuracy 0.01, 0.05 and 0.20), which suggests their operating behaviour
  shifted after the training year rather than brief noise. Normal-period accuracy
  falls from 0.98 on A+B to 0.86 on C.
- **More of each fault window is flagged on C.** Earliness rises to 0.39 on C
  from 0.08 on A+B and coverage to 0.34 from 0.19: flags fill more of the fault
  window, especially its early part. Part of this comes with the higher overall
  flag rate that also causes C's false alarms.
- **Gap to 0.66.** The published model used farm-specific network sizes and
  adaptive thresholds; ours is one fixed, simpler design across all farms.
  Candidates to close the gap are a threshold that adapts to regime changes and
  handling of long-running deviations. Any such change needs fresh confirmation
  data, because Farm C has now been used.

## Limits

Development evidence on a public benchmark, not field validation. Normal labels
and event windows come from the dataset publishers. Published reference numbers
come from the paper's implementation, which the authors state is not exactly
reproducible, so the comparison is indicative.
