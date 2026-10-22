# Isolation Forest baseline results

13 October 2026. This is an exploratory numerical baseline, not a validated operational
warning model. Read `shp-readiness-audit.md` for the source-selection and coverage caveats.

## Fixed experiment

Source commit: `c1724c6`. Bundle: `models/isolation-forest-2026-10-13-v1` (ignored by Git).
The bundle includes estimator, training scaler, schema, threshold, parameters, provenance,
checksums, per-observation scores and a per-event evaluation report. Loading the saved
bundle reproduced predictions. Only trusted local joblib artifacts should be loaded.

Inputs are V1..V6 at their original timestamps. No interpolation, physical ratios, rolling
features, segment identifiers, fault labels or outcome countdowns enter the estimator.
This avoids unresolved channel semantics and offline gap handling. The scaler is fitted
on the estimator training subset only. Pointwise scoring is backward independent: changing
a future reading cannot change an earlier prediction.

From the 30,807 chronological training-reference observations, the first 24,645 fit the
model and the final 6,162 calibrate its threshold. Calibration ends before validation starts.
These are the source publisher's retrospectively selected normal-reference measurements,
not independently verified healthy records. No fault timestamps are used to filter them.

Parameters: 200 trees, 256 samples per tree, random seed 42 and one CPU worker. The positive
anomaly score is negative scikit-learn `score_samples`. The threshold is the 99th percentile
of scores on the disjoint normal-reference calibration subset. A strict exceedance creates
a point alert, with no persistence or cooldown. This predeclared initial operating point
is not a measured 1% operational false-alarm guarantee. It was not optimised against faults.

Scikit-learn 1.9.1 and Python 3.14.4 were used. Model fit plus threshold calibration took
approximately 0.36 seconds in this local execution, excluding imports and file operations.
This is one measured run, not a hardware-independent timing guarantee.

## Held out results

Each event needs 80% of six hours of five-minute slots and an observation within ten minutes
before the fault. Warnings must occur strictly before the fault within that six-hour window.
The first qualifying warning sets lead time. The saved report includes every candidate
event, coverage and exclusions. Overlapping event windows can share warnings; recorded
events are not assumed to be independent failure mechanisms.

| Result | Validation | Final test |
| --- | ---: | ---: |
| Candidate faults | 9 | 11 |
| Eligible under coverage rule | 6 | 5 |
| Detected eligible faults | 0 | 1 |
| Eligible event recall | 0% | 20% |
| Point alerts | 6 / 605 observations | 7 / 733 observations |
| Median lead time among detected eligible events | Not defined | 384.817 seconds |

The single eligible test detection was about 6.4 minutes before its fault. Six test faults
had some qualifying warning if the coverage rule is ignored; that is a sensitivity count,
not permission to claim 6/11 adequately monitored recall. Eligibility was fixed before
scoring. Zero validation detections is a real result and is retained.

Among 10,913 later publisher-selected normal-reference observations, 148 exceeded threshold
(1.36%). This is a biased nominal-reference point exceedance rate, not false-alert episodes
per day. A general false-alert frequency is not estimable from pre-fault excerpts, so that
metric is intentionally null. Calendar gaps are not counted as observed healthy exposure.

## Reproduction and next work

```powershell
python scripts/train_isolation_forest.py --preparation-run data/processed/runs/2026-10-13-verified --output models/isolation-forest-my-new-run
```

Use a new output directory; old bundles are preserved. The standardisation and threshold
are contained in the bundle and must not be refitted during inference. Training uses local
CPU and needs no Colab connection. Colab remains planned for the LSTM.

Model API:

```python
from hydropredict.models.isolation_forest import SHPIsolationForest
from hydropredict.data.preprocessing import load_telemetry_dataframe

model = SHPIsolationForest.load("models/isolation-forest-2026-10-13-v1")
scores = model.score(load_telemetry_dataframe("path/to/validated.csv"))
```

The poor recall means this baseline is not ready for operational alerts. Next experiments
should address the data dictionary and a justified calibration/validation protocol, then
compare temporal models or features without changing this recorded result. The current
test set has been inspected: future tuning against these findings would make it development
data. Reserve a fresh final period or disclose repeated test use in the eventual report.

API reference: [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
