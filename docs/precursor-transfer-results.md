# Earlier chronological backtests: decision and way forward

24 October 2026. **None of the four fixed candidates passed. No candidate was
selected and no additional January confirmation was run.** The study followed
[the declared protocol](earlier-backtest-protocol.md); February-April measurement
files were never loaded by its runner.

This checks whether the failure of the first corrected supervised model was
specific to its model class or dependence on absolute sensor levels. Logistic
regression is simpler and regularized; relative features describe change rather
than absolute level. Neither alteration established stable six-hour warning.

| Candidate | November warned groups | November nominal point alerts | December warned groups | December nominal point alerts |
| --- | ---: | ---: | ---: | ---: |
| Absolute features, boosted trees | 0/3 | 0.00% | 0/5 | 0.00% |
| Absolute features, logistic regression | 1/3 | 4.40% | 0/5 | 1.98% |
| Relative features, boosted trees | 0/3 | 0.02% | 0/5 | 0.13% |
| Relative features, logistic regression | 0/3 | 0.02% | 0/5 | 0.60% |

Each model was trained, calibrated and evaluated chronologically, independently
in each fold. Thresholds used only the preceding calibration month. Admissibility
required a warning for at least one eligible group and at most 2% selected-nominal
point alerts in **each** validation month. All candidates failed. The one
November detection occurred with excessive nominal alerts and did not transfer
to December. These percentages concern publisher-selected nominal observations,
not independently adjudicated false alarms.

All four configurations shared endpoints. Full event reports retain ineligible
events, warning lead times, coverage and episodes, and counts retain split and
warm-up exclusions. Every saved model reproduced its validation probabilities
exactly after serialization. The result is exploratory development evidence,
not a claim that prediction is mathematically impossible or that eight fault
groups represent the entire fault population.

## What this changes

The earlier-package omission is fixed. More earlier positive examples alone did
not solve transfer. The tested regularized and relative-change alternatives also
failed the fixed warning rule. Repeatedly tuning against the same February-April
faults is therefore not a credible next step, and no model is promoted.

Keep the LSTM as an anomaly-monitoring baseline with the existing explanatory
proof of concept. Its 1/3 development warning result still does not establish
reliable prediction. Preserve explicit distinctions between anomalies, observed
warnings, missed faults and unavailable coverage.

The next modeling work should be conditional on new information:

1. Obtain fault modes, confirmed failure/trip times, repairs and operating-state
   or load information for timestamped events. The current timestamp-only log
   cannot distinguish failure mechanisms or verify that a recurring sensor
   change should precede a given event.
2. If such context exists, define a physically justified target for a specific
   failure mode and operating condition, then fit and validate that target on
   separate chronological event groups. Do not quietly change the six-hour
   horizon to improve the existing score.
3. If it cannot be obtained, retain an honest anomaly-monitoring project scope,
   or obtain an additional dataset with documented failure modes and precursor
   coverage for a separate predictive experiment. A dataset replacement is not
   required for the monitoring proof of concept.
4. Under issue #21, agree independent confirmation before claiming prediction.
   The previously inspected May-onward history is not an untouched final test.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe -m scripts.backtest_precursor_transfer --package artifacts/supervised-precursor-v1-verified --output models/precursor-transfer-new
```

Canonical model/score outputs: `models/precursor-transfer-v1/` (ignored).
Reviewable aggregates: `docs/results/precursor-transfer-v1/`, including all eight
fold reports, declared input/code identities and a machine-readable rejection
decision. The first fixed comparison is in
[supervised-precursor-results.md](supervised-precursor-results.md).

Six focused tests cover causal features, timestamp targets, purges, duplicates,
persistence, relative-level invariance and candidate selection. The full Python
suite has 119 passing tests. Changed Python files pass Ruff checks.

GitHub issue creation/update was attempted but the credential-access approval
was declined. These results are local; no issue was updated or closed for these
experiments, and no publication or deployment is claimed.
