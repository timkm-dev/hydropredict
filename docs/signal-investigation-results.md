# Missed fault investigation results

21 October 2026. Four additional normal-only models failed to improve detection
over the saved 30-minute Isolation Forest and LSTM. None met the declared success
criterion; none replaces a baseline or the dashboard's loaded model. A separate
descriptive analysis identified a gradual six-hour multichannel rise before the
February 3 fault. This is a hypothesis for future work, not a validated predictor.

## Experiment integrity

The [protocol](signal-investigation-protocol.md) was committed as `f8dcca7`
before candidate scoring. Implementation: `fb594be`. The run used the original
checksum-verified Colab input ZIP and stored comparator scores; no May-onward
final-test data was evaluated. Every new scaler and estimator used 24,645 earlier
normal-reference observations, followed by 6,162 distinct calibration observations
(temporal warm-up reduces these to 23,901 and 6,025). Every threshold remains the
normal calibration 99th percentile. No fault-specific channel weighting or
threshold search was performed. Saved-model round trips and prefix invariance
were verified. These source-selected normal references are not independently
verified healthy exposure.

This February–April development interval has already been inspected in previous
experiments. Results are exploratory and subject to selection bias. They are not
independent confirmation, an overall accuracy measurement, or fleet reliability.

## Same-timestamp comparison

All seven models share 6,452 development endpoints, including 5,887 nominal-reference
endpoints. Three of nine recorded faults meet scored coverage on this common
timeline. The original pre-fault coverage rule admits six faults; that denominator
is also retained in the saved evidence. Overlapping April 2 windows can share
warnings and must not be treated as independent failures.

| Model | Covered faults detected | Nominal point alerts | Nominal fraction | Nominal episodes |
| --- | ---: | ---: | ---: | ---: |
| Original raw Isolation Forest | 0 / 3 | 97 | 1.65% | 70 |
| Saved 30-minute Isolation Forest | 1 / 3 | 60 | 1.02% | 32 |
| Saved LSTM | 1 / 3 | 79 | 1.34% | 25 |
| PCA weak-component residual | 0 / 3 | 112 | 1.90% | 66 |
| Local level novelty, 20 neighbours | 0 / 3 | 267 | 4.54% | 132 |
| Eight-regime covariance distance | 0 / 3 | 381 | 6.47% | 113 |
| Local 30-minute temporal novelty | 0 / 3 | 98 | 1.66% | 38 |

The success criterion was at least two covered detections within a 2% matched
nominal point-alert budget. All four candidates fail. The point-alert fractions
and episode counts refer to selected normal-reference readings, not operational
false alarms. On native availability, all three new raw-channel methods detect
0 / 6 adequately observed faults; the new temporal method detects 0 / 3 and
excludes 290 readings for warm-up. Missing context does not explain the entire
negative result.

The [comparison CSV](results/signal-investigation-2026-10-21/comparison.csv),
[complete event table](results/signal-investigation-2026-10-21/events.csv), and
[monthly breakdown](results/signal-investigation-2026-10-21/monthly-nominal.csv)
retain all outcomes. The [score-ratio figure](results/signal-investigation-2026-10-21/event-score-comparison.png)
shows peak pre-fault scores relative to each model's own threshold; a ratio above
one indicates a crossing, not proof of a causal precursor.

## What the missed faults reveal

### Gradual February 3 rise

The [raw signal figure](results/signal-investigation-2026-10-21/february-3-signals.png)
shows V1–V4 rising gradually over six hours. None of the 71 preceding readings in
any channel falls outside that channel's normal calibration 0.1–99.9 percentile
band. A value can therefore be locally changing without being globally extreme.
Fitting-reference V1 and V2 correlation is 0.986; correlated rises are not four
independent warning signals.

After the fixed four-candidate suite, a descriptive follow-up measured six-hour
linear changes in fitting-standard-deviation units. It compared the event with
429 normal-reference control windows, at most one endpoint per UTC hour. Windows
intersecting labelled six-hour pre-fault intervals were excluded from controls.
No new detector was fitted and no threshold was changed.

| Channel | Fitted six-hour change | Controls with equally or more extreme absolute slope |
| --- | ---: | ---: |
| V1 | +1.66 fitting SD | 3.73% |
| V2 | +1.89 fitting SD | 2.10% |
| V3 | +1.83 fitting SD | 2.10% |
| V4 | +2.39 fitting SD | 3.26% |
| V5 | −0.07 fitting SD | 89.28% |
| V6 | +0.55 fitting SD | 17.25% |

Controls overlap and are publisher selected. These tail fractions are descriptive,
not p-values, accuracy, false-alarm probabilities, or independent trials. Selecting
a trend/channel after seeing this event creates further development bias. The
result motivates a longer-horizon change investigation, but does not prove that
the rise is a fault precursor rather than an ordinary operating transition.

The [trend table](results/signal-investigation-2026-10-21/six-hour-trends.csv)
retains every fault, including unavailable windows. In contrast, the first April 2
fault has ordinary six-hour trends in this analysis: between 28% and 92% of the
control slopes are at least as extreme, depending on channel. A single trend
mechanism is therefore not an established explanation for all missed faults.

### Score distributions overlap ordinary operation

At the February 3 pre-fault maximum, 5.42% of matched nominal LSTM scores and
23.00% of matched nominal 30-minute Isolation Forest scores are at least as large.
At the first April 2 fault these fractions are 4.04% and 2.89%. These are ranking
diagnostics, not proposed new thresholds. They explain why simply making the
models more sensitive risks flagging more ordinary measurements.
The [score-rank table](results/signal-investigation-2026-10-21/event-score-ranks.csv)
preserves every model and event.

### Operating regimes and calendar variation

The fixed numerical clustering describes different mixtures across periods. One
cluster accounts for 22.95% of fitting readings, 3.44% of calibration, and 10.53%
of development. Another accounts for 17.29%, 46.49%, and 41.52%. These clusters
are numerical groups, not verified physical operating states.

The regime-covariance candidate flags 14.92% of matched nominal March readings,
versus 1.52% in February and none in April. Local level novelty flags 9.80% in
March. A January-calibrated 1% reference threshold therefore does not guarantee
stable later alert burden. These changes may reflect actual operation, source
selection, or both; this dataset does not establish their cause.

### Existing detections are isolated

The LSTM's February 2 warning is one reading, at its first eligible sequence
endpoint 30 minutes after context begins, about 354 minutes before the fault.
The 30-minute Isolation Forest also produces only one qualifying reading, about
19 minutes before that fault. Neither is evidence of a persistent precursor.
No startup warnings were silently removed or promoted into stronger claims.

## Decision

Retain all negative experiments and keep the deployed LSTM unchanged. The current
best stored development comparators remain the 30-minute Isolation Forest and
LSTM; neither is ready for operational warning use. More epochs, a lower threshold,
or a new architecture is not justified merely by these results.

The most specific new hypothesis is slow coordinated drift that 30-minute context
does not fully describe. A future experiment must predeclare its longer-horizon
features and normal-only calibration, verify enough prior history for each event,
and report any coverage loss against the unchanged denominators. It also needs
verified channel semantics and maintenance/operating context to distinguish a
routine operating transition from degradation. New independent evaluation data
would be required before claiming useful general fault prediction.

## Reproduction and archived evidence

```powershell
.venv/Scripts/python.exe scripts/investigate_signals.py --output models/signal-investigation-new-run
.venv/Scripts/python.exe scripts/diagnose_signal_overlap.py --study models/signal-investigation-new-run --inputs artifacts/colab/hydropredict-lstm-input-2026-10-14.zip --output models/signal-overlap-new-run
```

Use new output names; existing runs are never overwritten. Local ignored folders
`models/signal-investigation-2026-10-21-v1` and
`models/signal-overlap-2026-10-21-v1` retain all scores, fitted candidates, model
parameters, plots, input hashes, run environment and output checksums. Joblib
bundles are local research artifacts and must only be loaded from trusted sources.
The timestamp parser fix is recorded separately as `896e547`.
