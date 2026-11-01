# Sequence-pattern experiment results — issue #20

The strongest exploratory candidate is **random-convolution shape features with a single-reading alert**. It warned 10 of 11 eligible fault groups across the three chronological evaluation periods, including all three February–April groups. This is the most promising candidate from this experiment for a research proof of concept. It is not established as a reliable predictive model: its alert burden exceeds the previous 2% ceiling, warnings do not survive three-reading persistence, and the timing controls below show how much apparent success can arise from frequent alerts.

## Complete comparison

Cells show eligible fault groups warned / eligible groups; nominal point-alert fraction. Repeated timestamps in a six-hour fault group are not counted as separate independent successes.

| Model | Alert rule | November 2018 | December 2018 | February–April 2019 |
|---|---|---|---|---|
| dictionary-levels | single | 2/3; 1.04% | 1/5; 1.78% | 1/3; 2.35% |
| dictionary-levels | three | 0/3; 0.00% | 0/5; 0.04% | 0/3; 0.08% |
| dictionary-shape | single | 1/3; 1.80% | 4/5; 1.53% | 1/3; 1.80% |
| dictionary-shape | three | 0/3; 0.02% | 0/5; 0.00% | 0/3; 0.00% |
| random_convolution-levels | single | 1/3; 1.35% | 4/5; 2.71% | 1/3; 2.27% |
| random_convolution-levels | three | 0/3; 0.00% | 3/5; 0.42% | 0/3; 0.47% |
| random_convolution-shape | single | 2/3; 2.05% | 5/5; 2.55% | 3/3; 2.52% |
| random_convolution-shape | three | 0/3; 0.07% | 0/5; 0.13% | 0/3; 0.12% |

## Strongest candidate versus the saved LSTM

On **exactly the same 6,469 February–April endpoints**, random-convolution shape features warned **3/3** eligible groups versus **1/3** for the saved LSTM. Nominal alerts increased from **80/5,904 (1.36%)** to **149/5,904 (2.52%)**. Separate nominal alert episodes increased from **26 to 117** over 488.75 observed nominal hours: **1.28 to 5.75 per observed day**. These are observed hours, not full calendar days, and nominal labels are publisher labels, not verified operational false-alarm adjudications. The three eligible events are a subset of nine fault timestamps in that period; missing coverage excludes the others.

| Eligible fault timestamp (UTC as parsed) | First warning lead within six hours |
|---|---|
| 2019-02-02T00:07:00+00:00 | 44.2 min |
| 2019-02-03T23:57:53+00:00 | 280.1 min |
| 2019-04-02T09:06:14+00:00 | 75.4 min |

Three consecutive exceedances remove every eligible warning for this candidate in all three periods. The new signals are short. Counting a single alert in a six-hour window makes chance hits likely, so 3/3 must not be presented as 100% predictive accuracy.

## Chance-timing controls

After selecting the strongest candidate, 2,000 circular-shift controls per fold moved its alarms within each continuous observed segment. These preserve the local alarm count and most clustering, while changing timing relative to faults. Short segments and local alarm enrichment limit how much this control can move alarms. A second simple comparator assumes independent random alerts at the observed nominal point-alert rate.

| Period | Actual groups warned | Mean groups warned after segment shifts | Shifts equalling/exceeding actual | Expected groups warned by independent random alerts |
|---|---|---|---|---|
| november | 2/3 | 1.64 | 64.1% | 2.29 |
| december | 5/5 | 4.29 | 44.0% | 4.33 |
| primary | 3/3 | 2.09 | 30.2% | 2.51 |

These are descriptive post-selection checks, not confirmatory significance tests. They must be considered alongside the raw detection count. Eleven groups and repeatedly inspected development periods cannot establish generalization.

## What was actually run

Four fixed models × three chronological folds = **12 fits**, each evaluated with two predeclared alarm rules = **24 evaluations**. No hyperparameter or threshold search against evaluation labels was performed. The best candidate was selected retrospectively from these evaluations.

- Input: seven observed readings (about 30 minutes) from six channels; reset at gaps outside 150–450 seconds; no filling or future context.
- Representations: training-standardized levels, or per-window centered/scaled channel shapes. Shape standard deviation floor 0.05 in training-standardized units.
- Dictionary: at most 24 positive prototypes per training fault group, 2,048 sampled negative prototypes; log nearest-negative / nearest-positive squared-distance ratio. Negative examples use the six-hour future-fault label, including earlier portions of publisher pre-fault windows.
- Random convolution: 256 fixed random multichannel kernels with lengths/dilations (3,1), (3,2), (5,1), (7,1); maximum and proportion-positive outputs; standardized features and ridge classifier alpha 10. Training positives weighted equally across fault groups and collectively balanced with negatives. This is an independent ROCKET-inspired implementation, not MiniRocket or a reproduction of its published scores.
- Train/calibrate/evaluate: August–September / October / November; August–October / November / December; August–December / January / February–April. Existing six-hour boundary purges and crossing-group exclusions retained.
- Threshold: strict exceedance of each fold's calibration-normal 98th percentile, frozen before evaluation. Alert rules: one exceedance or three consecutive exceedances. Evaluation fault labels do not enter fitting, scaling or calibration.
- All periods were previously used for project development. May onward was not accessed. Earlier folds reuse data in later training, so the 10/11 total is a chronological backtest summary, not eleven independent trials of one frozen model.
- Input checksums verified; saved-model replay exact on the same batch; prefix predictions agree within 1e-6. The initial incomplete attempt stopped on an overly strict cross-batch floating-point check (maximum difference about 5.3e-7); the complete run uses exact same-batch replay plus a separate numerical-tolerance prefix check. No model settings changed after that attempt.

## Decision and artifacts

Retain this shape-based convolution model as the **leading experimental candidate**, alongside the LSTM on the proof-of-concept evidence track. Do not replace production alerts or claim the earlier acceptance criteria are met. If a quieter candidate is required, the shape dictionary stayed below 2% nominal readings in all three periods but warned only 1/3, 4/5 and 1/3 groups respectively.

Runner: `scripts/experiment_sequence_patterns.py`. Re-run requires a fresh output directory (the script refuses to overwrite the completed run). Local models, scores, calibration scores and hashes: `models/sequence-patterns-v1-complete/`. Aggregate results and timing controls: `docs/results/sequence-patterns-v1/`. Restricted measurements and per-reading scores remain in ignored directories. No deployment, GitHub issue mutation, commit or push was performed.

## Verification

The complete Python suite passed: 128 tests. New checks cover batch-prefix independence for both model families and representations, exact serialized replay, and timing-control boundary cases.
