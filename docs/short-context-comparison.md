# Short temporal context comparison

14 October 2026. **Shorter context improves coverage modestly, but this remains development
evidence from a very small sample. The 30-minute model is a useful next comparison candidate;
no default model or operational alert policy has been replaced.** This concludes the declared
Isolation Forest context experiment. Do not expand the search based on these outcomes.

## Fixed protocol

Run artifacts: `models/short-context-2026-10-14-v1`; experiment implementation commit
`cf26dff`. Existing baseline and earlier experiment directories are preserved. Training
and calibration use exactly the same normal-reference subsets and temporal boundaries as
before: 24,645 observations for fitting and 6,162 for calibration before feature warm-up.
Source and prepared-data hashes were checked before the experiment.

Only the 15-minute and 30-minute models are newly fitted. The existing two-hour q=0.99
model is loaded unchanged. Each model has 36 numerical inputs: six raw channels, mean and
standard deviation at two time windows, and six first derivatives. The window pairs are
10/15 minutes, 15/30 minutes and 30/120 minutes, respectively. Consequently this compares
three context designs, not just three warm-up settings applied to identical features.

Warm-up equals the longest window. All designs use actual timestamps, reset after gaps
over 450 seconds and never interpolate. The 99th-percentile normal-calibration rule,
200 trees, 256 samples per tree, random seed 42 and one CPU worker remain fixed. The numeric
thresholds differ because each model has a different score distribution; no threshold was
tuned to fault timestamps. The six-hour event eligibility rule, 80% slot coverage and
ten-minute latest-observation requirement are unchanged.

Evaluation uses only reconstructed February-April validation context. The held-out test
period is not scored or used for selection. Prefix prediction tests and saved-model reload
checks pass. Fitting plus calibration took about 1.51 seconds for the 15-minute model and
1.35 seconds for 30 minutes in this run, excluding setup, validation scoring and file output.

## Measured results

| Context | Scored validation rows out of 6,742 | Events with adequate scored coverage | Detected adequately covered events | Selected nominal point alerts |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | 6,574 | 3 | 1 / 3 | 86 / 5,995 = 1.43% |
| 30 minutes | 6,452 | 3 | 1 / 3 | 60 / 5,887 = 1.02% |
| Two hours | 5,814 | 2 | 0 / 2 | 54 / 5,322 = 1.01% |

The new adequately covered detection is the 2 February fault, warned about 19.24 minutes
before its timestamp. Its score-slot coverage is now 100% for the 15-minute model and
98.6% for 30 minutes, compared with 73.6% previously. The monitoring criterion itself did
not change. This is a real availability improvement for that recorded event, not evidence
of general 33% recall across the plant.

On the fixed original six-event denominator, warnings occur before 2/6 events for the
15-minute model, and 1/6 for both 30 minutes and two hours. The additional 15-minute warning
precedes the 15 March event, but that event has only 75% scored coverage. It is excluded
from the adequately monitored detection count. All candidate events and exclusions are
retained in `results.json`, rather than dropping missed or insufficiently covered faults.

Native-availability nominal percentages have different denominators. On the common 5,322
normal-reference timestamps available to all three designs, the comparison is:

| Context | Point alerts | Point-alert fraction | Alert episodes |
| --- | ---: | ---: | ---: |
| 15 minutes | 61 | 1.15% | 44 |
| 30 minutes | 45 | 0.85% | 26 |
| Two hours | 54 | 1.01% | 25 |

The 30-minute design flags fewer matched nominal points than the others, but has one more
episode than the two-hour design. All are below the predeclared 2% nominal point-alert
development budget. These are publisher-selected normal-reference measurements, not
independent operational false-alarm labels, and neither percentages nor episode counts
establish a general fleet false-alert rate. More coverage necessarily exposes more readings
and can change episode counts.

## Decision and next milestone

Retain the original baseline as the historical reference. Keep the 30-minute temporal bundle
as a development comparator because it recovers the same adequately monitored detection
as 15 minutes with lower nominal alert burden. Do not advertise it as a production model:
only three validation faults have adequate score coverage, two remain missed, source
channel semantics and timezone are still unresolved, and no new independent final test
has been established. Multiple development experiments on the same validation interval
also increase selection bias.

The declared next milestone is the report's compact LSTM Autoencoder and thin Colab
notebook, rather than further Isolation Forest threshold searches. Carry forward explicit
sequence-length and warm-up accounting, gap resets, training-only transformations and
normal-reference calibration. Compare against these stored models on development data,
with an explicit final evaluation plan before reporting general performance. Shorter
context helping this baseline does not guarantee that an LSTM will perform better.

## Reproduction

Use the installed analysis extra and a new output directory:

```powershell
python scripts/compare_short_context.py --preparation-run data/processed/runs/2026-10-13-verified --context-run models/continuous-context-2026-10-14-v2 --comparison-run models/validation-comparison-2026-10-13-v2 --output models/short-context-new-run
```

Artifacts include protocol, complete event results, score CSVs, checksums, and separate
15-minute and 30-minute bundles with their saved context configuration. Old bundles missing
that configuration load with the original 120-minute default, preserving compatibility.
All 73 tests, lint, formatting and strict type checks pass.
