# Validation model comparison

13 October 2026. **Decision: retain the original baseline and do not promote a replacement.**
Temporal features show a limited signal, but coverage loss prevents a reliable claim of
improved fault detection. Lower thresholds trade additional detections for much higher
normal-reference alert burden. This study uses validation only; the previously inspected
test scores and original model bundle were preserved and were not used for selection.

## Experiment and saved evidence

Source commit `3a65cc5`; output `models/validation-comparison-2026-10-13-v2` (ignored by Git).
The initial v1 run is preserved. The v2 reporting run uses the identical models and threshold
grid, with explicit post-warm-up coverage counts and a comparison chart. No new candidates
were added after observing v1 results.

Files include `protocol.json`, `comparison.json`, `missed_fault_diagnostics.json`, nine
`validation_event_*.png` figures, `validation_tradeoff.png`, score CSVs, six saved model
bundles and output checksums. Reloaded predictions matched; the reproduced raw q=0.99
scores also matched the original baseline. All 65 tests, lint, formatting and strict types
passed. Tests cover future independence, outage resets, real elapsed-time derivatives,
label exclusion and artifact round trips.

The fixed development grid has two inputs (raw and temporal) and three normal-reference
calibration quantiles (0.99, 0.975, 0.95). Trees=200, max_samples=256, seed=42, one worker.
Fit and calibration reuse the same chronological source subsets as the original experiment.
All thresholds derive from disjoint normal calibration scores; validation faults are used
only for development evaluation. The source data remains publisher-selected reference data.

Temporal inputs comprise the six raw channels, rolling means and standard deviations at
30 and 120 minutes, and first derivatives: 36 numerical inputs. Windows use actual time,
never interpolate, and reset when consecutive readings are more than 450 seconds apart.
Two hours of context are required after each reset. Each scoring call starts fresh; a future
streaming application must pass preceding history, not score individual readings without it.
This warm-up is an explicit experiment choice, not a report requirement or optimal setting.

## What the missed validation faults show

Six original events meet the predeclared six-hour coverage policy. Their maximum original
scores range from approximately 0.530 to 0.591, below the frozen threshold of 0.6195.
The scores therefore did not cross the threshold, but they were not all close to crossing.
Late-window median scores rise for some events and fall for others; there is no uniform
increase that simply justifies lowering the threshold for every event.

Three candidate events fail the original monitoring policy. One has no pre-fault observations.
One of the poorly covered events had a warning, which is not counted as a successful
adequately monitored detection. Several April events have overlapping windows and share
the same high-scoring readings; they are not independent evidence of distinct failure modes.
Plots show readings as points so unobserved gaps are not connected by misleading lines.

## Comparable validation tradeoffs

The paired comparison restricts both models to the temporal model's 400 available
validation timestamps and 5,321 selected normal-reference timestamps. Original event
eligibility remains fixed at six so abstaining during warm-up cannot improve the denominator.
Normal-reference measurements come only from February-April 2019, not the later test period.

| Input | Calibration quantile | Warnings before original eligible events on matched timestamps | Normal-reference point alerts | Nominal alert episodes |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.99 | 0 / 6 | 1.50% | 64 |
| Raw | 0.975 | 0 / 6 | 3.82% | 105 |
| Raw | 0.95 | 3 / 6 | 11.63% | 209 |
| Temporal | 0.99 | 1 / 6 | 1.01% | 25 |
| Temporal | 0.975 | 1 / 6 | 2.48% | 48 |
| Temporal | 0.95 | 1 / 6 | 5.32% | 100 |

Using all 605 raw validation readings, raw q=0.95 warns before 4/6 events rather than the
paired 3/6. This difference is precisely why timestamp-matched and native-availability
results are both saved. Do not compare raw 4/6 with temporal 1/6 without noting availability.

The temporal q=0.99 warning occurs about 19.24 minutes before the 2 February fault.
However, only two events retain adequate model-score coverage after warm-up. The detected
event has only 73.6% score-slot coverage, below the 80% requirement. **Temporal detection
among events with adequate scored coverage is therefore 0/2.** This is not evidence of
robust improvement, and reporting only 1/6 would hide an important qualification.

The 2% nominal point-alert budget was an explicit development comparison rule, not a
measured operational guarantee. Temporal q=0.99 and raw q=0.99 meet it. The higher raw
detection count at q=0.95 exceeds that budget substantially.

Nominal episodes start at an alert following a non-alert or a gap over 450 seconds. Matched
normal-reference exposure sums only consecutive timestamp intervals up to 450 seconds,
about 441 hours; outages do not count as healthy exposure. These selected reference counts
are descriptive alert burden, not independently validated fleet false-alert frequency.

## Next justified work

Do not change the default baseline or tune on final test outcomes. First establish whether
continuous historical context can be reconstructed within each development interval from
the normal and fault-adjacent source observations. If possible, join those measurements
chronologically without giving provenance or outcome labels to the model. This may reduce
artificial resets caused by the source split, but must preserve gaps, duplicates, temporal
boundaries and traceability. Keep event eligibility and model availability explicit.

Only then predeclare another small experiment, such as a shorter warm-up or an LSTM
comparison. Do not shrink the coverage criterion simply to count the current detection.
Threshold selection must weigh detection, lead time and sampled normal alert burden.
Any chosen configuration still needs an uninspected final evaluation period or a candid
statement that no independent final test remains.

## Reproduction

Install the optional plotting dependencies once with `python -m pip install -e ".[analysis]"`.
Run from a clean checkout and use a new output directory:

```powershell
python scripts/compare_validation_models.py --preparation-run data/processed/runs/2026-10-13-verified --baseline models/isolation-forest-2026-10-13-v1 --output models/validation-comparison-new-run
```

The command does not read the held-out test partition or its score files. It verifies
training and validation partition checksums, writes the fixed protocol, produces validation
diagnostics and saves every candidate. Source timestamps retain the existing common UTC
computational convention; the physical plant timezone and channel dictionary remain open.
