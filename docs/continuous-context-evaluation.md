# Continuous development context evaluation

14 October 2026. **Context reconstruction is implemented and verified, but it did not
resolve the temporal model's event-coverage or detection limitations. No replacement model
was promoted.** The original baseline and all earlier experiment outputs remain preserved.

## Fixed experiment

Use only the existing February-April 2019 validation interval. Merge the two source files
by exact observation time within that interval, keeping measurement values unchanged.
No missing intervals are filled, no timestamp jitter is rounded, and no fitting, calibration
or test boundary is crossed. Source membership and source row numbers are stored in a
separate provenance table; they never enter predictor features. Identical duplicate
observations would be collapsed with both origins recorded. Conflicting duplicates fail.

Load the previously frozen raw-q0.99 and temporal-q0.99 bundles. Do not refit, recalibrate,
change the coverage rule or inspect final test predictions. This isolates the effect of
context availability from changing the model. Models still use their original training-only
scalers and normal-reference thresholds. Prefix checks confirm earlier predictions do
not depend on later readings; the raw model's scores are exactly unchanged by added context.

Implementation source commit: `a154f9c`. Final outputs:
`models/continuous-context-2026-10-14-v2`. The earlier v1 run is preserved; v2 adds a residual
gap table and a provenance serialization typing correction without changing the experiment.
Files include `protocol.json`, `validation_timeline.csv`, `source_provenance.csv`,
`remaining_outages.csv`, raw and temporal score CSVs, `results.json` and checksums.

## Reconstruction result

The interval contains 6,137 publisher-selected normal-reference observations and 605
publisher-selected pre-fault observations: 6,742 distinct original timestamps. There are
no duplicates or conflicts, and zero synthetic rows. There remain 48 gaps over 450 seconds.
These are gaps in the available records; they do not by themselves prove actual plant
downtime or sensor failure. The source was already filtered by its publisher.

Temporal inputs require two hours after a reset. Joining the files restores 92 pre-fault
score rows, increasing available pre-fault scores from 400 to 492 out of 605. The temporal
model scores 5,814 of the 6,742 combined observations. Some file-split context was therefore
recoverable, but real gaps in available records still trigger resets and warm-up loss.

| Measurement | Raw frozen model | Temporal frozen model |
| --- | ---: | ---: |
| Pre-fault score rows before reconstruction | 605 | 400 |
| Pre-fault score rows after reconstruction | 605 | 492 |
| Originally eligible validation events | 6 | 6 |
| Warnings before originally eligible events | 0 | 1 |
| Events retaining adequate scored coverage | 6 | 2 |
| Detections among adequately scored events | 0 | 0 |

The temporal warning is still the earlier 2 February event. Its scored-window coverage
remains 73.6%, below the fixed 80% criterion. It must not be described as an adequately
monitored successful detection. The two events retaining adequate temporal score coverage
remain undetected. The original six-event denominator is retained separately so abstention
cannot silently improve the reported recall.

## Alert burden and interpretation

On all available selected normal-reference readings, the raw model alerts on 133 of 6,137
(2.17%); temporal alerts on 54 of 5,322 (1.01%). Those denominators differ, so these are
availability summaries rather than a matched superiority claim.

On the temporal normal-reference timestamps available both before and after reconstruction,
the same 54 of 5,321 readings are flagged in both runs. There is no measured alert-burden
improvement attributable to reconstructed context. The difference of one additional nominal
score row does not change the conclusion. As before, selected normal-reference alert fractions
are not independently established fleet false-alert rates.

## Engineering and research status

All 69 tests pass, along with Ruff lint, formatting and strict Mypy. Tests verify exact
timestamp reconstruction, source lineage, duplicate conflicts, interval exclusions, real
gap preservation and prefix independence. Source and partition artifacts remain immutable.

This completes the planned investigation of artificial context gaps. It does not establish
that the LSTM will succeed, nor justify lowering monitoring eligibility to count more faults.
The next model experiment needs a declared strategy for limited temporal context. A small
shorter-window comparison could test whether coverage can improve without excessive alerts;
alternatively implement the report's compact LSTM with explicit gap masks and sequence
eligibility. In either case, maintain a fixed development protocol and an honest account of
the small number of adequately monitored events. Avoid further tuning of the inspected test
period; a genuinely independent final evaluation still needs to be established.

## Reproduction

Use a new output directory and the existing frozen bundles:

```powershell
python scripts/evaluate_continuous_context.py --preparation-run data/processed/runs/2026-10-13-verified --comparison-run models/validation-comparison-2026-10-13-v2 --output models/continuous-context-new-run
```

The current runner imports shared validation reporting helpers from the comparison script,
so its environment also needs the optional `.[analysis]` dependencies. It scores only the
declared development interval. Source files may contain later records, but they are removed
before context construction or inference, and the test partition and test score files are
not used in this evaluation.
