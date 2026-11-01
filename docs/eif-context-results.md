# EIF and operating-context results

24 October 2026, follow-up to #20 and the
[author-source review](shp-source-code-review.md).
**Neither forest nor the conditional-vibration candidate passed the predeclared
improvement criteria. Retain the existing baseline; no deployment changes.**

Nine fits were completed: ordinary IF, genuine fully extended IF, and the
conditional-vibration model, each on the primary development fold and two earlier
chronological folds. Settings and fallback were frozen in
[the protocol](eif-context-protocol.md) before fitting. No parameters or thresholds
were revised after observing results.

## Primary development result: February-April

| Candidate | Eligible native fault groups warned | Shared-endpoint groups warned | Native selected-normal alerts | Native nominal episodes |
| --- | ---: | ---: | ---: | ---: |
| Ordinary IF, 500 trees / 2,048 samples | 0/4 | 0/3 | 91/6,137 (1.48%) | 12 |
| Fully extended IF, 500 / 2,048 | 0/4 | 0/3 | 83/6,137 (1.35%) | 32 |
| Conditional vibration residuals | 0/4 | 0/3 | 544/6,137 (8.86%) | 36 |

The frozen LSTM warns before 1/3 eligible groups on the shared endpoints. Native
raw-channel coverage admits six fault timestamps in four overlapping-window
groups; LSTM/shared coverage admits three timestamps in three groups. Counts
are not six independent failures. The extra raw coverage did not yield extra
warnings. All nine development fault timestamps remain in each event report.

The primary alarm requires three consecutive threshold exceedances at valid
150-450-second intervals. Fractions above are fractions of selected-normal
readings marked by that alarm, not independently verified false-alarm probabilities.
The threshold was fitted only in January at the 98th percentile of selected-normal
point scores. Subsequent fractions can differ because of persistence and time shift.

## Earlier stability folds

| Candidate | November warned groups | November nominal alerts | December warned groups | December nominal alerts |
| --- | ---: | ---: | ---: | ---: |
| Ordinary IF | 0/3 | 1.14% | 0/5 | 0.11% |
| Fully extended IF | 1/3 | 0.15% | 0/5 | 0.15% |
| Conditional vibration | 0/3 | 0.31% | 0/5 | 0.42% |

The EIF November warning covers the group containing three November 2 timestamps.
The first eligible event in that group has about 219 minutes of lead time; the
other two timestamps are later in the same group. Count this as **one warned
group**, not three independent successes. It is evidence of a useful local signal,
but not stable transfer across the evaluated periods.

## What was learned

- The paper-inspired change is substantive: the new EIF uses six-dimensional
  Gaussian hyperplane splits, not sklearn's axis-aligned IF. Its November result
  differs from ordinary IF, but its improvement is not consistent across time.
- Isolated point exceedances occur before some faults. In the primary period,
  EIF has a point exceedance before one eligible group; the conditional model
  has one before a different group. Neither survives the frozen three-reading
  persistence rule. Full point-only diagnostics are retained, without changing
  the winning criterion after seeing them.
- Fewer nominal alert readings do not necessarily mean fewer interruptions:
  primary EIF has fewer flagged readings than ordinary IF but more nominal
  episodes (32 versus 12).
- Conditioning vibration on lubrication inflow and apparent power, using this
  fixed regression model and maximum standardized residual, did not transfer well.
  Its primary-period alert burden also exceeded budget. This rejects this
  configuration, not every possible physical/conditional model.

The sensible next modeling experiment returns to the previously researched
weak-label precursor dictionary and supervised sequence representation. It should
test whether short patterns recur before different events, rather than reinterpret
isolated exceedances as a completed predictive solution. All inspected periods
remain exploratory development data; independent confirmation is still separate.

## Implementation and reproducibility

`scripts/eif_reference.py` independently implements the hyperplane algorithm in
[Hariri et al.](https://arxiv.org/abs/1811.02141) and the SHP paper's tree/path
description. Training-only global standardization is shared with ordinary IF.
Extension is full (all six dimensions); depth is 11 for sample size 2,048.
The leaf correction is zero for size <=1, one for size 2, and the usual
log/Euler-constant approximation otherwise. Gaussian normals and bounding-box
intercepts are sampled with seed 42. This is a transparent reference adaptation,
not a claim to reproduce the missing historical isotree call exactly.

The residual model fits four fixed gradient-boosting regressors for V1-V4 using
only V5/V6. Absolute residuals are divided by fitting-normal vibration SDs; the
maximum is the score. Saved residual contributions show which channel contributed.
Models train on publisher-normal history, never on future evaluation measurements.

No new dependency installation or environment downgrade was needed. No May-onward
measurements were used. Source-package and frozen-baseline hashes were verified.
The full Python suite has **122 passing tests**, including analytical oblique-tree
paths, full-dimensional split checks, deterministic fitting, outlier scores,
duplicate/empty-branch handling, causal prefixes and existing split/persistence
regressions. Changed Python files pass Ruff checks. Models and scores were also
replayed exactly from saved artifacts in a separate interpreter.

```powershell
.venv/Scripts/python.exe -m scripts.compare_eif_context --package artifacts/supervised-precursor-v1-verified --diagnostics models/lstm-diagnostics-2026-10-15-v1 --output models/eif-context-new
```

Canonical model/score files: `models/eif-context-v1/` (ignored).
Aggregate results, decision, frozen settings, input/code identities and replay
evidence: `docs/results/eif-context-v1/`. A fresh output path is required; the
runner refuses to overwrite previous evidence.

For loading these trusted local joblib files outside the runner, import
`Candidate` from `scripts.compare_eif_context` into the main script first: the
runner was invoked with `python -m`, so its wrapper is serialized as
`__main__.Candidate`. `ExtendedForest` has its normal importable module path.
The independent-process check explicitly verifies this loading procedure.

Public GitHub API reads now work: #19, #20 and #21 were freshly read at task start.
Their bodies still predate the completed local studies. No authenticated write
was attempted after the previously declined credential-access approval; no issue
update, closure, push or publication is claimed. The results above are local.
