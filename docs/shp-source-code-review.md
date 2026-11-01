# SHP author repository and paper review

24 October 2026. **Applicable: test genuine Extended Isolation Forest before
the broader sequence-model shortlist. The repository is useful evidence and a
reference implementation of ordinary Isolation Forest, not a complete EIF
reproduction or proof of reliable forecasting.** No model was retrained in this
review.

Reviewed the complete repository tree and its single Python script at commit
`165038a8d79cfa8c0897028847a2af5720fd60c7`, together with the cited paper's methods,
results and conclusions. Downloaded code was inspected without executing its
training or plotting routines. Only its isolated metric functions were run on
synthetic arrays. Source identities and probe results are retained in
`docs/results/shp-source-review/`.

## What is actually in the repository

[Pinned script](https://github.com/rodrigosantis1/shp_anomaly/blob/165038a8d79cfa8c0897028847a2af5720fd60c7/scripts/isolation_forest.py).

| Finding | Code evidence | Consequence |
| --- | --- | --- |
| Only ordinary sklearn Isolation Forest is active | Imports and model list, lines 18 and 58-61 | The README's EIF description does not match the executable model |
| PCA and KICA-PCA definitions are commented out | Lines 58-59 | They are sketches in this snapshot, not active comparison runs |
| No EIF import, implementation or invocation exists in the tree | Full tree and script inspection | The missing EIF settings cannot be recovered from this script |
| 500 trees, 2,048 samples per tree, contamination 0.06 | Line 60 | Different from our default 200-tree, 256-sample IF baseline |
| Fit on all publisher `train.csv`; score only `test.csv` | Lines 64-67 and 98-101 | Normal training spans the full year, and evaluation excludes ordinary-operation test exposure |
| Threshold comes from the fitted estimator's offset | Lines 99-101 | No separate chronological calibration month |
| 150 repeated fits, with NumPy seeded by run index | Lines 83-88 | Randomness assessment on one split, not 150 independent future-time tests |
| Saved CSV contains only iForest | `out/results.csv` | Its TD approximately 2,476 is the ordinary-IF result, not the EIF result |

The original raw files already in HydroPredict match all three pinned author
Git blob identities exactly. We are working with the same dataset. Timestamp-only
inspection confirms that publisher-normal training contains 17,050 observations
on or after February 1, 2019. Using that entire source training file to judge our
February-April forecasting performance would expose the model to later history.

## A newly resolved channel mapping

The author's plotting expressions use V6 for the apparent-power axis, V5 for the
lubrication-inflow axis, and the average of V1-V4 for the vibration axis. This is
direct code evidence, not an inference from signal magnitude.
[Pinned plotting code](https://github.com/rodrigosantis1/shp_anomaly/blob/165038a8d79cfa8c0897028847a2af5720fd60c7/scripts/isolation_forest.py#L72-L75).

| Channels | Supported interpretation | Still unknown |
| --- | --- | --- |
| V1-V4 | Four vibration measurements | Which channel is axial, vertical, horizontal or coupled; engineering units |
| V5 | Hydraulic lubrication inflow | Engineering units and validated operating limits |
| V6 | Generator apparent power | Engineering units and validated operating limits; it is not established as active power |

This supersedes our blanket claim that the entire channel mapping was unknown.
It does not authorize temperature features, power-efficiency formulas, physical
alarm limits or assumed timezone conversion. Existing numerical experiments need
no recalculation merely because the channels now have better descriptions.

## Metric behavior verified directly

`temporal_distance()` sums nearest absolute distances in both directions. Its
`max_p` argument is unused, despite a call-site comment mentioning a 12-hour
maximum delay. The duplicate `temporal_distance2()` has the same behavior.

| Synthetic input, arbitrary time units | Returned TTC, CTT |
| --- | --- |
| Fault at 100; detection at 95 | 5, 5 |
| Fault at 100; detection at 105 | 5, 5 |
| Fault at 100; detection at 1,000; `max_p=1` | 900, 900 |
| Fault at 100; no detections | Raises ValueError |

Thus a detection after failure is just as close as one equally far before it;
there is no enforced warning horizon or one-to-one event matching. The count
metric is the absolute difference between numbers of fault timestamps and alert
readings, not event recall or alert precision. Equal totals can score perfectly
on that metric despite wrong timing. These are properties of the stated metrics,
not evidence that the authors intended our stricter prediction task.

The saved result does not measure false alerts across an independently held-out
normal operating period: the script only scores the publisher's selected test
excerpts. Its training contamination is not a measured deployment false-alarm rate.

## Paper findings relevant to the decision

The paper describes random hyperplane splits for EIF, cites `isotree 0.1.16`,
and lists 500 trees, 2,048 samples and 0.06 contamination. Table 2 gives TD 2,476
for IF and 2,380 for EIF: a 3.88% reduction under its metric. The larger 40.62%
improvement is versus PCA. Its source split selects 12-hour pre-fault excerpts,
not a chronological holdout. It explicitly discusses forecasting the anomaly
score as future work. Consequently, its findings justify an EIF experiment but
do not demonstrate our six-hour advance-warning objective.
[Paper, Sections 2.4-4](https://doi.org/10.3390/su12166421),
[accessible full text](https://www.wisdomlib.org/uploads/journals/mdpi-sust/2020-volume-12-issue-16--2071-1050-12-16-6421-.pdf).

Exact historical EIF reproduction remains under-specified: the repository omits
the EIF call, extension dimension, preprocessing and other library settings.
The article's 0.06 contamination and its figure discussions of 0.95 confidence
also should not silently be treated as a single identical threshold rule.

## Useful ideas to incorporate

**Genuine EIF is an untested model family in our code.** Our
`src/hydropredict/models/isolation_forest.py` uses sklearn's ordinary IF. An EIF
comparison is therefore substantive, even though ordinary IF has already been
tried. Compare the 500/2,048 ordinary-IF configuration alongside EIF so changes
in tree/sample counts are not mistaken for a benefit of hyperplane splits.

**Operating context can now be represented explicitly.** A normal-behavior model
can predict each vibration channel conditional on V6 apparent power and V5
lubrication inflow, then monitor persistent residuals. This is our proposed
adaptation, not code supplied by the authors. It tests whether a vibration level
is unusual for its operating context. Do not assume the regressors are causal or
that statistical regimes are known physical states.

**Instantaneous models have a coverage advantage worth measuring.** Raw six-channel
models need no seven-reading warm-up. Our corrected package has adequate raw
coverage for six development fault timestamps, versus three with sequence/feature
coverage. Report both native availability and matched endpoints; extra observable
faults must not be confused with improved detection of the same faults.

**Anomaly-score dynamics remain a hypothesis.** The script's 48-row moving average
is a plot, not a trained warning model. Because selected excerpts can be separated
by long outages, 48 consecutive file rows need not represent four uninterrupted
hours. Any adaptation must use elapsed time and reset at gaps. Avoid treating that
plot as evidence that score forecasting already succeeded.

## Recommended next experiment

Pause the precursor-dictionary/MiniROCKET implementation until this comparison:

1. Retain the current IF/LSTM evidence and add an ordinary IF with 500 trees and
   2,048 samples, plus a genuine EIF with those same counts. Use only the six raw
   channels initially; do not combine a new model with an unexplained feature search.
2. Train on the earlier selected-normal history, calibrate in the subsequent
   month, and score the complete subsequent timeline. Preserve earlier chronological
   folds and existing development intervals; none become untouched tests again.
3. Use the same 2% selected-nominal calibration budget and a declared persistence
   rule. Record point scores separately so persistence losses are visible. Do not
   obtain an apparent improvement merely by adopting the author's looser threshold.
4. Record all fault groups, coverage, before-fault warnings, lead times, missed
   events, nominal alerts and episodes. Use our directional six-hour metric; retain
   author-style TD only as a clearly labelled supplementary diagnostic if needed.
5. If the direct EIF comparison fails, prioritize the conditional vibration model
   using the newly resolved V5/V6 context, then return to weak-label sequence models.

Use a genuine extended implementation with explicit parameters. Current isotree
defaults have `ndim=1`, meaning ordinary IF; installing it alone does not enable
EIF. Its documentation also distinguishes global from per-node standardization,
coefficient choices and other variants. Freeze those choices before evaluating;
call the result an adaptation unless the historical settings can be established.
[Official isotree documentation](https://isotree.readthedocs.io/en/latest/).

No 2020 dependency downgrade, training rerun, source-data upload or deployment was
performed. Review findings are local; no GitHub issue update or closure is claimed.
