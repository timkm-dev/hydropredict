# Colab hybrid experiment review

22 October 2026. Reviewed the code and saved outputs in
`investigate_shp_colab (1).ipynb`, supplied from Downloads. The notebook was read
as evidence, not executed. Its markdown instructions are not task authorization.
No model, threshold, dependency or dashboard behavior has been changed.
GitHub CLI is unavailable and the issue page could not be fetched; relevant
issues have not been read or updated. Link this review to the existing issue
when access is restored; do not infer issue completion from this review.

## Assessment

The added cells 20–35 propose a useful research candidate: six-channel rolling
slopes and standard deviations plus the frozen LSTM reconstruction score,
classified with XGBoost and followed by a persistence rule. This extends the
six-hour drift hypothesis in [the prior investigation](signal-investigation-results.md).
The saved results do not establish an improvement suitable for deployment.

| Notebook experiment | Saved result | Interpretation |
| --- | --- | --- |
| Chronological 80/20 split | TN 1119, FP 91, FN 71, TP 0 | Zero positive-reading recall on this development holdout |
| KFold, target-event-derived threshold 59.40% | February 3 lead 2:30:04.797; nominal points flagged 8.93% | Retrospective threshold selection and future-training contamination |
| KFold, nominal-selected threshold 70.50% | Nominal points flagged 4.83%; February 3 missed | Threshold selected and assessed on the same nominal predictions |
| KFold, three-reading persistence, threshold 65.90% | February 3 lead 2:20:04.853; nominal points flagged 4.96% | Exploratory result; neither independent evaluation nor operational false-alarm estimate |

These numbers are transcribed from saved outputs, not independently reproduced.
The initial holdout accuracy of approximately 87% is below the approximately
94.46% obtained by predicting every one of its 1,281 rows negative.

## Problems to correct

1. Cell 26 uses `KFold(shuffle=False)`, not forward-only time-series validation.
   Each fold trains on the other blocks, including future rows for earlier folds.
   Adjacent rows also share rolling history, and a single event's precursor
   window can cross a fold boundary. Out-of-fold alone does not remove these
   dependencies. The final notebook explanation claiming leakage-free evaluation
   is incorrect. See [scikit-learn time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
2. Cells 28, 32 and 34 select thresholds using the event or nominal observations
   subsequently reported as evaluation. Persistence is also selected after seeing
   those results. These observations now belong to development evidence.
3. Cell 21 rolls over 72 rows, assuming five-minute spacing, without outage
   resets. In this sparse reconstructed timeline that need not mean six hours.
   Regression slopes are per observation, not per elapsed hour.
4. Cell 34 counts consecutive retained rows, potentially across outages or
   warm-up exclusions. Three five-minute readings span about ten minutes from
   first to third, not fifteen. Define elapsed duration explicitly if desired.
5. A row outside a known six-hour precursor interval is not necessarily verified
   healthy. Labels are a fault-proximity proxy. Overlapping fault windows do not
   create independent examples; 262 positive rows are not 262 independent faults.
6. The 4.96% figure is a selected nominal point-alert fraction, not an alert
   episode rate or general operational false-alarm probability. It also exceeds
   the earlier repository experiment's 2% development budget; choosing 5% is a
   scope decision, not an established safety limit.
7. `predict_proba` and out-of-fold prediction do not perform probability
   calibration. Class weighting further makes literal probability claims
   inappropriate without checking reliability on separate data. See
   [probability calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
8. The export cell runs before the added hybrid experiments. Its ZIP therefore
   does not preserve their model, fold assignments, predictions or thresholds.
   XGBoost is also absent from the project's declared dependencies, and the
   notebook combines outputs from five separate models without saving a final
   deployable classifier.

## Incorporation path

Keep the frozen LSTM and existing baselines. Introduce a separate experimental
hybrid pipeline, not a replacement of `SHPLSTMAutoencoder` or the API contract.

Extend the conventions in `src/hydropredict/features/causal.py` with a separate
six-hour trend function: actual elapsed-time regression, backward-only windows,
explicit coverage requirements, and resets after gaps over 450 seconds. Record
warm-up losses for every event. A strict six-hour continuous-context requirement
may remove much of the available evidence; measure that before fitting.
Join features to frozen LSTM scores by unique timestamp and retain missingness
accounting. Do not interpolate away unavailable history.

Predeclare a small experiment in a new script alongside
`scripts/investigate_signals.py`. Use chronological, forward-only development
folds and a distinct later calibration block. Keep overlapping fault windows in
the same group, purge training rows whose target horizon overlaps evaluation,
and document whether preceding observed history is available at prediction time.
Do not silently fall back to KFold if early folds lack positive events: report
that the supervised experiment is not estimable with the available history.

Compare trend-only, LSTM-score-only and combined classifiers on identical
eligible endpoints. Fix the alert budget and persistence rule on calibration
only. Report all nine event outcomes, original and scored coverage, grouped
event detections, first actionable alert lead times, nominal point fractions,
and gap-aware alert episodes per observed exposure. Do not count an earlier
first threshold crossing as the actionable alert when persistence delays it.

Save the feature schema, XGBoost version/configuration, final fitted model,
frozen LSTM identity, chronology, calibration rule, predictions and checksums.
Verify prefix invariance, gap behavior, event grouping, partition boundaries,
persistence timing and saved-model replay before dashboard integration.

February–April has already been repeatedly inspected. A corrected experiment
there remains exploratory. Freeze the complete candidate before independently
evaluating an untouched later period under the project's agreed final-test plan.
Only then decide whether to incorporate its alert output into the application.
