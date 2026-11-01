# Earlier-history transfer experiment

Declared on 24 October 2026 after the fixed #20 comparison failed and the owner
authorized a broader search for a solution. This is a bounded development study.
Its issue creation is pending GitHub access approval. Do not use February-April
readings or outcomes for model/threshold selection in this experiment.

Use the verified #19 package, restricted to its fitting and January observations.
The package's existing December 31 six-hour purge remains; no observations are
restored or interpolated. Purge six hours before each new partition boundary,
exclude crossing precursor groups, and reset feature history at every boundary.
Keep the six-hour label and three-consecutive-alert rule unchanged.

Two chronological selection folds:

| Fold | Fit | Calibrate | Validate |
| --- | --- | --- | --- |
| November | August-September | October | November |
| December | August-October | November | December |

Compare exactly four configurations: the original fixed gradient-boosted tree
and a balanced logistic regression (C=0.1, L2 regularization, max_iter=2000), each
with absolute-level or relative-change features. Logistic regression uses a
training-only StandardScaler; the tree has the #20 configuration. Absolute
features are the existing 36. Relative features are six deviations from the
last 30-minute mean, six 30-minute standard deviations and six elapsed-time
least-squares slopes. They exclude absolute levels. Use the same scored
endpoints for all configurations, with 30-minute history and 150-450-second
cadence. No other features, horizons, hyperparameters or persistence settings
will be searched in this study.

Each fold independently fits on its past, calibrates a strict 98th-percentile
selected-nominal threshold in the calibration month, and then freezes it.
Selected nominal, coverage, group membership, episodes and lead times use the
#20 definitions. Report every fold and configuration, including failures.

An admissible candidate must warn at least one eligible group in **each**
validation fold, and have at most 2% selected-nominal point alerts in **each**
fold. Among admissible candidates choose the most total warned groups, then
the lowest pooled selected-nominal point-alert fraction, then model name.
Fit at most that one candidate on August-November, calibrate on December and
evaluate January once. January is additional development confirmation, not an
untouched independent test. Accept it as worth further investigation only if
it warns at least one eligible January group and stays within 2%; otherwise
reject it. Never use January to select a runner-up or revise settings.

If no candidate meets the selection rule, stop without a January run. Preserve
the monitoring baseline and document that these four compact approaches did
not establish stable six-hour precursors across earlier periods. Do not conclude
that all prediction is impossible. New fault-mode/operating context or additional
independent history would then be a more defensible next step than repeated
tuning on the same few development faults.
