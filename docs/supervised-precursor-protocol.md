# Fixed supervised precursor experiment (#19, #20)

Declared before the first run on 24 October 2026. This is a development experiment,
not an independent test or a production fault predictor.

Use the checksum-verified exact union of publisher normal and prefault readings.
Fit on August–December 2018, calibrate on January 2019, and evaluate on
February–April 2019. Reset history at each split; remove the final six hours of
each split so labels never depend on events outside it. Faults with overlapping
six-hour precursor windows form one group. If a group crosses a split boundary,
exclude its entire precursor span from both splits. Retain all events in coverage
reports, including excluded and unobservable ones. Never interpolate readings.

The target is a recorded fault strictly after the reading and at most six hours
ahead. Zero means no recorded fault in that horizon, not confirmed healthy.
Publisher membership is provenance only and is never a predictor.

Use the existing causal 30-minute feature configuration: six current readings,
per-channel 15/30-minute means and standard deviations, and six rates of change
(36 features). Require a full 30-minute history after a gap; reset for sampling
intervals outside 150–450 seconds. Fit one HistGradientBoostingClassifier with
learning_rate=0.05, max_iter=100, max_leaf_nodes=15, max_depth=4,
min_samples_leaf=20, l2_regularization=1, class_weight=balanced,
early_stopping=False, random_state=42. No scaling is required. No LSTM score
is a feature. No hyperparameter search or outcome-driven revision is permitted.

Set the threshold to the January selected-nominal score's 98th percentile using
the higher order statistic. Alert only for scores strictly above it at three
consecutive available endpoints, with 150–450 second intervals. Freeze threshold
and persistence before evaluating February–April. Selected nominal means
publisher-normal membership and target zero; this is not verified health.

Report native and matched-endpoint LSTM/classifier results. Preserve native
alert decisions when matching endpoints. An event is eligible with at least 80%
of its 72 five-minute slots scored and a last reading within ten minutes before
the event. Report every event, overlapping groups, first-warning lead times,
selected-nominal point alerts, and alert episodes per observed day. A candidate
is promising only if it adds eligible matched groups without losing a group
warned by the baseline and stays at or below 2% selected-nominal point alerts
on its native development endpoints. Otherwise retain the LSTM baseline.

Record source, protocol, code, input, model and output hashes plus dependency
versions. Replay serialized-model predictions exactly. Raw records, fitted
models and portable data packages stay ignored; commit aggregate evidence only.
Later (May onward) measurements must not enter the package. Issue #21 owns
independent confirmation, including the fact that the old holdout was inspected.
