# CARE deeper research: component trajectories and grouped supervision

29 October 2026; issue #21. User authorized renewed investigation after two failed
screens. This is development research, not independent confirmation. A/B only;
reserve C performance. Do not reinterpret an annotation endpoint as exact failure.

Hypothesis: a maximum across unrelated channels confuses normal shifts with fault
patterns. Signed component residuals and their sustained trajectory, with examples
of labelled degradation, may distinguish them. Compare one fixed supervised model
to v1/v2. No post-result search in this run.

Representation: reuse each file's frozen v2 normal-history regression, bias and
scale, verifying source/model hashes. Compute signed standardized temperature
residuals for prediction history. Preserve each target separately. Include current
residuals, exact-cadence trailing 6- and 36-reading means, and current minus
36-reading mean. Require complete features; reset on gaps/missing rows. No future
context. Start trajectory history at prediction; record warm-up exclusions. No
status/IDs/time/labels or ambient inputs in classifier features. Farm-specific
feature schema, not pooling different sensor names across farms.

Labels: normal event files give negative examples across their prediction rows.
Anomaly positives only inside annotation interval and >=24h before its end.
Other anomaly-file rows are excluded from classifier fitting (uncertain controls).
At most 500 evenly spaced valid rows per file; each class receives half the total
weight and each file within a class equal weight. Avoid long-file domination.

Classifier: HistGradientBoostingClassifier, max_iter=100, learning_rate=.05,
max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=10, early_stopping=False,
random_state=42. Two CPU threads. It predicts annotated early degradation, not
an exact failure time or calibrated probability of physical failure.

Outer evaluation: leave one whole turbine out within each farm. Every file of the
outer turbine is excluded from classifier training AND threshold selection.
Its earlier normal-history adaptation remains permitted (v2); this is adapted
cross-turbine label transfer, not zero-shot or globally chronological deployment.
Anonymized dates prevent assuming between-file calendar ordering.

Threshold calibration: on the remaining turbines, use inner leave-one-turbine-out
models to produce scores for normal control files. For the outer final model,
select the lowest of [.5,.6,.7,.8,.9,.95,.975,.99,.995,1] yielding <=1 episode/30
observed normal days and <=1% alerted normal rows across inner control scores.
The threshold 1 is a no-alarm fallback. Only normal controls select the threshold;
inner anomaly detection is not an optimization target. Refit classifier on all
outer training turbines. This is empirical nested calibration, not a distribution-
free guarantee under turbine shift. Reuse inner fits for identical excluded pairs.

Alerts: six consecutive valid exceedances, reset at gaps. Same v1/v2 evaluation
gate and denominators: >=50% events detected >=24h before annotated end, <=1 normal
episode/30 observed days, <=1% normal rows alerted, >=80% valid rows per file.
Report all 37 files, turbine-level results, threshold degeneracy, warm-up coverage,
and pre-existing alerts. Keep failures; never drop hard controls. Include within-
event vs normal score diagnostics for v1/v2 as descriptive, not tuned estimates.

Uncertainty: report a fixed-seed 2,000-replicate turbine-cluster bootstrap of event
detection proportion and normal episode rate, excluding undefined resamples from
the relevant statistic and reporting their count. These descriptive intervals
condition on fitted cross-validation predictions; they do not capture model-
selection/adaptation uncertainty or create independent confirmation.

Stop after this configuration. Any improvement must be called development evidence.
Even a passing gate still requires independent confirmation and endpoint clarity.
