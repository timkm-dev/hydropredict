# Model-family comparison: score-scale sensitivity check

29 October 2026. Specified while the v4 family run is executing, before inspecting
its detection/alarm summaries. This addresses a methodological limitation of the
inherited fixed score grid, not a change selected from outer test outcomes.

Different uncalibrated model scores do not share a meaningful numeric scale.
Naive Bayes can saturate probability outputs and the SVM sigmoid is arbitrary.
Keep the original v4 run intact, and conduct an explicitly separate, fixed
normal-quantile calibration sensitivity check using the SAME four classifiers.
This is not four additional model families or independent confirmation.

Reuse each saved outer classifier without refitting. Refit the same 46 excluded-
pair inner classifiers per family deterministically to produce native normal
scores: logistic decision margin, GaussianNB log posterior odds, random-forest
class-1 probability, approximate RBF SVM decision margin. Never use outer scores
or anomaly labels for threshold selection.

For each outer turbine, consider thresholds at pooled finite inner-normal score
quantiles [.90,.95,.975,.99,.995,.999,.9995,1.0]. Select the first satisfying <=1
normal episode/30 observation-days and <=1% normal rows alerted under the same
six-reading persistence. The maximum normal calibration score gives an empirical
zero-alarm endpoint; it does not guarantee zero alarms on a new turbine.

All labels, features, file weights, samples, folds, coverage gates and >=24h
end-relative detection definitions remain unchanged. Save quantiles/thresholds,
inner warnings and native scores. Present both policies; do not silently replace
an unfavorable result or call selection among these exposed results confirmation.
This is a calibration sensitivity check, not a hyperparameter search over models.
