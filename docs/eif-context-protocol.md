# EIF and operating-context comparison (#20 follow-up)

Declared before fitting on 24 October 2026. No model promotion is automatic.
Use the verified #19 package and its existing purges; refilter on each fold.
Fit only publisher-normal rows with no upcoming recorded fault within six hours.

Compare sklearn ordinary IF and an independent fully extended EIF reference:
500 trees, 2,048 samples without replacement, seed 42, depth ceil(log2(sample)).
Both use six raw channels standardized using fitting-normal means/SD only.
EIF uses Gaussian six-dimensional split normals and intercepts sampled uniformly
inside each node's bounding box, with standard leaf path-length adjustment.
This is an adaptation of the published hyperplane algorithm, not an exact
reproduction of the missing SHP isotree call. No dependency changes are needed.

The primary fold fits August-December, calibrates January, evaluates February-April.
The two stability folds fit Aug-Sep/calibrate Oct/evaluate Nov and
fit Aug-Oct/calibrate Nov/evaluate Dec. All are previously inspected development
periods. Run both models on all three folds regardless of outcomes.

Threshold is the higher-order 98th percentile of calibration selected-normal
scores; score strictly above threshold is a point exceedance. The primary alarm
requires three consecutive exceedances with intervals 150-450 seconds. Save both
point and persistent decisions, but do not select whichever gives better results.
Fit/scaler/calibration never sees evaluation measurements or outcomes.

Compare the primary fold with the frozen LSTM on shared endpoints, preserving
its original decisions; also report native coverage and all events. Eligible
events need 80% of 72 five-minute slots and a last score within ten minutes.
Events with overlapping six-hour precursor windows form groups. A primary
improvement must add matched eligible warned groups without losing any LSTM
warned group, at <=2% native selected-normal point-alert fraction after persistence.
A stability pass additionally needs at least one eligible group warned and <=2%
alerts in each earlier fold. All model/fold results remain visible.

If neither forest meets both gates, run exactly one predeclared conditional
vibration model on the same folds. Fit four independent HistGradientBoostingRegressor
models (learning_rate=.05, max_iter=100, max_leaf_nodes=15, max_depth=4,
min_samples_leaf=30, l2_regularization=1, early_stopping=False, random_state=42)
to predict V1-V4 from V5 and V6. Use absolute residuals divided by each vibration
channel's fitting-normal SD, and take the maximum across channels as the score.
Use the same calibration threshold and persistence; no residual feature searches.
Save predictions and residual contributions. No claim that these are causal faults.

Verify prefix/batch invariance, oblique splits and analytical tree traversal;
verify serialized-model score replay and input checksums. Save source/config/
artifact identities, per-fault coverage/leads, nominal episodes and observed
exposure. If all models fail, retain the existing baseline and document the failure.
Do not alter the six-hour target or consume May-onward measurements.
