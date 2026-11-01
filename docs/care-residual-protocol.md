# CARE condition-aware experiment v2

29 October 2026. Frozen before model fitting; related issue #21. A/B are already
development-exposed. This is one fixed comparison, not independent confirmation.
Farm C remains performance-uninspected. No automatic dataset/model promotion.

Hypothesis: component temperature departures from expected temperature under
current and recent load are more useful than unconditional sensor outliers.

- Evaluate the same 37 files separately. Never pool history across files.
- Context A: ambient sensor_0_avg, wind_speed_3_avg, power_30_avg, sensor_52_avg
  (rotor rpm). Context B: sensor_8_avg (outside temperature), wind_speed_61_avg,
  power_62_avg, sensor_25_avg (rotor speed). Use current values and trailing six-
  reading means of those four context channels. Windows require exact 10-minute
  cadence and complete context, reset on gaps, and never look forward.
- Outputs: all average channels whose source dictionary description contains
  'temperature', except the context ambient channel. Exclude fitting-constant
  targets. Fit one multioutput ExtraTreesRegressor per file: 100 trees,
  max_depth=12, min_samples_leaf=20, max_features=1.0, random_state=42, n_jobs=2.
  Use at most 12,000 evenly spaced fitting examples. No target temperatures in
  model inputs; no lagged targets that can track away degradation.
- Training chronology: first 80% of supplied training rows fits regression;
  next 10% estimates each residual's median and scale (1.4826*MAD, floor 1
  source temperature unit); last 10% calibrates the 99th percentile of the
  maximum absolute standardized residual across targets. Status 0/2 used only
  for training selection. Require 1,000 fitting and 250 rows in each calibration
  segment. Retain only complete selected input/target rows; no interpolation.
- Use six consecutive above-threshold valid prediction readings, resetting on
  gaps/invalid rows. Context may use prior training history; alert persistence
  starts fresh at prediction. Preserve v1 episode/detection denominators and
  no-alarm comparator. Invalid prediction rows are reported and not alerted.
- Same gate per farm: >=50% anomaly detection at least 24h before annotated end,
  <=1 normal episode/30 observed days, <=1% normal rows alerted, >=80% valid
  prediction rows in every file. End-relative lead is not verified failure lead.
- Save all results, models and hashes, including failures. Do not alter settings
  after outcomes. Compare the complete configurations rather than claiming a
  controlled ablation of the regression alone. Further development on A/B cannot
  be described as independent evidence. No additional model search in this run.
