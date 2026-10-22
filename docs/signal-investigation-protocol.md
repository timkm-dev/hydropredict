# Missed fault and operating regime investigation protocol

Declared 21 October 2026 before running this experiment suite. This is further
development on an already inspected February–April 2019 interval, not independent
confirmation. Preserve the original Isolation Forest and LSTM bundles and results.

## Question and hypotheses

Do missed faults have unusual multichannel relationships, rare operating states,
or temporal changes that global tree partitions and averaged reconstruction errors
miss? First describe channel ranges, dependence and regime occupancy in the normal
fitting/calibration blocks and before every development fault. Physical channel
meanings and maintenance context remain unknown; do not invent equipment diagnoses.

Run exactly four additional normal-reference-only candidates:

1. **PCA residual**: standardise six raw channels using fitting statistics, fit a
   full PCA, and sum squared projections on the last three components divided by
   their fitted eigenvalues (floored at 1e-8). Tests unusual weak relationships.
2. **Local level novelty**: mean Euclidean distance to the 20 nearest fitting
   observations in standardised six-channel space. Tests rare local states.
3. **Regime covariance**: eight KMeans clusters, seed 42, ten initialisations;
   fit LedoitWolf covariance within each fitting cluster. Score by the minimum
   squared Mahalanobis distance over the eight clusters. Tests deviation from all
   known operating regimes, rather than one global distribution.
4. **Local temporal novelty**: the same 20-neighbour score on the existing 36
   causal raw/mean/std/derivative features with 30-minute context. Fit its own
   standardiser using eligible fitting rows only. Tests unusual local dynamics.

Algorithms use scikit-learn [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[nearest neighbours](https://scikit-learn.org/stable/modules/neighbors.html), and
[LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).
These are candidate mechanisms, not claims that the dataset meets Gaussian assumptions.

## Frozen data and decision rules

- Verify the saved preparation manifest, reconstructed development timeline,
  diagnostic score files, and input-package checksums before use. Do not score
  the May-onward final period. Reading source files to verify hashes or filter
  the declared interval does not permit using later rows as predictors.
- Keep the original 30,807-reference chronological split: first 80% fit,
  last 20% calibrate. No fault labels filter either normal-reference subset.
- Calibrate every new threshold at the separate normal calibration 99th
  percentile. Use strict exceedance. No threshold grid, fault-specific channel
  weighting, additional model families, or model changes after seeing results.
- No interpolation, synthetic data, time-to-fault, source membership, timestamps,
  or segment identifiers as predictors. Reset temporal context at existing gap
  boundaries; preserve warm-up and unavailable-observation accounting.
- Compare candidates and stored baselines on the same timestamps available to
  every model. Report native availability separately. Retain all nine candidate
  faults, the original eligible denominator, and scored coverage eligibility.
- Retain the existing six-hour warning interval, 80% five-minute slot coverage,
  and latest reading within ten minutes. Overlapping fault windows can share a
  warning; report first/last warnings and number of positive readings per event.
- A promising candidate must detect at least two adequately scored matched
  events (one more than the stored best baselines), with at most 2% point alerts
  on matched publisher-selected nominal readings. This budget is not a proven
  operational false-alarm rate. Compare alert episodes as well as point counts.
- Inspect normal-reference alert burden by calendar month and initial context
  proximity of qualifying warnings. Do not silently remove startup warnings or
  change eligibility to obtain a better result.
- Verify saved-model score round trips and causal prefix invariance. Record
  all results, scores, fitted bundles, input hashes, runtime, and environment.

## Stopping rule and delivery

Stop after these four candidates and the declared robustness diagnostics. Do not
keep searching until a favourable result appears. If none passes, deliver that
negative result with concrete evidence about the missed faults. If one passes,
call it a development candidate, not a replacement or validated predictor. Leave
the dashboard's deployed LSTM unchanged. Any future final evaluation needs a
declared independent protocol; the old test has already been inspected previously.
