# CARE autoencoder normal-behaviour model: protocol

1 November 2026; related issue #21. Written before training. Goal: an early
fault warning detector judged by the benchmark CARE score
([care-score-results.md](care-score-results.md)), not by the project's earlier gates.

## Model (one per dataset, as in the benchmark paper)

- Inputs: `_avg` channels whose feature description marks neither angle nor
  counter. Drop channels constant or entirely missing in the fitting rows.
- Training rows (`train_test == "train"`) in time order: first 70% fitting,
  next 10% early stopping, last 20% threshold calibration. Each uses only
  status 0/2 rows with at most 20% missing selected channels.
- Standardise with fitting-row mean and standard deviation (floor 1e-6); missing
  values become 0 (the fitting mean) after scaling.
- Autoencoder: d -> max(8, d/2) -> max(4, d/8) -> max(8, d/2) -> d, ELU,
  Adam lr 1e-3, batch 256, at most 40 epochs, stop after 5 epochs without
  early-stopping loss improvement and keep the best weights. Seed 42.
- Raw score: root mean squared standardised reconstruction error per reading.
  Readings with more than 20% missing channels get no score and are never flagged.

## Decision rule and selection

- Smoothed score: causal time-based rolling mean over the last w readings' span,
  computed across the full series so prediction starts with available history.
- Threshold: quantile q of smoothed calibration scores. Flag readings above it.
- Grid: w in {1, 6, 36, 144} readings (10 minutes, 1, 6, 24 hours) and
  q in {0.9, 0.95, 0.975, 0.99, 0.995, 0.999}: 24 configurations.
- Select the configuration with the highest pooled Farms A+B CARE score
  (reference settings). Report every configuration and each farm separately.

## Confirmation

- Farm C performance is not inspected until the A+B selection is recorded in
  `docs/results/care-autoencoder-v1/selection.json`. Then train Farm C with the
  same code and score it once with the selected configuration. No changes after.
- Report Farm C, and all 95 datasets, beside the published 0.66 autoencoder
  reference. Selection on A+B is development tuning; only Farm C is held out.
