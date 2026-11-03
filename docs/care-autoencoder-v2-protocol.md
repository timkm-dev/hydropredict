# CARE autoencoder v2 (cleaned inputs): protocol

3 November 2026; related issue #21. Written before any v2 model was trained on
Farm C or scored by leave-one-farm-out. Builds on the frozen v1 method in
[care-autoencoder-protocol.md](care-autoencoder-protocol.md) and its
[results](care-autoencoder-results.md).

## Why a v2

A diagnosis of v1's 10 Farm C false alarms and a sample of its missed faults found:

- **Exploding inputs.** Channels that are almost constant while the model is fitted
  (a 24 V supply at 26.0 V with standard deviation 0.00013, on/off filter flags) get
  standardised values up to 190,000 once they move. One such channel wrecks the
  reconstruction of every other channel, inflates calibration scores and raises the
  threshold. v1 only removed channels that were exactly constant.
- **One sensor dominates** 8 of the 10 false alarms, typically slow-state sensors
  (water conductivity, fluid levels, pump currents) rather than component health.
- **Most missed faults show little or no signal** in the model's inputs.
- **Under-training.** 29 of the 37 Farm A/B v1 models stopped at the 40-epoch cap.
- Downtime leaking into the 6-hour smoothing was tested and caused none of the
  false alarms, so it is not changed.

Retraining Farms A and B locally with the cleaning below changed their scores
within noise, as expected (A and B have almost no exploding inputs). On one Farm C
dataset (C30) it lowered the median score from 27.7 to 0.79. These are development
observations; Farm C's CARE score under v2 is unknown when this is written.

## Unchanged from v1

Dataset files, `_avg` non-angle non-counter channels, row eligibility (status 0/2,
at most 20% missing), the chronological 70/10/20 fit / early-stopping / calibration
split, network shape, optimiser, learning rate, batch size, seed 42, the causal
smoothed score, the calibration-quantile threshold, the 24-setting window/quantile
grid and the benchmark CARE score.

## Cleaning shared by every v2 candidate

1. Drop channels with fewer than 10 distinct values in the fit rows.
2. Standardise with fit-row mean and standard deviation, then clip to +/-10.
   Missing values become 0 (the fit mean) before clipping, as in v1.

## Candidates (trained in one run, `scripts/train_care_autoencoder_v2.py`)

| Candidate | Training | Inputs |
|---|---|---|
| `clean` | at most 40 epochs, patience 5 (v1) | cleaned |
| `clean_long` | at most 200 epochs, patience 10 | cleaned |
| `clean_long_delta` | at most 200 epochs, patience 10 | cleaned; every channel described as a temperature, other than the farm's ambient sensor, is replaced by itself minus ambient temperature (A `sensor_0`, B `sensor_8`, C `sensor_7`) |

Each candidate yields two per-reading scores:

- `scores`: root mean squared standardised error (as v1);
- `scores_per_sensor`: each channel's squared error divided by its mean squared
  error on the early-stopping rows (floor 0.01), capped at 100, then root mean.
  The early-stopping rows are never calibration rows.

## Selection and evaluation

- **Leave-one-farm-out.** For each farm in turn, pick the candidate, score type,
  window and quantile with the highest pooled CARE on the other two farms; score the
  held-out farm once with that choice. Report each held-out farm and the mean.
- Apply exactly the same procedure to v1 (`scores` only) for comparison. Its
  leave-one-farm-out mean before this protocol was 0.576.
- **Adopt v2 only if** its leave-one-farm-out mean exceeds v1's and it is not worse
  on two of the three held-out farms. Otherwise v1 remains the reported model.
- The configuration for the app is then chosen on all three farms pooled. Its
  performance estimate is the leave-one-farm-out figure, not the pooled one.

## Limits stated in advance

Farm C informed the diagnosis behind this protocol, so leave-one-farm-out is a
fairer comparison than v1's single A+B to C split, not an untouched confirmation.
With 12, 6 and 27 faults per farm, differences below about 0.02 are noise. The
benchmark labels and windows are the publisher's.
