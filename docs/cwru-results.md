# CWRU bearing classification — issue #9

Verification: 137 Python tests passed before the final upload/provenance test was
added; the subsequent bearing/dashboard suite passed all 17 tests. Existing
frontend tests (19), changed-code lint/format checks, JavaScript syntax and Python
type checks passed. A separate real-model HTTP replay verified 100.mat and the
multi-signal 175.mat upload, TreeSHAP responses, persistence across store reopen,
and rejection of unauthenticated and other-owner access. Aggregate evidence is
`docs/results/cwru-v1/api-replay.json`. Browser visual review remains outstanding;
the page was opened through the Windows browser handler.

The fixed XGBoost baseline classified **8/10 held-out load recordings correctly (80%, macro-F1 0.8375)**. Validation at load 2: 9/10 correct, macro-F1 0.9143.

The stronger unseen-size challenge classified **5/13 correctly (38.46%, macro-F1 0.375)**. It classified the healthy recording correctly but assigned all twelve 14-mil fault recordings to rolling-element fault. This is evidence that the baseline does not generalize reliably across unseen fault sizes. No further tuning was performed after these results. The served model is the fixed primary model, explicitly presented as a laboratory benchmark, not an approved plant diagnostic.

## Primary recording-level predictions

| File | Reference | Prediction |
|---|---|---|
| 100.mat | Healthy | Healthy |
| 112.mat | Inner race fault | Inner race fault |
| 125.mat | Rolling element fault | Rolling element fault |
| 138.mat | Outer race fault | Outer race fault |
| 177.mat | Inner race fault | Inner race fault |
| 192.mat | Rolling element fault | Rolling element fault |
| 204.mat | Outer race fault | Rolling element fault |
| 217.mat | Inner race fault | Rolling element fault |
| 229.mat | Rolling element fault | Rolling element fault |
| 241.mat | Outer race fault | Outer race fault |

See `docs/results/cwru-v1/results.json` for all per-class precision/recall/F1 values, confusion matrices, validation results, window metrics and challenge predictions. Confusion-matrix order: Healthy, Inner race fault, Outer race fault, Rolling element fault. Windows are correlated; recording counts are the reporting unit. Ten test files include only one healthy recording. Do not interpret 80% as guaranteed accuracy on arbitrary recordings.

## Preparation and provenance

Forty official MATLAB files were downloaded from the university's normal baseline and 48 kHz drive-end pages. Source URLs, labels, loads, diameters, retrieval times, access-term caveats and file SHA-256 values are in `data/cwru_manifest.csv`. Only 7/14/21-mil seeded faults and outer-race position @6 are included. Sampling-rate convention and selection limits are recorded in `docs/cwru-protocol.md`.

Files 99, 175 and 217 contain extra arrays belonging to other recordings. The loader selects X099_DE_time, X175_DE_time and X217_DE_time respectively, using original numeric recording names. Extra arrays are never used as independent examples. All 40 selected signal hashes are unique. Filename identifies a signal only; filename, RPM, load, size and reference labels are not predictor features. Each recording is partitioned before non-overlapping window extraction. Every signal is resampled 48 → 12 kHz with anti-alias filtering. Window size is 4,096; unused trailing samples are recorded separately.

Primary fits on load 0/1, reports load 2 validation and load 3 test. These recordings may share seeded bearings across loads. The separate size challenge holds every 14-mil fault family out of fitting and holds normal load 3 out. Neither establishes new-bearing or hydro-plant generalization. The primary model is not refitted on validation/test recordings for serving.

Two runs stopped before fitting while identifying ambiguous MAT arrays and the official 175.mat file's size (17.85 MB). Signal selection and the upload limit were corrected before any model results were examined. Numerical model settings were unchanged. Exact saved-model probability replay passed on every evaluation recording. `models/cwru-xgb-v1/` contains both model bundles, environment, code snapshot and checksums. Raw files and bundles remain ignored.

## Application

`/bearing.html` provides import of a 48 kHz CWRU DE .mat recording or selection of a held-out benchmark example. It shows predicted class, four model-support scores, window agreement and three exact TreeSHAP contributions averaged over recording windows for the winning class. Those contributions are classifier margin contributions, not causal diagnoses. Model support is not calibrated confidence.

The server stores source bytes by checksum under ignored `data/raw/cwru-imports/` and account-scoped assessment records in `data/bearing_assessments.sqlite3`. SQLite tables are initialized automatically, separate from SHP records. Stop the server before backing up both account and assessment databases plus imported source files and model bundle. There is no shared-role access, retention UI or public deployment claim. The same-account history survives restart. Wrong-rate inputs are rejected based on required user metadata; rate cannot be inferred reliably from an arbitrary MAT file. This model does not validate other bearings, fan-end sensors or other sample rates. Keep numeric filenames for multi-signal official files.

Install `.[bearing]` or XGBoost >=3.1,<4. Reproduce acquisition with `scripts/acquire_cwru.py` and training with `python -m scripts.train_cwru` on a clean output directory. User authorization to retain CWRU is recorded; supervisor milestone approval remains unverified. No issue closure or publication is claimed.
