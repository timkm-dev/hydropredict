# HydroPredict

HydroPredict is an AI-driven predictive-maintenance prototype for hydroelectric power-plant equipment, supporting the academic investigation documented in `Munyiri_Timothy_Kimani_166095.docx`.

The target system comprises two distinct analytical workflows sharing common platform services.
The following describes the planned system, not completed model or application functionality:

1. **Primary Small Hydroelectric Plant (SHP) Workflow (Unsupervised Anomaly Detection)**:
   - Ingests operational telemetry ($t, V_1, \dots, V_6$) at nominal 5-minute sampling.
   - Learns normal operating regimes using **Isolation Forest** and **LSTM Autoencoder**.
   - Replays held-out historical observations chronologically without future lookahead.
   - Evaluates early-warning alerts against 59 ground-truth fault events (fault-event recall, false-alert frequency, and warning lead time).
2. **Supplementary Bearing Classification Workflow (Laboratory Benchmark)**:
   - Ingests Case Western Reserve University (CWRU) vibration recordings in MATLAB format.
   - Partitions whole recordings before window segmentation.
   - Extracts time-domain and frequency-domain features.
   - Classifies 4 bearing health states using **XGBoost** (Healthy, Inner Race Fault, Outer Race Fault, Rolling Element Fault).
3. **Shared Platform Services**:
   - Role-based authentication (ML Engineer vs. Maintenance Operator).
   - Provenance tracking, model versioning, and feature drift monitoring.
   - Durable persistence for telemetry records, analysis runs, alerts, and operator responses.
   - Web-based operator dashboard.

## Current Implementation Status

**Wind turbine early warning (CARE benchmark) — the main result.** One small
autoencoder per turbine learns its normal behaviour; a 6-hour smoothed score and
the benchmark's alarm counter raise warnings, and each alert names its top
deviating sensors. Settings were chosen on Farms A and B and confirmed once on
held-out Farm C: CARE score 0.578 on Farm C and 0.583 on all 95 datasets
(published autoencoder 0.66, random 0.5). See
[CARE autoencoder results](docs/care-autoencoder-results.md) and
[alert explanations](docs/care-explanation-results.md). The signed-in home page
replays these results: fleet overview, period replay and alert review
([frontend guide](frontend/README.md)). Farm C was trained on Colab via
`notebooks/train_care_autoencoder_colab.ipynb`.

CWRU bearing classification is now available locally at `/bearing.html`: a
fixed XGBoost model, MATLAB import, held-out examples, feature contributions and
account-scoped saved assessments. It scored 8/10 held-out load recordings and
5/13 on an unseen-size challenge; use it as a laboratory benchmark, not a validated
plant diagnostic. See [CWRU results](docs/cwru-results.md). Install `.[bearing]`.

Implemented and tested: SHP validation, chronological partitioning, segmented offline
preparation, training-scaler fitting and reuse, segment-aware features, fault alignment
and EDA. Each preparation run records provenance, row accounting and artifact checksums.

Also implemented: source-readiness and fault-coverage auditing, an exploratory six-channel
Isolation Forest, disjoint normal-reference threshold calibration, saved-bundle CPU
inference and retrospective event evaluation. See `docs/shp-readiness-audit.md` and
`docs/isolation-forest-baseline.md`. The first baseline detected only 1 of 5 eligible test
events and is not ready for operational warning use. Physical metadata was withdrawn after
the source-paper audit; exact channel semantics still need a source data dictionary.

Implemented LSTM engineering: gap-safe sequence construction, compact autoencoder,
normal-only early stopping and calibration, resumable training, CPU model export/loading,
portable input packaging, separate validation command and a thin Colab notebook.
The first Colab GPU run is complete and its frozen CPU scores have been reproduced.
It detected 1 of 3 adequately covered development faults; operational usefulness
remains unverified. See `docs/lstm-baseline-diagnostics.md` for results and limitations.

Frontend: the wind turbine early warning app is the home page; hydro plant
monitoring, bearing diagnosis and hydro evaluation evidence sit under **Research**.
Local email/password and optional Google accounts with profile onboarding are
implemented; alert and episode reviews persist per account in local SQLite.

Pending: live telemetry, production account recovery and role permissions, model
registry, drift monitoring and deployment. Fault-alignment labels are retrospective
analysis assumptions, not evaluated predictions.

## Planned Execution Environment

- **Local development and inference**: Target laptop is an Intel Core i5-1235U with 16 GB RAM. Data preparation and the Isolation Forest baseline run locally. The initial baseline fit and calibration took about 0.36 seconds; this excludes setup and is not a general performance guarantee.
- **LSTM training**: Implemented for Google Colab hosted GPU execution through `notebooks/train_lstm_colab.ipynb`. The notebook uploads a committed source/data package, calls repository training code and exports a complete bundle for local CPU inference. The first hosted run is complete; see the development diagnostics above. No live laptop connection or synchronisation service is required.

## Quick start on Windows

Run the setup script from PowerShell:

```powershell
Set-Location "C:\Users\tmuny\OneDrive\Desktop\NOTES\YEAR 4\4.1\ICS PROJECT II\hydropredict"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run all development checks (Ruff format, Ruff lint, Mypy strict types, Pytest):

```powershell
.\scripts\check.ps1
```

## Web app

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --local-accounts --port 8766
```

Open http://127.0.0.1:8766 and sign in. Build the CARE replay data once first;
see [the frontend guide](frontend/README.md).

## CLI Workflows

```powershell
# 1. Validate telemetry against canonical schema
hydropredict validate data\raw\shp_train.csv --accepted-output data\processed\shp_train_accepted.csv --quarantine-output data\quarantine\shp_train_rejected.csv

# 2. Partition telemetry chronologically
hydropredict partition --train-source data\raw\shp_train.csv --test-source data\raw\shp_test.csv --output-dir data\processed\partitions

# 3. Preprocess and fit scaler on training partition
hydropredict preprocess data\processed\partitions\shp_train.csv --output data\processed\shp_train_preprocessed.csv --fit-scaler data\processed\scaler.json --scaler-type standard

# Export scaled model measurements separately from physical feature inputs
hydropredict preprocess data\processed\partitions\shp_validation.csv --output data\processed\shp_validation_scaled.csv --scaler data\processed\scaler.json --scale

# 4. Extract segment-aware features
hydropredict extract-features data\processed\shp_train_preprocessed.csv --output data\processed\shp_train_features.csv

# 5. Retrospective fault alignment for evaluation
hydropredict align-faults data\raw\shp_test.csv data\raw\shp_faults.csv --output data\processed\shp_test_labeled.csv
```

For a complete repeatable preparation run, choose a new run directory:

```powershell
python scripts\prepare_shp.py --run-dir data\processed\runs\my-first-run
```

This command exports chronological partitions, unscaled prepared data, separately scaled
measurements, physical features and manifests. It refuses to overwrite an existing run.
Standard scaling fits only the prepared training measurements and is reused for validation
and testing. Domain features use engineering-unit measurements, not scaled values.
Do not pass timestamp or segment identifiers to a future estimator as numeric predictors.

The default cutoffs are 1 February and 1 May 2019. They exclude later training-source
rows and earlier test-source rows; manifests count these exclusions. Their suitability
for event coverage still needs evaluation. The optional published split is a comparison
only and does not satisfy the report's chronological evaluation requirement.

Preparation is currently offline and excludes entire extended outages. A future streaming
replay must define when bins close, when gaps become ineligible and how warm-up operates;
passing an offline forward-fill test does not prove full streaming causality.
Linear interpolation remains available for exploratory use and must not be used for
causal early-warning training or replay. Timestamp-naive source times are consistently
interpreted as UTC by the current contract; source timezone evidence remains to be confirmed.
Sensor descriptions and typical ranges also require a recorded source reference before
claiming validated physical semantics; range warnings are currently an explicit helper.

See `docs/data-preparation-status.md` for verification and remaining scientific limitations.

Audit and train the initial pointwise baseline using new artifact paths:

```powershell
python scripts\audit_shp.py --preparation-run data\processed\runs\2026-10-13-verified --output data\processed\runs\2026-10-13-verified\readiness_audit.json
python scripts\train_isolation_forest.py --preparation-run data\processed\runs\2026-10-13-verified --output models\isolation-forest-my-new-run
```

The baseline selects only raw V1..V6 and calibrates on a later separate normal-reference
subset; it does not consume the earlier prepared physical features. Source test excerpts
are fault-selected, so general false-alert frequency remains unverified. Save the reported
poor initial result rather than repeatedly tuning on the same final test period.

Earlier hydro experiments (temporal features, context lengths, signal and
precursor studies) are summarised in [the docs index](docs/README.md). Their code
was removed in the 1 November 2026 cleanup and remains at Git tag
`research-archive-2026-11-01`.

## Project Layout

```text
hydropredict/
|-- .github/workflows/       Continuous integration
|-- data/                    Dataset manifest, samples, raw and processed data stages
|-- docs/                    Protocols, results and decisions (start at docs/README.md)
|-- frontend/                Early warning web app, research pages and frontend tests
|-- models/                  Trained model bundles and scalers (ignored by Git)
|-- notebooks/               Colab training notebooks (LSTM, CARE autoencoder)
|-- scripts/                 Data preparation, training, evaluation and setup scripts
|-- src/hydropredict/        Current data, feature and analysis modules
|-- tests/                   Automated tests (unit, integration, regression)
|-- pyproject.toml           Package and tool configuration
`-- README.md                Project documentation
```

## Data locations

The CARE wind farm dataset (19 GB) lives outside the repository, by default in
`C:\data\CARE_To_Compare`, set by `CARE_DATA_DIR` in `.env`. CARE scripts use that setting
unless a folder is passed. On a new machine, download and verify it with progress
bars (resumable; checks the published checksums):

```powershell
.venv/Scripts/python.exe -m scripts.acquire_care
```

The trained CARE autoencoders exist only in ignored `.local/` folders. Back them up
after any retraining and keep the zip somewhere private, such as Google Drive:

```powershell
.venv/Scripts/python.exe -m scripts.backup_care_models                 # create
.venv/Scripts/python.exe -m scripts.backup_care_models --verify ZIP    # check
.venv/Scripts/python.exe -m scripts.backup_care_models --restore ZIP   # put back
```

## Data Policy

Raw data in `data/raw/` is immutable. Validation outputs belong in `data/processed/`, while invalid records belong in `data/quarantine/`. These directories are ignored by Git. Dataset checksums, provenance, and licenses are tracked in `data/dataset_manifest.csv`.
