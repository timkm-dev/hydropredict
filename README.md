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
Full Colab GPU training and LSTM fault-based performance are still pending; local smoke
tests are not scientific results. See `docs/lstm-colab-workflow.md` for the owner steps.

Implemented frontend: responsive local dashboard with synthetic preview data, telemetry
CSV import/export, six-channel trend exploration, temporary illustrative alert review,
and documented model readiness. See `frontend/README.md`. It is not connected to live
telemetry, model inference, authentication, or durable application records.

Pending: full LSTM GPU execution and comparison, CWRU acquisition and XGBoost, sequential replay,
application-integrated event evaluation, authentication,
durable application records, dashboard backend integration, model registry, drift monitoring and deployment.
Fault-alignment labels are retrospective analysis assumptions, not evaluated predictions.

## Planned Execution Environment

- **Local development and inference**: Target laptop is an Intel Core i5-1235U with 16 GB RAM. Data preparation and the Isolation Forest baseline run locally. The initial baseline fit and calibration took about 0.36 seconds; this excludes setup and is not a general performance guarantee.
- **LSTM training**: Implemented for Google Colab hosted GPU execution through `notebooks/train_lstm_colab.ipynb`. The notebook uploads a committed source/data package, calls repository training code and exports a complete bundle for local CPU inference. Hosted GPU execution remains pending. No live laptop connection or synchronisation service is required.

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

## Frontend dashboard

Start the dashboard from the repository root and open http://127.0.0.1:8765:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory frontend
```

The initial screen uses clearly labeled synthetic data. Import a telemetry CSV
containing `t,V1,V2,V3,V4,V5,V6` to explore your own observations locally.
See [dashboard instructions](frontend/README.md) for controls and integration limits.

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

The validation-only temporal feature and threshold comparison is documented in
`docs/validation-model-comparison.md`. It found a limited temporal signal but substantial
warm-up coverage loss; no replacement model was promoted. Install `.[analysis]` for the
optional Matplotlib diagnostics and run `scripts/compare_validation_models.py` as documented.

Observed development context can now be reconstructed from both source files with exact
timestamp provenance and no synthetic filling. The frozen-model validation check restored
92 pre-fault temporal score rows, but adequate temporal event coverage remained at two
events and neither was detected. See `docs/continuous-context-evaluation.md` for the fixed
experiment, limitations and reproduction command. No default model was changed.

The declared 15/30/120-minute context comparison is complete; see
`docs/short-context-comparison.md`. Both shorter designs detected one of three adequately
covered validation faults, versus zero of two for the two-hour model. The 30-minute design
is retained as a development comparator, with no operational performance claim or default
replacement. Next planned milestone: compact LSTM Autoencoder and Colab notebook.

## Project Layout

```text
hydropredict/
|-- .github/workflows/       Continuous integration
|-- data/                    Dataset manifest, samples, raw and processed data stages
|-- docs/                    Architecture decisions and requirement traceability
|-- frontend/                Responsive local research dashboard and data validation tests
|-- models/                  Trained model bundles and scalers (ignored by Git)
|-- notebooks/               Planned Colab notebook directory (not implemented)
|-- scripts/                 Environment setup and verification scripts
|-- src/hydropredict/        Current data, feature and analysis modules
|-- tests/                   Automated tests (unit, integration, regression)
|-- pyproject.toml           Package and tool configuration
`-- README.md                Project documentation
```

## Data Policy

Raw data in `data/raw/` is immutable. Validation outputs belong in `data/processed/`, while invalid records belong in `data/quarantine/`. These directories are ignored by Git. Dataset checksums, provenance, and licenses are tracked in `data/dataset_manifest.csv`.
