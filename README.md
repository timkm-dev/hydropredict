# HydroPredict

HydroPredict is an AI-driven predictive-maintenance prototype for hydroelectric power-plant equipment, supporting the academic investigation documented in `Munyiri_Timothy_Kimani_166095.docx`.

The system comprises two distinct analytical workflows sharing common platform services:

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

## Execution Environment & Hardware Profile

- **Local Development & Inference**: Tested for Intel Core i5-1235U CPU with 16 GB RAM. Ingestion, preprocessing, feature extraction, Isolation Forest, XGBoost, and CPU inference execute locally.
- **Deep Learning Training**: The LSTM Autoencoder is trained in a Google Colab hosted GPU runtime using the repository implementation called via `notebooks/train_lstm_colab.ipynb`. The resulting bundle is exported for local CPU loading and inference.

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

## CLI Workflows

```powershell
# 1. Validate telemetry against canonical schema
hydropredict validate data\raw\shp_train.csv --accepted-output data\processed\shp_train_accepted.csv --quarantine-output data\quarantine\shp_train_rejected.csv

# 2. Partition telemetry chronologically
hydropredict partition data\raw\shp_train.csv data\raw\shp_test.csv --output-dir data\processed\partitions

# 3. Preprocess and fit scaler on training partition
hydropredict preprocess data\processed\partitions\shp_train_part.csv --output data\processed\shp_train_preprocessed.csv --scaler-output data\processed\scaler.json

# 4. Extract segment-aware features
hydropredict extract-features data\processed\shp_train_preprocessed.csv --output data\processed\shp_train_features.csv

# 5. Retrospective fault alignment for evaluation
hydropredict align-faults data\raw\shp_test.csv data\raw\shp_faults.csv --output data\processed\shp_test_labeled.csv
```

## Project Layout

```text
hydropredict/
|-- .github/workflows/       Continuous integration
|-- data/                    Dataset manifest, samples, raw and processed data stages
|-- docs/                    Architecture decisions and requirement traceability
|-- models/                  Trained model bundles and scalers (ignored by Git)
|-- notebooks/               Colab GPU training notebooks
|-- scripts/                 Environment setup and verification scripts
|-- src/hydropredict/        Application source code (data, features, models, app)
|-- tests/                   Automated tests (unit, integration, regression)
|-- pyproject.toml           Package and tool configuration
`-- README.md                Project documentation
```

## Data Policy

Raw data in `data/raw/` is immutable. Validation outputs belong in `data/processed/`, while invalid records belong in `data/quarantine/`. These directories are ignored by Git. Dataset checksums, provenance, and licenses are tracked in `data/dataset_manifest.csv`.
