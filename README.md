# HydroPredict

### 👉 Try it live: **[hydropredict-tkm26.azurewebsites.net](https://hydropredict-tkm26.azurewebsites.net)**

*A machine learning–based early fault warning system for renewable energy equipment.*

HydroPredict watches the sensor readings from renewable energy equipment, such as
wind turbines, and raises a warning when a machine starts behaving unlike its normal
self. Maintenance teams can then decide whether to inspect it before a small problem
becomes a breakdown.

This is a final-year university prototype (ICS Project II). It replays recorded
historical data; it is not yet connected to live equipment.

---

## Using the app

1. Open the [live site](https://hydropredict-tkm26.azurewebsites.net) and create an
   account with your email and a password.
2. Finish your profile and choose your **role in this app**:
   - **Maintenance Operator** (default): see what needs attention and act on alerts.
   - **ML Engineer**: everything above, plus the models and research pages.
3. Start on **What needs attention**. Each turbine has a plain-language status:

   | Status | Meaning |
   | --- | --- |
   | 🔴 **Inspect now** | Readings are well outside normal. Look at it soon. |
   | 🟡 **Keep an eye on** | Something is starting to drift. |
   | 🟢 **Normal** | Behaving as expected. |

4. Open an alert to see *which sensors* look unusual, then choose
   **Send a technician**, **Seen it**, or **Not a problem** (with a short reason).
   Your decisions are saved to your account.

> **Good to know:** the app spots *unusual behaviour*. It does not say exactly which
> part is failing, and it cannot promise a warning days in advance. Treat a warning
> as a reason to look, not a diagnosis.

---

## How well does it work?

The main model is a small **autoencoder per wind turbine**, trained only on normal
operation from the public [CARE to Compare](https://zenodo.org/records/15846963)
wind farm benchmark. When a turbine's readings stop matching what the model expects,
the score rises and, if it stays high, an alert is raised.

| Test | CARE score | For comparison |
| --- | --- | --- |
| Held-out Farm C (never used for tuning) | **0.578** | Random guessing: 0.5 |
| All 95 datasets | **0.583** | Published autoencoder: 0.66 |

So it is clearly better than chance but not yet as good as the best published
result. Details: [autoencoder results](docs/care-autoencoder-results.md) and
[alert explanations](docs/care-explanation-results.md).

**Bearing check (optional, lab benchmark).** An XGBoost classifier on the CWRU
bearing vibration dataset can name a likely bearing fault type. It scored 8/10 on
held-out recordings and 5/13 on an unseen bearing size, so it is a lab
demonstration, not a validated plant tool. See [CWRU results](docs/cwru-results.md).

---

## Run it on your own computer (Windows)

You need Python 3.11 or newer and Git.

```powershell
git clone https://github.com/timkm-dev/hydropredict.git
cd hydropredict
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

Start the app:

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --local-accounts --port 8766
```

Then open http://127.0.0.1:8766. The first time, build the wind farm replay data;
the [frontend guide](frontend/README.md) explains how.

**Optional: Google sign-in.** Copy `.env.example` to `.env` and fill in
`GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` from a Google Cloud OAuth client.

**Run the checks** (formatting, linting, type checks and tests) before committing:

```powershell
.\scripts\check.ps1
```

---

## For developers and researchers

### Project layout

```text
hydropredict/
|-- frontend/             The web app (pages, styles, frontend tests)
|-- src/hydropredict/     Python package: web server, accounts, data and models
|-- scripts/              Setup, data download, training and evaluation scripts
|-- notebooks/            Google Colab training notebooks
|-- docs/                 Protocols, results and decisions (start at docs/README.md)
|-- tests/                Automated tests
|-- data/                 Dataset manifest and data stages (raw data not in Git)
|-- models/               Trained models (not in Git)
|-- Dockerfile            Image used for the Azure deployment
`-- .github/workflows/    CI checks and the manual Azure deploy
```

### Data

The CARE wind farm dataset (19 GB) is kept outside the repository, by default in
`C:\data\CARE_To_Compare` (set `CARE_DATA_DIR` in `.env`). Download and verify it,
with progress bars and resume support:

```powershell
.venv/Scripts/python.exe -m scripts.acquire_care
```

Raw data in `data/raw/` is never edited. Cleaned data goes to `data/processed/` and
rejected rows to `data/quarantine/`. These folders are ignored by Git; checksums,
sources and licences are recorded in `data/dataset_manifest.csv`.

### Model training and backups

Heavy training runs on Google Colab (see `notebooks/train_care_autoencoder_colab.ipynb`).
The trained CARE autoencoders live only in ignored `.local/` folders, so back them up
after retraining and keep the zip somewhere private:

```powershell
.venv/Scripts/python.exe -m scripts.backup_care_models                 # create
.venv/Scripts/python.exe -m scripts.backup_care_models --verify ZIP    # check
.venv/Scripts/python.exe -m scripts.backup_care_models --restore ZIP   # put back
```

### Earlier hydroelectric (SHP) research

The project began with a small hydroelectric plant (SHP) dataset. That work built a
tested data pipeline (validation, chronological splits, scaling, features and fault
alignment) plus Isolation Forest and LSTM autoencoder baselines. The baselines caught
only 1 of 5 and 1 of 3 eligible faults, which is why the main focus moved to the larger
CARE wind farm benchmark. Results and limitations are in the [docs index](docs/README.md),
[data preparation status](docs/data-preparation-status.md),
[Isolation Forest baseline](docs/isolation-forest-baseline.md) and
[LSTM diagnostics](docs/lstm-baseline-diagnostics.md). Older experiment code is kept at
Git tag `research-archive-2026-11-01`.

<details>
<summary>SHP command-line workflow</summary>

```powershell
# 1. Validate telemetry against the canonical schema
hydropredict validate data\raw\shp_train.csv --accepted-output data\processed\shp_train_accepted.csv --quarantine-output data\quarantine\shp_train_rejected.csv

# 2. Partition telemetry chronologically
hydropredict partition --train-source data\raw\shp_train.csv --test-source data\raw\shp_test.csv --output-dir data\processed\partitions

# 3. Preprocess and fit the scaler on the training partition
hydropredict preprocess data\processed\partitions\shp_train.csv --output data\processed\shp_train_preprocessed.csv --fit-scaler data\processed\scaler.json --scaler-type standard

# Export scaled model measurements separately from physical feature inputs
hydropredict preprocess data\processed\partitions\shp_validation.csv --output data\processed\shp_validation_scaled.csv --scaler data\processed\scaler.json --scale

# 4. Extract segment-aware features
hydropredict extract-features data\processed\shp_train_preprocessed.csv --output data\processed\shp_train_features.csv

# 5. Retrospective fault alignment for evaluation
hydropredict align-faults data\raw\shp_test.csv data\raw\shp_faults.csv --output data\processed\shp_test_labeled.csv
```

A complete, repeatable preparation run (refuses to overwrite an existing run):

```powershell
python scripts\prepare_shp.py --run-dir data\processed\runs\my-first-run
```

Audit the run and train the pointwise Isolation Forest baseline:

```powershell
python scripts\audit_shp.py --preparation-run data\processed\runs\2026-10-13-verified --output data\processed\runs\2026-10-13-verified\readiness_audit.json
python scripts\train_isolation_forest.py --preparation-run data\processed\runs\2026-10-13-verified --output models\isolation-forest-my-new-run
```

Important caveats:

- Default cutoffs are 1 February and 1 May 2019; manifests count excluded rows. The
  optional published split is for comparison only.
- Scaling is fitted on training data only and reused for validation and testing.
  Domain features use engineering units, not scaled values. Do not feed timestamps or
  segment IDs to a model as numeric inputs.
- Preparation is offline and drops long outages. Linear interpolation is for
  exploration only, never for causal early-warning training or replay.
- Timestamps without a timezone are treated as UTC, and sensor meanings still need a
  source data dictionary.
- Fault-alignment labels are retrospective analysis assumptions, not predictions.

</details>

### Not done yet

Live telemetry, production account recovery, a model registry and drift monitoring.
