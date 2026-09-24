# HydroPredict

HydroPredict is a predictive-maintenance system for hydroelectric power-plant
equipment. It will ingest time-series telemetry, identify anomalous operating
conditions, explain alerts, and support operator review.

## Sprint 1 scope

This repository currently provides:

- an isolated and repeatable Python development environment;
- a `src`-based, testable application layout;
- CSV validation for the required `t, V1, V2, V3, V4, V5, V6` schema;
- separate locations for raw, processed, and quarantined data;
- a dataset provenance manifest;
- linting, formatting, type-checking, tests, pre-commit hooks, and CI automation;
- Git version control scoped to this application only.

## Quick start on Windows

Run the setup script from PowerShell:

```powershell
Set-Location "C:\Users\tmuny\OneDrive\Desktop\NOTES\YEAR 4\4.1\ICS PROJECT II\hydropredict"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

Activate the environment in future terminals:

```powershell
.\.venv\Scripts\Activate.ps1
```

Validate the included sample telemetry:

```powershell
hydropredict validate data\sample\telemetry_sample.csv
```

Write accepted and rejected records to the appropriate local directories:

```powershell
hydropredict validate data\sample\telemetry_sample.csv `
  --accepted-output data\processed\accepted.csv `
  --quarantine-output data\quarantine\rejected.csv
```

Run all development checks:

```powershell
.\scripts\check.ps1
```

## Project layout

```text
hydropredict/
|-- .github/workflows/       Continuous integration
|-- data/                    Dataset manifest, samples, and local data stages
|-- docs/                    Architecture decisions and project documentation
|-- models/                  Local trained models (ignored by Git)
|-- scripts/                 Environment and quality-check automation
|-- src/hydropredict/        Application source code
|-- tests/                   Automated tests
|-- pyproject.toml           Package and tool configuration
`-- README.md                Project entry point
```

## Data policy

Raw data is immutable. Never edit files in `data/raw` in place. Validation and
cleaning output belongs in `data/processed`; rejected records belong in
`data/quarantine`. These local data directories are excluded from Git to avoid
publishing large or restricted datasets. Dataset origin, licensing, retrieval
date, and checksums must be recorded in `data/dataset_manifest.csv`.

## Git workflow

The stable branch is `main`. Develop each unit of work on a short-lived branch:

```powershell
git switch -c feature/telemetry-ingestion
git add .
git commit -m "feat: add telemetry ingestion"
```

See `CONTRIBUTING.md` for the full quality workflow.

