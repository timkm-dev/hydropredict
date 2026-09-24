# Contributing to HydroPredict

## Workflow

1. Update `main` and create a focused branch named `feature/...`, `fix/...`, or
   `docs/...`.
2. Add or update tests with every behavioural change.
3. Run `scripts/check.ps1` before committing.
4. Use concise commit messages such as `feat: validate telemetry timestamps`.
5. Merge only when automated checks pass.

## Definition of done

A change is complete when it is documented, typed, tested, formatted, free of
secrets, and reproducible from a clean checkout.

## Data changes

Do not commit confidential, licensed, or large raw datasets. Update the dataset
manifest whenever a data source or version changes. Record a SHA-256 checksum
for every locally acquired source file.

