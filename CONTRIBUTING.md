# Contributing to HydroPredict

## Workflow

Track work in [GitHub Issues](https://github.com/timkm-dev/hydropredict/issues).
Read the relevant issue before implementation, reference its number in commits/PRs,
and record verification evidence before closing it. Sprint milestones organise the
remaining review requirements; scope decisions and owner verification are tracked
explicitly rather than assumed complete. See `AGENTS.md` for agent task tracking.

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

