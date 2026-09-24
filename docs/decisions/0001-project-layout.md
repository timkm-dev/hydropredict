# ADR 0001: Use a Python `src` layout

## Status

Accepted on 2026-09-24.

## Context

HydroPredict needs independently testable ingestion, preprocessing, inference,
explanation, alerting, and presentation components. Sprint 1 also requires a
clean, reproducible development environment.

## Decision

The Python package lives under `src/hydropredict`, automated tests live under
`tests`, and mutable datasets and model artefacts are excluded from Git. Package,
test, lint, formatting, and type-check settings are centralised in
`pyproject.toml`.

## Consequences

Tests exercise the installed package rather than accidentally importing code
from the repository root. New components can be added as package modules without
mixing application code with research documents or local datasets.

