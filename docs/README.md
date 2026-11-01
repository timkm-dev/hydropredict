# Documentation index

Start with the first section; the rest is supporting evidence.

## Main result: wind turbine early warning (CARE benchmark)

1. [CARE score results](care-score-results.md): why the benchmark metric replaced
   our own pass/fail gates, and how the earlier models scored (about 0.48).
2. [Autoencoder protocol](care-autoencoder-protocol.md) and
   [results](care-autoencoder-results.md): 0.578 on held-out Farm C, 0.583 on all
   95 datasets (published autoencoder 0.66, random 0.5).
3. [Explanation protocol](care-explanation-protocol.md) and
   [results](care-explanation-results.md): do alerts name the failed component?
4. [Dataset assessment](care-dataset-assessment.md): what CARE contains and its terms.

## Supporting modules

- Bearing diagnosis (CWRU laboratory benchmark): [protocol](cwru-protocol.md),
  [results](cwru-results.md).
- Hydro plant (SHP) pipeline: [data preparation](data-preparation-status.md),
  [readiness audit](shp-readiness-audit.md),
  [source code and paper review](shp-source-code-review.md),
  [dataset suitability verdict](dataset-suitability-verdict.md),
  [Isolation Forest baseline](isolation-forest-baseline.md),
  [LSTM diagnostics](lstm-baseline-diagnostics.md),
  [LSTM Colab workflow](lstm-colab-workflow.md),
  [anomalies against recorded faults](anomaly-fault-mapping.md),
  [operator workflow](shp-operator-workflow.md).
- [ADR 0001: project layout](decisions/0001-project-layout.md).

## Earlier experiments (superseded)

These record what was tried and why it was not adopted. Their results stand, but
the code was removed on 1 November 2026 to keep the project focused. Recover it with
`git checkout research-archive-2026-11-01 -- scripts tests notebooks src`.

- CARE, before the autoencoder: basic Isolation Forest
  ([protocol](care-baseline-protocol.md), [results](care-baseline-results.md)),
  condition-aware residuals ([protocol](care-residual-protocol.md),
  [results](care-residual-results.md)), supervised trajectories
  ([protocol](care-supervised-protocol.md), [results](care-supervised-results.md)),
  model-family comparison ([protocol](care-model-comparison-protocol.md),
  [calibration addendum](care-model-calibration-addendum.md)).
- Hydro plant: [validation model comparison](validation-model-comparison.md),
  [continuous context](continuous-context-evaluation.md),
  [short context](short-context-comparison.md),
  signal investigation ([protocol](signal-investigation-protocol.md),
  [results](signal-investigation-results.md)),
  [Colab investigation](colab-investigation.md), [Colab hybrid review](colab-hybrid-review.md),
  normal trends ([protocol](normal-trend-protocol.md), [results](normal-trend-results.md)),
  supervised precursors ([protocol](supervised-precursor-protocol.md),
  [results](supervised-precursor-results.md)),
  [earlier-history backtest](earlier-backtest-protocol.md),
  [precursor transfer](precursor-transfer-results.md),
  extended Isolation Forest context ([protocol](eif-context-protocol.md),
  [results](eif-context-results.md)), [sequence patterns](sequence-pattern-results.md),
  [model improvement research](model-improvement-research.md).
