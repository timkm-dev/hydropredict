# CARE basic feasibility experiment v1

Frozen before fitting on 28 October 2026; related issue #21.

Question: does a simple normal-trained detector produce persistent alerts during
annotated degradation while remaining quiet on normal prediction files?

- Run all 22 Farm A and all 15 Farm B event files, with one independent model per
  file. No pooling, cross-file feature selection, hyperparameter search or outcome
  tuning. Farm C remains performance-uninspected. This evaluates adaptation to
  each turbine's supplied history, not unseen-turbine transfer. Repeated turbines
  are correlated; report turbine summaries and do not treat files as independent trials.
- Inputs: average channels whose dictionary marks neither angle nor counter;
  exclude metadata, status and labels. Drop constant/all-missing fitting channels.
  Median imputation fitted only on fitting observations.
- Partition the supplied training rows chronologically: first 80% fitting, last
  20% calibration. Use status 0/2 only for fitting/calibration normal selection.
  Prediction rows are never status-filtered. No rolling model inputs or interpolation.
- Isolation Forest: 100 trees, max_samples=256, random_state=42, n_jobs=2;
  at most 12,000 evenly spaced eligible fitting rows per file. Negative
  score_samples is anomaly score. Threshold is calibration-normal 99th percentile.
- Alert requires six consecutive 10-minute scores strictly above threshold.
  A missing timestamp, invalid input row (>20% missing selected channels) or
  non-exceedance resets the run. Six readings span 50 minutes; no future lookahead.
  Start prediction persistence afresh. Contiguous alert readings form one episode.
- Primary detection: alert during an anomaly interval at least 24 hours before
  its annotated end. Also report any within-interval alert, end-relative lead time,
  interval coverage, and alerts outside the interval (not automatically false alarms).
  An already-active alert at interval start is counted as within-interval detection
  but separately marked as pre-existing. End-relative lead is NOT verified failure lead.
- Normal controls: all prediction rows of normal-labelled files; report alerted
  fraction, observed days (rows/144), episodes per 30 observed days, and files
  with alerts. No-alarm baseline has zero detections and zero alarms.
- Screening gate, an engineering assumption rather than supervisor approval:
  at least 50% of anomaly files detected >=24h before annotated end, <=1 normal
  alarm episode per 30 observed days and <=1% alerted normal rows, separately
  for each farm. Require >=80% valid prediction rows per file; report every file.
- Passing supports retaining CARE for further research, not deployment. Failure
  rejects this basic configuration; it cannot prove the dataset contains no
  predictive signal. Do not tune after seeing results. Save models, source hashes,
  configuration, per-file evidence and aggregate results.

The supplied historical normal labels are retrospectively curated. This is a
benchmark feasibility study, not a prospective operational evaluation. No exact
archive release is asserted. Publisher status caveats and average-only guidance:
https://zenodo.org/records/15846963 .
