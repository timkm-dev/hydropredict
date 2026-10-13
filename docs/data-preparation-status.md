# SHP data preparation verification

Updated 13 October 2026. This records preparation evidence, not trained-model performance.

Verified local artifacts: `data/processed/runs/2026-10-13-verified/run_manifest.json`,
produced from source commit `74bbc8d`. All 56 tests passed, with Ruff lint, formatting and
strict Mypy checks passing. Data artifacts remain ignored by Git; source and documentation
changes are committed separately by meaningful action.

The standard scaler uses the training mean and sample standard deviation, with a unit
scale for constant or single-observation columns. Artifacts must have matching nonempty
parameter columns, finite centers and finite positive scales. Validation and testing
reuse the saved training parameters. The existing robust scaler remains an explicit option.

Chronological cutoffs default to 1 February 2019 and 1 May 2019. Training uses only the
published normal-operation source before the first cutoff; validation and final testing
use the other source in their respective later intervals. Fault timestamps do not select
training rows. All chronological partitions must be nonempty, and cutoffs must be ordered.
Exports preserve fractional timestamp precision. Source hashes, actual paths and excluded
record counts are retained in the manifest. The published split is labelled nonchronological.

The verified real-data preparation produced:

| Partition | Selected observations | Bin collisions | Filled steps | Prepared rows | Segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 30,807 | 56 | 146 | 30,897 | 77 |
| Validation | 605 | 0 | 5 | 610 | 10 |
| Final test | 733 | 15 | 20 | 738 | 10 |

All numeric prepared, scaled and feature values were checked for finiteness. Physical
features were computed before scaling, and all exports were hashed. The pipeline retains
89 numeric feature columns, including the six raw measurements, plus timestamp and segment
coordinates. Coordinates are metadata, not estimator predictors.

Use `python scripts/prepare_shp.py --run-dir data/processed/runs/<new-name>` to reproduce
preparation. Raw files and prior artifacts are preserved. Each run has a source commit,
partition manifest, row accounting, training scaler and output checksums. Run from a clean
committed checkout when publishing evidence so the recorded commit identifies the code.

Remaining work before scientific conclusions:

- Confirm recorded evidence for source sensor semantics, units, typical ranges and timezone.
- Verify normal-operation training suitability and held-out event coverage. Cutoff-interval
  fault counts are not automatically eligible observable fault events.
- Implement online replay eligibility. Offline segment preparation uses complete gap
  information and must not silently rewrite previously emitted predictions.
- Construct LSTM sequences within segments, define warm-up and freeze model-specific schemas.
- Add model training, the thin Colab GPU notebook and local CPU model-bundle loading.
- Establish threshold calibration without final-test tuning or fault-derived predictors.
- Implement event matching, alert grouping and performance evaluation.

No training-time, GPU, model-quality or deployment results are claimed by this verification.
