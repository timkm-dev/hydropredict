# Corrected supervised precursor experiment

24 October 2026; issues #19 and #20. **Reject this candidate and retain the
existing LSTM as the anomaly-monitoring baseline.** Neither model establishes
reliable fault prediction.

The corrected package restores earlier fault examples. It provides 25,855
fitting endpoints (1,473 positive six-hour labels), 7,424 January calibration
endpoints (370 positives), and 6,452 February-April development endpoints
(327 positives). Feature coverage admits 18/31 fitting faults, 3/8 calibration
faults, and 3/9 development faults. Ineligible faults remain in the event reports.
The fitting faults with usable coverage span 15 overlapping-window groups.

The fixed 36-feature gradient-boosted classifier used August-December fitting
and January-only threshold calibration. Threshold: 0.7193782627106018, with
three consecutive exceedances required. January selected-nominal point alerts
after persistence were 0.61%. See [the prior protocol](supervised-precursor-protocol.md).

| February-April result | Classifier | Frozen LSTM, matched endpoints |
| --- | ---: | ---: |
| Scored endpoints | 6,452 | 6,452 |
| Eligible fault groups warned | 0/3 | 1/3 |
| Selected-nominal alerts | 7/5,887 (0.12%) | 79/5,887 (1.34%) |
| Selected-nominal alert episodes | 2 | 25 |
| Episodes per observed nominal day | 0.098 | 1.231 |

The native LSTM has 6,469 endpoints, 80/5,904 selected-nominal alerts (1.36%),
and the same 1/3 warned groups. Matching retains each model's original alert
decisions; it does not recalculate persistence after filtering endpoints.
The classifier misses all three usable faults, including the baseline's
February 2 warning. Its fewer alerts therefore do not constitute improved warning.

Post-run diagnosis: point-level ROC AUC is 0.902 in fitting, 0.522 in January,
and 0.503 in February-April. Average precision is 0.447, 0.049 and 0.054,
respectively. These are descriptive, correlated-reading metrics, not independent
event estimates. The loss of ranking ability already in January suggests weak
temporal generalization; simply lowering a threshold is not a supported remedy.

The next bounded investigation tests temporal transfer on earlier chronological
folds with regularized models and relative-change features. February-April stays
out of that selection. Any later January confirmation is development evidence,
since January has already been used for calibration. Independent validation and
fault/operating context remain under #21.

## Artifacts and verification

- Aggregate evidence: `docs/results/supervised-precursor-v1/`.
- Verified input: `artifacts/supervised-precursor-v1-verified/`.
- Model, frozen configuration and scores: `models/supervised-precursor-v1/`.
- Portable input: `artifacts/hydropredict-supervised-precursor-v1.zip` and its
  `.sha256` sidecar. No May-onward measurements are packaged.
- The first incomplete build at `artifacts/supervised-precursor-v1/` is marked
  INVALID: a regression test caught a timestamp-unit error. No model used it.
- Serialized-model predictions replayed exactly. All 117 Python tests passed;
  causal-prefix, gap-reset, target boundary, overlapping-group purge, duplicate
  conflict and persistence tests cover this work.

The native source files contain 41,108 observations before May. Boundary purging
removes 72 fitting observations; feature warm-up removes 837 fitting, 178
calibration and 290 development observations. There are no fabricated readings.
Zero target means no recorded fault in the next six hours, not confirmed health.

## Local reproduction

Run from the project root with the project virtual environment. Use fresh output
directories; existing evidence is never overwritten.

```powershell
.venv/Scripts/python.exe scripts/supervised_precursor.py build --output artifacts/supervised-new
.venv/Scripts/python.exe scripts/supervised_precursor.py run --package artifacts/supervised-new --diagnostics models/lstm-diagnostics-2026-10-15-v1 --output models/supervised-new
.venv/Scripts/python.exe scripts/package_supervised_replay.py --package artifacts/supervised-new --diagnostics models/lstm-diagnostics-2026-10-15-v1 --output artifacts/supervised-new.zip
```

## Optional Colab reproduction

No new Colab run is needed for the results above. To reproduce, upload **only**
`hydropredict-supervised-precursor-v1.zip` to a fresh notebook, verify its SHA-256
against the accompanying sidecar, and run:

```python
import hashlib, os, zipfile
from google.colab import files

uploaded = files.upload()
assert list(uploaded) == ["hydropredict-supervised-precursor-v1.zip"]
payload = next(iter(uploaded.values()))
assert (
    hashlib.sha256(payload).hexdigest()
    == "7fa4bfb5ff7b1956c97181697a83a224f64de87d440b5a87f6fe02e6b297e910"
)
with zipfile.ZipFile("hydropredict-supervised-precursor-v1.zip") as z:
    z.extractall("supervised")
with zipfile.ZipFile("supervised/package/code.zip") as z:
    z.extractall("supervised/code")
os.environ["PYTHONPATH"] = os.path.abspath("supervised/code/src")
```

```python
%pip install numpy==2.5.3 pandas==3.0.6 scikit-learn==1.9.1 joblib
!python supervised/code/scripts/supervised_precursor.py run --package supervised/package --diagnostics supervised/baseline --output supervised/reproduced
```

The canonical run used Python 3.14.4; cross-version bitwise agreement is not
promised. If the Colab runtime cannot install those recorded versions, use the
local environment instead of silently changing the experiment. Compare
`results.json` with the committed aggregate evidence. Data packages stay local
and ignored; do not upload them to GitHub.
