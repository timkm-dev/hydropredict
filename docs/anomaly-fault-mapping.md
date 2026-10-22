# Anomalies alongside faults: proof of concept

22 October 2026. This experiment maps the existing frozen LSTM decisions to
recorded faults. It does not train a model, select a threshold, change alert
rules or evaluate a final-test period. GitHub access was unavailable; no issue
was read, updated or closed.

## Result and interpretation

On February–April development data there are 6,469 saved score endpoints from
6,742 source observations. The frozen threshold is 0.3044476807117462.

- 84 anomalous readings form 29 episodes.
- One anomalous reading falls in the six hours before a recorded fault.
- The other 83 fall outside all recorded six-hour precursor windows. These
  cannot automatically be called false alarms: labels and maintenance context
  are incomplete, and anomalies need not precede a recorded failure.
- Nine fault timestamps remain visible: one with a warning, two missed, six
  with insufficient scored coverage. Only three are adequately scored.
- February 2 has a single crossing about 5h 54m before the fault. It appears near
  sequence startup and is weaker evidence than a sustained precursor.
- February 3 and April 2 at 09:06:14 have adequate scored coverage but no warning.
- 80 of 5,904 publisher-selected nominal-reference scores are anomalous (1.36%).
  That selected-reference percentage is not operational false-alarm probability.

An anomaly is a strict score-above-threshold decision. A preceding warning is a
decision at least zero and at most six hours before the fault, with the fault
instant itself excluded. Eligibility requires 80% of fault-relative five-minute
slots and a score within ten minutes. Overlapping fault windows share group IDs
and can share a warning; they are not independent failure examples.

## Frontend

Choose **Anomalies & faults** in the dashboard navigation, or open `/evidence.html`.
The page has summary counts, the whole-study anomaly/fault timeline, a fault
selector, score-versus-threshold inspection, gap shading, explanatory outcomes,
all nine event rows, and evidence provenance. The six-hour warning band is
distinct from optional 12-hour context and the half-hour post-fault context.
Post-fault readings never count as warnings. Curves do not bridge gaps over
450 seconds; missing scores never imply normal equipment.

The page uses real, locally generated evidence rather than demo equipment or
uploaded dashboard telemetry. It works with the inference server or a static
local preview. It does not require model retraining or executing PyTorch.

```powershell
.venv/Scripts/python.exe scripts/map_anomalies_to_faults.py
python -m http.server 8772 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:8772/evidence.html`. The generator requires the original
ignored local Colab input package and frozen diagnostics. It verifies their
checksums and refuses final-test data, invalid timestamps or inconsistent saved
decisions. `--inputs`, `--diagnostics` and `--output` support explicit paths.

`frontend/evidence-data.json` is ignored by Git. It contains timestamps, normalized
scores and event association evidence, not raw V1–V6 readings. It is served as
a local static asset; deployers must not publish it as part of a public site
without authorization to disclose this evidence. No datasets or model bundles
are added to commits. Missing evidence produces setup instructions, not invented
results. Existing inference and authentication behavior are unchanged.
