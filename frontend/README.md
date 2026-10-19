# HydroPredict dashboard

A responsive, dependency-free frontend for the research project. It provides an
overview, SVG telemetry charts, a six-channel explorer, illustrative alert review,
and a model readiness workspace. Python analysis modules remain unchanged.

From the repository root, start the local CPU LSTM inference server. The Python
environment must have CPU PyTorch and the project's LSTM extra installed:

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --port 8766
```

Open **http://127.0.0.1:8766** in a browser. Stop the server with Ctrl+C.
Only frontend assets and inference endpoints are served, not datasets or models.
The default bundle is the local checksum-verified copy at
`models/lstm-diagnostics-2026-10-15-v1/frozen_bundle_copy`. Model files are ignored
by Git. Supply a trusted exported bundle on another machine:

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --bundle models/my-run/bundle --port 8766
```

Missing or incompatible bundles produce a startup error; no model is trained or
silently substituted. A static-only preview remains available using
`python -m http.server 8765 --bind 127.0.0.1 --directory frontend`, with inference
explicitly offline. The inference server is intended for local research use and
binds to loopback only.

## Run an LSTM assessment

1. Open port 8766 and confirm the displayed model version and fixed threshold.
2. Import source telemetry with the same V1–V6 semantics as the training dataset.
   Arbitrary measurements are not scientifically meaningful inputs to this model.
3. Select **Run LSTM assessment**. Data is sent to Python on your own computer;
   neither telemetry nor results are persisted by the service.
4. Inspect anomaly scores, the saved threshold, and scored/unscored counts.
5. Review flagged episodes and per-channel errors at each episode's peak.

The full import is scored before display-period filtering. With the default
seven-reading model, the first six readings per eligible segment are unscored.
Steps outside 150–450 seconds reset sequence history. Short inputs may produce
zero scores. Missing scores never mean normal equipment.

Consecutive positive decisions form display episodes. A normal decision, gap,
or segment change ends an episode; the frozen `score > threshold` rule is not
modified. Detail panels show up to 20 episodes and tables up to 50: narrow the
review period for large results. Episodes overlapping the period retain their
full start time and peak evidence. Reconstruction error does not identify causes
or confirmed faults. A successful reassessment resets episode review state.

## Data and controls

- The overview starts with three example generating units, each with its own
  288-observation six-channel series. Two illustrative events flag Unit 01 / V1
  and Unit 02 / V4. Unit 03 has no events in the demo. These equipment identities
  and assignments are fictional, not inferred from the research dataset.
- Select an equipment card to inspect its signals. Cards show anomaly status
  separately from review status. "Mark reviewed" never resolves an anomaly or
  changes the equipment status to healthy. "View unit" selects the affected
  equipment and signal; orange dashed chart lines mark demo event times.
- Select V1–V6 and a time window. Windows are relative to the dataset's latest
  timestamp, not the computer clock. Charts break at observed gaps over 10 minutes.
- Import a CSV with `t,V1,V2,V3,V4,V5,V6`. Columns may be reordered; extra columns
  are ignored. ISO timestamps without a timezone are interpreted as UTC, matching
  the current Python contract. Measurements must be finite. Duplicate timestamps,
  malformed records, and invalid dates reject the entire import without replacing
  the current dataset. Limit: 25 MB and 100,000 records.
- Imports clear previous results and show "Unassigned equipment / Not assessed"
  until inference runs. Equipment stays unassigned after inference because the
  CSV has no verified equipment identity. Refreshing restores the demo.
- Export downloads all six channels of the selected equipment for the chosen
  time window as CSV; the filename includes its equipment ID.
- Mark reviewed marks an illustrative event as reviewed for the current session.
  Filter by status on the alert review page; Reset demo restores initial state.
- Channel physical units remain unverified. No physical labels or health claims
  are inferred from their numerical values.

## Verification

Node.js is only required for the data validation tests, not for serving the app:

```powershell
node --test frontend/tests/*.test.js
node --check frontend/app.js
```

The UI uses native HTML controls, accessible chart titles, focus indicators, and
responsive layouts. Optional Google Fonts use local sans-serif fallbacks when
offline. No JavaScript dependencies or build step are required.

## Integration boundary

`GET /api/model` describes the loaded bundle. `POST /api/score` accepts a JSON
`records` array and returns scored endpoints, decisions, per-channel errors and
coverage counts. It reuses frozen weights, scaler, sequence settings and threshold.
Requests are limited to 25 MB and 100,000 observations. Invalid inputs, duplicates,
unordered timestamps and cross-origin inference requests are rejected.

The completed LSTM detected 1 of 3 adequately covered development faults; this
does not establish operational reliability. See `docs/lstm-baseline-diagnostics.md`.
Live telemetry, chronological replay, durable records, authentication, equipment
mapping and drift monitoring remain future work.
