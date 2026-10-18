# HydroPredict dashboard

A responsive, dependency-free frontend for the research project. It provides an
overview, SVG telemetry charts, a six-channel explorer, illustrative alert review,
and a model readiness workspace. Python analysis modules remain unchanged.

From the repository root, start a local server:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory frontend
```

Open **http://127.0.0.1:8765** in a browser. Stop the server with Ctrl+C.
Serve `frontend/` rather than the repository root to keep source datasets and
model bundles outside the web server's document directory.

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
- Imported data stays in memory in the browser tab. Imports clear the demo alert
  view and show "Unassigned equipment / Not assessed" because the CSV schema has
  no verified equipment identity or model results. No anomaly scores or alerts
  are computed. Refreshing restores the demo.
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

This is a frontend milestone. It does not connect to live telemetry, execute the
Python models, persist operator actions, or authenticate users. Model cards report
the baseline status documented in the project README, not a live model registry.
Backend integration, replay, durable records, authentication, and drift monitoring
remain future work.
