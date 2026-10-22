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
Live telemetry, chronological replay, durable analysis records, role permissions, equipment
mapping and drift monitoring remain future work.

## Accounts and profiles

### Google sign-in

Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REDIRECT_URI` in
the root `.env`, then restart the dashboard. Register the redirect URI exactly
in the Google Cloud web OAuth client:
`http://127.0.0.1:8766/api/auth/google/callback`.
Use the same hostname when opening the app. The local server supports HTTP
loopback callbacks; public hosting requires a separate HTTPS deployment setup.

**Continue with Google** uses Google's authorization-code flow with PKCE,
browser-bound single-use state, a nonce, and server-side ID-token verification.
It requests only `openid email profile`. No Google access or refresh tokens are
retained. Google identities are stored by their stable subject identifier in
`external_identities`; new users complete the existing profile onboarding.

If a password account already uses the same email, sign in with your password,
open your profile, and select **Connect Google**. Accounts are never linked by
matching email alone. After linking, either login method works. Signing out ends
the HydroPredict session, without signing you out of Google.

For a consent-screen access error during development, review the audience and
test-user settings of your Google Cloud project. See
[Google's OpenID Connect guide](https://developers.google.com/identity/openid-connect/openid-connect).

Start the Python dashboard server as described above, then open
http://127.0.0.1:8766. Choose **Create account**, enter an email and a password
of 12–128 characters, then complete your name, company, username and job position.
Returning users sign in with email and password. Select your name in the sidebar
to edit your profile, or **Sign out** to end the session.

Accounts use PostgreSQL when `DATABASE_URL` is set in the environment or root
`.env` file. Otherwise, accounts persist in `data/accounts.sqlite3` (ignored by
Git); override the SQLite path with `--auth-db PATH`.
Passwords use salted scrypt hashes. Opaque session cookies are
HttpOnly and SameSite=Strict, expire after eight hours, and are invalidated on
sign-out. Five failed attempts per email temporarily limit sign-in for five minutes.
Inference requires an authenticated account with a completed profile. Company and
job position are profile information; they do not grant permissions.

These accounts support the loopback-only local server. Public deployment, email
verification, password recovery, and role permissions are not implemented. A static
preview cannot create accounts or sign in; use the Python dashboard server.

## PostgreSQL setup

### Docker PostgreSQL (recommended local setup)

Start Docker Desktop, then run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/setup_docker_postgres.py
```

This starts PostgreSQL 18 on `127.0.0.1:5433`, creates database `hydropredict`
with a restricted application login, imports SQLite accounts into an empty
database, and saves the dashboard connection in `.env`. Existing Docker accounts
are retained on subsequent runs. Restart the dashboard after setup. Any previous
`.env` database configuration is backed up before switching. An environment-level
`DATABASE_URL` takes precedence over `.env`; unset it to use the Docker connection.

Passwords are generated in ignored `.local/docker/` files. Keep these files and
`.env` private and retain them when restarting the container. Database files live
in Docker's named volume `hydropredict_postgres_data`, mounted at
`/var/lib/postgresql` for PostgreSQL 18. See the
[official Docker PostgreSQL guide](https://docs.docker.com/guides/postgresql/).

```powershell
docker compose ps                 # Check status
docker compose stop postgres      # Stop, retaining data
docker compose up -d --wait        # Start again
docker compose exec postgres psql -U postgres -d hydropredict
```

The native PostgreSQL service on port 5432 is unaffected. Normal container
restarts retain data. `docker compose down -v` deletes the database volume;
do not use it unless you intend to erase the database. A volume is not a backup.

### Native PostgreSQL

With PostgreSQL running locally, run from the repository root:

```powershell
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe scripts/setup_postgres.py
```

Setup prompts locally for the PostgreSQL administrator password, creates database
`hydropredict` and a dedicated application login, generates a random application
password, and saves `DATABASE_URL` in the ignored `.env` file. It copies existing
SQLite accounts with their password hashes and profiles. The source SQLite file
is retained. Sessions are not migrated; sign in again. Restart the dashboard.
Setup refuses to overwrite an existing role, database, or configured URL.

PostgreSQL stores accounts, profiles, session hashes, and failed sign-in attempts.
Telemetry uploads and analysis results are not yet persisted to the database.
Database files are managed by your PostgreSQL Windows service, outside the repo.
For an existing database, configure `DATABASE_URL` directly; application tables
are initialized at startup. See the
[Psycopg connection documentation](https://www.psycopg.org/psycopg3/docs/basic/usage.html).
