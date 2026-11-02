# HydroPredict dashboard

## Wind turbine early warning (home page)

The signed-in home page (`care.html`) replays the CARE wind-farm benchmark through
the frozen autoencoder early warning model:

- **Fleet**: all 95 monitoring periods, sorted by the highest warning level reached,
  with the component to inspect first and any recorded fault.
- **Period replay**: press **Replay history** (or drag the time slider) to step through
  a period as if live. The chart shows only what was known at the selected time.
  **Look here first** lists the sensors furthest from expected normal behaviour, and
  **Actual vs expected** plots a sensor against the model's expectation.
- **Alerts**: send for inspection, acknowledge, dismiss (a reason is required) or
  reopen. Decisions are append-only and stored per account in
  `data/assessments.sqlite3`.

Status rules: **Alarm** when the warning counter reaches 72 (the CARE criticality
rule), **Watch** from 24 (a display rule only). An alarm stays marked after the
counter falls until it is reviewed. Sensor component groups are for display only.

The replay data is generated locally and ignored by Git (derived from CC BY-SA 4.0
data). Build it once after the CARE models exist (about 15 minutes on a laptop); the dataset
location comes from `CARE_DATA_DIR` in `.env`:

```powershell
.venv/Scripts/python.exe -m scripts.build_care_app_data --scores .local/care-autoencoder-v1-all --models .local/care-autoencoder-v1-scores .local/care-autoencoder-v1-C-colab/scores --results docs/results/care-autoencoder-v1 --output frontend/care-data
```

The earlier hydro plant, bearing and evaluation pages remain under **Research**.

**Bearing assessment** (`/bearing.html`, under Research).
The page opens with ten held-out recordings preloaded and the first selected.
Select **Assess selected recording**; no upload or sample-rate confirmation is
needed for these known files. Saved results remain in the history selector.
Alternatively, expand **Upload a different recording** to import a 48 kHz CWRU drive-end `.mat` file (up to
20 MB). Keep original numeric filenames for files with multiple DE arrays.
Predictions, feature contributions and benchmark limitations are displayed;
assessment history is stored per account. See [CWRU results](../docs/cwru-results.md).

**Hydro plant monitoring** (`/shp.html`): import an SHP CSV (`t,V1`–`V6`), run the
frozen LSTM, inspect scores and measurements, and save episode review responses.
See [the operator workflow](../docs/shp-operator-workflow.md). Unscored readings
(gaps, steps outside 150–450 seconds, sequence warm-up) never mean normal equipment.

**Hydro evaluation evidence** (`/evidence.html`) compares the nine development
fault timestamps with frozen LSTM scores. Generate its local data once with
`.venv/Scripts/python.exe scripts/map_anomalies_to_faults.py`; the output
`frontend/evidence-data.json` stays outside Git. See
[experiment and interpretation](../docs/anomaly-fault-mapping.md).

## Start the server

From the repository root (CPU PyTorch and the LSTM extra installed):

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --local-accounts --port 8766
```

Open **http://127.0.0.1:8766** and sign in. The server binds to loopback only and
serves frontend assets and inference endpoints, not datasets. The hydro page uses
the checksum-verified bundle at `models/lstm-diagnostics-2026-10-15-v1/frozen_bundle_copy`
(ignored by Git); pass `--bundle PATH` for another trusted bundle. Missing or
incompatible bundles stop startup; no model is trained or silently substituted.

## Verification

```powershell
node --test frontend/tests/*.test.js
```

No JavaScript dependencies or build step are required. Pages use native controls,
accessible chart labels, focus indicators and responsive layouts.

## Integration boundary

`GET /api/model` describes the loaded bundle. `POST /api/score` accepts a JSON
`records` array and returns scored endpoints, decisions, per-channel errors and
coverage counts. It reuses frozen weights, scaler, sequence settings and threshold.
Requests are limited to 25 MB and 100,000 observations. Invalid inputs, duplicates,
unordered timestamps and cross-origin inference requests are rejected.

The completed LSTM detected 1 of 3 adequately covered development faults; this
does not establish operational reliability. See `docs/lstm-baseline-diagnostics.md`.
Live telemetry, role permissions and drift monitoring remain future work.

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
