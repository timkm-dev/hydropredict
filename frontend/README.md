# Early Warning dashboard

The interface follows the redesign agreed in issue #24: plain language for the
maintenance operator first, technical detail on the Models page and in `docs/`.
Shared styles are in `theme.css`; the logo and welcome illustration are
`logo.svg` and `welcome-illustration.svg`.

## Pages

- **Welcome** (`/` when signed out, `welcome.html`): public introduction to the
  project, how it works, the two user types and the headline results.
- **What needs attention** (`care.html`, the signed-in home): turbine periods grouped
  as **Inspect now**, **Keep an eye on** and **Normal**, with the component to check
  first and the biggest changes in plain words.
- **Turbine page**: a headline sentence, **What we noticed** (now vs normal for the
  sensors furthest from normal), an actual-vs-normal chart and the decision buttons.
  **How was this decided?** holds the warning build-up, the score chart and the
  replay controls. For periods with a recorded fault, a demo-only box says what
  happened next and whether the first alarm came before or after it.
- **Alerts**: **Send a technician**, **Seen it, keep watching** or **Not a problem**
  (a reason is required), and **Undo**/**Reopen**. The button shows *Saving…*, then
  the card confirms the decision in place (the list does not jump; decided alerts
  leave **To decide** the next time it is opened). Decisions are append-only, stored
  per account in the `alert_decisions` table of the account database (PostgreSQL
  when `DATABASE_URL` is set, otherwise `data/accounts.sqlite3`). Decisions saved
  earlier in `data/assessments.sqlite3` are copied over once on startup.
- **How well it works**: the benchmark results and every recorded fault with the
  timing of its first alarm.
- **Models** (`models.html`, ML Engineers only): what each model does, its data,
  frozen settings and results.
- **Admin** (`admin.html`, administrators only): accounts and access, described
  under Roles below.

Status rules: **Inspect now** when the warning counter reaches 72 (the CARE
criticality rule, about 12 hours of unusual readings), **Keep an eye on** from 24 (a
display rule only). Sensor component groups are for display only; the autoencoder
detects unusual behaviour but does not classify faults.

## Roles (issue #14)

| Action | Maintenance Operator | ML Engineer | Administrator |
|---|---|---|---|
| Welcome, sign in, profile | yes | yes | yes |
| What needs attention, turbine, alerts, how well it works | yes | yes | yes |
| Record alert decisions | yes | yes | yes |
| Models page | no | yes | yes |
| Research pages, `evidence-data.json` and model-scoring APIs | no | yes | yes |
| **Admin** page (`admin.html`) and `/api/admin/*` | no | no | yes |

Each person chooses between Maintenance Operator and ML Engineer in their profile
(**Role in this app**); new accounts start as Maintenance Operators. This is
deliberately permissive for the prototype: anyone can choose ML Engineer. The server
still checks the stored role on every request. Only an administrator can make
someone an administrator, and an administrator's profile cannot change their own role.

**Admin page.** Administrators see an **Admin** link in the header. It lists every
account (search and role filters, who is signed in now, last sign-in), adds accounts
with a generated temporary password to hand over privately, and for each account can
change the role, reset the password, sign out everywhere, turn the account off or
on, or delete it (typing the email to confirm). Turned-off accounts cannot sign in
and keep their history; deleting also removes the account's saved assessments and
alert decisions. Administrators cannot change their own role, turn off or delete
their own account, so there is always at least one administrator. The page also
lists recent alert decisions across all accounts.

**Creating the first administrator.** Run this against the same account database as
the dashboard (`DATABASE_URL` from `.env`, otherwise `data/accounts.sqlite3`). It asks
for the password, or use `--generate-password` to print a random one once:

```powershell
.venv/Scripts/python.exe -m hydropredict.account_admin --email admin@example.com
```

If the email already has an account, it is promoted to administrator and keeps its
password.

**On the hosted site** the container has no SSH server, so add two App Service
settings instead: `ADMIN_EMAIL` and `ADMIN_PASSWORD` (12–128 characters). Saving them
restarts the site, and on startup `docker/entrypoint.sh` creates that administrator
once (`--create-only`). An email that already has an account is left unchanged, so
someone who registered it first is never promoted. After signing in, change the
password on the Admin page and delete both settings.

The replay data is generated locally and ignored by Git (derived from CC BY-SA 4.0
data). Build it once after the CARE models exist (about 15 minutes on a laptop); the dataset
location comes from `CARE_DATA_DIR` in `.env`:

```powershell
.venv/Scripts/python.exe -m scripts.build_care_app_data --scores .local/care-autoencoder-v1-all --models .local/care-autoencoder-v1-scores .local/care-autoencoder-v1-C-colab/scores --results docs/results/care-autoencoder-v1 --output frontend/care-data
```

The earlier hydro plant, bearing and evaluation pages remain under **Research**
(ML Engineers only).

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
Live telemetry and drift monitoring remain future work.

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
verification and password recovery are not implemented; roles are described above. A static
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
