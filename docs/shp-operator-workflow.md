# Minimal SHP operator workflow

Implements a local delivery slice of #13 using the frozen LSTM. The authenticated
homepage now opens `frontend/shp.html`; the older synthetic dashboard assets are
retained but are no longer the default served homepage.

Import SHP CSV → run assessment → inspect scores and source measurements → review
episodes → save a response. Saved assessments can be reopened from the history
selector, including after a server restart. The initial view has no invented
equipment, health percentages, predicted fault types or remaining-life estimates.
The existing retrospective evaluation page remains available through a secondary link.

Every assessment saves its submitted readings, model metadata, scores, creation
time and owner. Review responses append an action, timestamp and note to that
assessment under the authenticated owner's ID. Reopening an episode preserves
earlier responses. Dismissal requires a reason; review actions never edit model
scores. Records are private to their owner; shared operator/engineer role-based
assignment is not implemented by this change.

Storage: `data/assessments.sqlite3`, ignored by Git. SQLite creates its two tables
on startup without changing account tables. For backup, stop the server and copy
the database along with the account database and model bundle to an authorized
private backup location. There is no retention/deletion UI or PostgreSQL migration
for assessments yet. History lists the latest 50 runs, with 20 episodes per page.

Run without Docker using local accounts:

```powershell
.venv/Scripts/python.exe -m hydropredict.dashboard --local-accounts --port 8766
```

Open http://127.0.0.1:8766 and sign in. Local SQLite accounts are separate from any
PostgreSQL accounts; existing PostgreSQL credentials are not migrated. Omit
`--local-accounts` to retain the existing configured account database behavior.

This is historical batch assessment, not live telemetry or incremental replay.
The loaded model and threshold are unchanged. The UI explicitly distinguishes
unscored readings, threshold exceedances and human review state. Predictive
reliability remains unproven. New experimental classifiers are not promoted.

Verification: account-scoped storage/restart tests, HTTP assessment/retrieval and
authentication tests, Python type/lint checks and existing frontend tests. Browser
automation was unavailable in this session; desktop/mobile visual verification
remains outstanding. The frontend was launched through the Windows browser handler.
No GitHub issue was closed or updated by this implementation.
