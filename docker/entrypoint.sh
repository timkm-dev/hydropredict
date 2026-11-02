#!/bin/sh
# Keep accounts, reviews and imported recordings on the persistent share
# (App Service mounts /home); the image itself is replaced on every deploy.
set -e
state="${STATE_DIR:-/home/hydropredict}"
mkdir -p "$state/cwru-imports" /app/data/raw
for name in accounts.sqlite3 assessments.sqlite3 bearing_assessments.sqlite3; do
    ln -sf "$state/$name" "/app/data/$name"
done
ln -sfn "$state/cwru-imports" /app/data/raw/cwru-imports
exec python -m hydropredict.dashboard --host 0.0.0.0 --port "${PORT:-8000}" "$@"
