"""Start Docker PostgreSQL, provision the app database, and connect the dashboard."""

from __future__ import annotations

import secrets
import subprocess
from pathlib import Path
from urllib.parse import quote

import psycopg
from psycopg import sql

from hydropredict.auth import AuthStore
from hydropredict.database import migrate_accounts

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    secret_dir = ROOT / ".local/docker"
    secret_dir.mkdir(parents=True, exist_ok=True)
    admin_file = secret_dir / "postgres_password"
    app_file = secret_dir / "app_password"
    for path in (admin_file, app_file):
        if not path.exists():
            path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    admin_password = admin_file.read_text(encoding="utf-8").strip()
    app_password = app_file.read_text(encoding="utf-8").strip()
    subprocess.run(
        ["docker", "compose", "up", "-d", "--wait", "--wait-timeout", "180", "postgres"],
        cwd=ROOT,
        check=True,
    )
    with psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="postgres",
        user="postgres",
        password=admin_password,
        autocommit=True,
        connect_timeout=10,
    ) as admin:
        if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname='hydropredict'").fetchone():
            admin.execute(
                sql.SQL(
                    "CREATE ROLE hydropredict LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE"
                ).format(sql.Literal(app_password))
            )
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname='hydropredict'").fetchone():
            admin.execute("CREATE DATABASE hydropredict OWNER hydropredict")
    url = f"postgresql://hydropredict:{quote(app_password, safe='')}@127.0.0.1:5433/hydropredict"
    store = AuthStore(url)
    with store.connect() as db:
        populated = db.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"] > 0
    if not populated:
        count = migrate_accounts(ROOT / "data/accounts.sqlite3", store)
        print(f"Migrated {count} SQLite accounts; the original file is retained.")
    else:
        print("Existing Docker accounts retained; SQLite import skipped.")
    env_file = ROOT / ".env"
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    lines = existing.splitlines()
    previous = next((line for line in lines if line.startswith("DATABASE_URL=")), None)
    if previous and previous != "DATABASE_URL=" + url:
        backup = ROOT / (".env.before-docker-" + secrets.token_hex(4))
        backup.write_text(existing, encoding="utf-8")
        print("Previous database configuration backed up in an ignored .env.before-docker file.")
    lines = [line for line in lines if not line.startswith("DATABASE_URL=")]
    lines.append("DATABASE_URL=" + url)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Docker PostgreSQL ready: 127.0.0.1:5433 / hydropredict.")
    print("Dashboard connection saved to .env. Restart the dashboard and sign in again.")


if __name__ == "__main__":
    main()
