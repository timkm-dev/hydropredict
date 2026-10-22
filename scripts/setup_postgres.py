"""Create a dedicated local PostgreSQL database and preserve existing accounts.

Run with .venv/Scripts/python.exe scripts/setup_postgres.py.
The PostgreSQL administrator password is requested locally and is never saved.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
from pathlib import Path
from urllib.parse import quote

import psycopg
from psycopg import sql

from hydropredict.auth import AuthStore
from hydropredict.database import migrate_accounts

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--admin", default="postgres")
    args = parser.parse_args()
    env_path = ROOT / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if any(line.startswith("DATABASE_URL=") for line in existing.splitlines()):
        parser.exit(1, "A DATABASE_URL is already configured in .env; no changes made.\n")
    password = secrets.token_urlsafe(32)
    admin_password = getpass.getpass("Local PostgreSQL administrator password: ")
    try:
        with psycopg.connect(
            host=args.host,
            port=args.port,
            user=args.admin,
            password=admin_password,
            dbname="postgres",
            autocommit=True,
            connect_timeout=5,
        ) as admin:
            if admin.execute("SELECT 1 FROM pg_roles WHERE rolname='hydropredict'").fetchone():
                parser.exit(1, "Role hydropredict already exists; no changes made.\n")
            if admin.execute("SELECT 1 FROM pg_database WHERE datname='hydropredict'").fetchone():
                parser.exit(1, "Database hydropredict already exists; no changes made.\n")
            admin.execute(
                sql.SQL(
                    "CREATE ROLE hydropredict LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE"
                ).format(sql.Literal(password))
            )
            admin.execute("CREATE DATABASE hydropredict OWNER hydropredict")
        url = (
            f"postgresql://hydropredict:{quote(password, safe='')}@"
            f"{args.host}:{args.port}/hydropredict"
        )
        # Save the generated application credential before migration, so a migration
        # conflict cannot leave the newly created database inaccessible to the app.
        env_path.write_text(existing.rstrip() + "\nDATABASE_URL=" + url + "\n", encoding="utf-8")
        store = AuthStore(url)
        count = migrate_accounts(ROOT / "data/accounts.sqlite3", store)
    except psycopg.Error:
        parser.exit(
            1,
            "PostgreSQL setup failed. Check credentials and server availability. "
            "If .env was written, retain it; it contains the new application credential.\n",
        )
    print(f"Created PostgreSQL database hydropredict at {args.host}:{args.port}.")
    print(f"Migrated {count} existing accounts. Original SQLite file retained.")
    print("Connection saved to .env. Restart the dashboard; users must sign in again.")


if __name__ == "__main__":
    main()
