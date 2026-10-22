"""Account migration helpers for PostgreSQL setup."""

import sqlite3
from pathlib import Path

from hydropredict.auth import AuthStore


def migrate_accounts(source: Path, target: AuthStore) -> int:
    """Copy accounts atomically, retaining hashes and leaving the source intact."""
    if not source.exists():
        return 0
    with sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True) as old:
        rows = old.execute(
            "SELECT email, password, name, company, username, job_position FROM users"
        ).fetchall()
    with target.connect() as db:
        for row in rows:
            db.execute(
                "INSERT INTO users(email,password,name,company,username,job_position) "
                "VALUES (?,?,?,?,?,?)",
                row,
            )
    return len(rows)
