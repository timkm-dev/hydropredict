"""Migration preserves accounts and rolls back conflicting imports."""

import sqlite3
from pathlib import Path

import pytest

from hydropredict.auth import AuthStore
from hydropredict.database import migrate_accounts


def test_migrate_accounts_preserves_password_and_profile(tmp_path: Path) -> None:
    source = AuthStore(tmp_path / "source.sqlite3")
    credentials = {"email": "migration@example.com", "password": "a migration passphrase"}
    token = source.register(credentials)
    user = source.user(token)
    assert user is not None
    source.profile(
        user["id"],
        {"name": "Example", "company": "Hydro", "username": "example", "job_position": "Engineer"},
    )
    target = AuthStore(tmp_path / "target.sqlite3")
    assert migrate_accounts(Path(source.path), target) == 1
    migrated = target.user(target.login(credentials))
    assert migrated is not None and migrated["username"] == "example"
    assert source.user(token) is not None
    assert target.user(token) is None
    assert migrate_accounts(tmp_path / "absent.sqlite3", target) == 0


def test_migration_conflict_rolls_back_entire_import(tmp_path: Path) -> None:
    source = AuthStore(tmp_path / "source.sqlite3")
    target = AuthStore(tmp_path / "target.sqlite3")
    for email in ("first@example.com", "duplicate@example.com"):
        source.register({"email": email, "password": "a migration passphrase"})
    target.register({"email": "duplicate@example.com", "password": "a migration passphrase"})
    with pytest.raises(sqlite3.IntegrityError):
        migrate_accounts(Path(source.path), target)
    with target.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
