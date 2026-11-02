"""Account persistence, password storage, profile ownership and session boundaries."""

import hashlib
import sqlite3
import time
from pathlib import Path

import pytest

from hydropredict.auth import AuthStore


def test_account_lifecycle(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "accounts.sqlite3")
    credentials = {"email": "Person@Example.com", "password": "a long passphrase"}
    token = store.register(credentials)
    user = store.user(token)
    assert user is not None
    assert user["email"] == "person@example.com"
    assert "password" not in user
    with store.connect() as db:
        stored = db.execute("SELECT password FROM users").fetchone()[0]
        assert credentials["password"] not in stored
        assert db.execute("SELECT token FROM sessions").fetchone()[0] != token
    store.profile(
        user["id"],
        {"name": "A Person", "company": "Hydro", "username": "Person", "job_position": "Engineer"},
    )
    reopened = AuthStore(store.path)
    assert reopened.user(token)["username"] == "person"
    with pytest.raises(ValueError):
        store.register(credentials)
    with pytest.raises(ValueError, match="Incorrect"):
        store.login({**credentials, "password": "wrong password"})
    signed_in = store.login(credentials)
    store.logout(token)
    assert store.user(token) is None
    assert store.user(signed_in) is not None
    with store.connect() as db:
        db.execute(
            "UPDATE sessions SET expires=? WHERE token=?",
            (time.time() - 1, hashlib.sha256(signed_in.encode()).hexdigest()),
        )
    assert store.user(signed_in) is None


def test_profile_validation_and_login_limit(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "accounts.sqlite3")
    with pytest.raises(ValueError):
        store.register({"email": "bad", "password": "short"})
    token = store.register({"email": "a@example.com", "password": "long passphrase here"})
    user = store.user(token)
    assert user is not None
    with pytest.raises(ValueError):
        store.profile(user["id"], {"name": ""})
    for _ in range(5):
        with pytest.raises(ValueError, match="Incorrect"):
            store.login({"email": "a@example.com", "password": "bad"})
    with pytest.raises(ValueError, match="Too many"):
        store.login({"email": "a@example.com", "password": "long passphrase here"})


def test_existing_accounts_gain_operator_role(tmp_path: Path) -> None:
    path = tmp_path / "accounts.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, "
            "password TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', company TEXT NOT NULL "
            "DEFAULT '', username TEXT UNIQUE, job_position TEXT NOT NULL DEFAULT '')"
        )
        db.execute("INSERT INTO users(email, password) VALUES ('old@example.com', '!')")
    db.close()
    store = AuthStore(path)
    AuthStore(path)  # Reopening must not try to add the column twice.
    with store.connect() as db:
        assert db.execute("SELECT role FROM users").fetchone()["role"] == "operator"
