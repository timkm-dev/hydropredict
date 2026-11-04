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
        row = db.execute("SELECT role, disabled, created FROM users").fetchone()
        assert (row["role"], row["disabled"], row["created"]) == ("operator", 0, 0)


def test_administrator_account_actions(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "accounts.sqlite3")
    assert store.make_admin("Admin@Example.com", "admin passphrase 1", "Admin")
    admin = store.user(
        store.login({"email": "admin@example.com", "password": "admin passphrase 1"})
    )
    assert admin is not None and admin["role"] == "admin"
    person = {"email": "p@example.com", "password": "first passphrase"}
    token = store.register(person)
    target = store.user(token)["id"]

    listed = {a["email"]: a for a in store.accounts()}
    assert listed["p@example.com"]["sessions"] == 1
    assert listed["p@example.com"]["last_login"] > 0
    assert "password" not in listed["p@example.com"]

    store.administer(admin["id"], target, "role", {"role": "ml_engineer"})
    assert store.user(token)["role"] == "ml_engineer"
    with pytest.raises(ValueError):
        store.administer(admin["id"], target, "role", {"role": "owner"})

    # Turning an account off ends its sessions and refuses sign-in until it is turned on.
    store.administer(admin["id"], target, "disable", {})
    assert store.user(token) is None
    with pytest.raises(ValueError, match="turned off"):
        store.login(person)
    store.administer(admin["id"], target, "enable", {})
    token = store.login(person)

    store.administer(admin["id"], target, "password", {"password": "second passphrase"})
    assert store.user(token) is None
    with pytest.raises(ValueError, match="Incorrect"):
        store.login(person)
    token = store.login({**person, "password": "second passphrase"})
    store.administer(admin["id"], target, "sign_out", {})
    assert store.user(token) is None

    # An administrator cannot lock themselves out, and keeps the role when saving a profile.
    for action in ("role", "disable", "delete"):
        with pytest.raises(ValueError, match="your own"):
            store.administer(admin["id"], admin["id"], action, {"role": "operator"})
    store.profile(
        admin["id"],
        {"name": "A", "company": "C", "username": "admin", "job_position": "x", "role": "operator"},
    )
    assert store.accounts()[0]["role"] == "admin"
    with pytest.raises(ValueError):
        store.administer(admin["id"], True, "sign_out", {})

    store.administer(admin["id"], target, "delete", {})
    assert [a["email"] for a in store.accounts()] == ["admin@example.com"]
    with pytest.raises(ValueError, match="no longer exists"):
        store.administer(admin["id"], target, "sign_out", {})


def test_created_accounts_and_promotion(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "accounts.sqlite3")
    created = {"email": "new@example.com", "password": "temporary passphrase", "role": "operator"}
    store.create_account(created)
    with pytest.raises(ValueError, match="already uses"):
        store.create_account(created)
    with pytest.raises(ValueError):
        store.create_account({**created, "email": "x@example.com", "role": "owner"})
    with pytest.raises(ValueError):
        store.create_account({**created, "email": "y@example.com", "password": "short"})
    token = store.login(created)
    # Promoting an existing account keeps its password and turns it back on.
    assert not store.make_admin("new@example.com")
    assert store.user(token)["role"] == "admin"
