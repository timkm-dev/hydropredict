"""Account persistence, password storage, profile ownership and session boundaries."""

import hashlib
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
