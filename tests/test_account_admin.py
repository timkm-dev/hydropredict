"""Creating and promoting administrators from the command line."""

import io
from pathlib import Path

import pytest

from hydropredict.account_admin import main
from hydropredict.auth import AuthStore


def test_create_and_promote_administrators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "accounts.sqlite3"
    base = ["--local-accounts", "--auth-db", str(database)]
    monkeypatch.setattr("sys.stdin", io.StringIO("admin passphrase 1\n"))
    assert main([*base, "--email", "Admin@Example.com", "--password-stdin"]) == 0
    store = AuthStore(database)
    admin = store.user(
        store.login({"email": "admin@example.com", "password": "admin passphrase 1"})
    )
    assert admin is not None and admin["role"] == "admin"

    assert main([*base, "--email", "gen@example.com", "--generate-password"]) == 0
    password = capsys.readouterr().out.rsplit(": ", 1)[1].strip()
    assert store.user(store.login({"email": "gen@example.com", "password": password}))

    token = store.register({"email": "p@example.com", "password": "person passphrase"})
    assert main([*base, "--email", "p@example.com"]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert store.user(token)["role"] == "admin"

    # Startup creation never promotes or resets an account someone else registered.
    other = store.register({"email": "taken@example.com", "password": "someone elses pass"})
    monkeypatch.setattr("sys.stdin", io.StringIO("bootstrap passphrase\n"))
    assert main([*base, "--email", "taken@example.com", "--password-stdin", "--create-only"]) == 0
    assert "nothing changed" in capsys.readouterr().out
    assert store.user(other)["role"] == "operator"
    with pytest.raises(ValueError, match="Incorrect"):
        store.login({"email": "taken@example.com", "password": "bootstrap passphrase"})
    monkeypatch.setattr("sys.stdin", io.StringIO("bootstrap passphrase\n"))
    assert main([*base, "--email", "boot@example.com", "--password-stdin", "--create-only"]) == 0
    assert store.login({"email": "boot@example.com", "password": "bootstrap passphrase"})

    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    with pytest.raises(SystemExit):
        main([*base, "--email", "x@example.com", "--password-stdin"])
