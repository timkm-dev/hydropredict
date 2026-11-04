"""Create an administrator account, or promote an existing one, for the Early Warning site.

Uses the same account database as the dashboard: DATABASE_URL from the environment or
`.env`, otherwise data/accounts.sqlite3. The password is never taken as an argument.
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

from hydropredict.auth import AuthStore, clean_email

ROOT = Path(__file__).resolve().parents[2]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hydropredict.account_admin", description=__doc__
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--generate-password", action="store_true", help="create a random password and print it"
    )
    source.add_argument(
        "--password-stdin", action="store_true", help="read the password from standard input"
    )
    parser.add_argument("--auth-db", type=Path, default=ROOT / "data/accounts.sqlite3")
    parser.add_argument(
        "--local-accounts",
        action="store_true",
        help="Use the local account database without PostgreSQL.",
    )
    args = parser.parse_args(arguments)
    load_dotenv(ROOT / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL")
    store = AuthStore(args.auth_db if args.local_accounts else database_url or args.auth_db)
    try:
        email = clean_email(args.email)
        with store.connect() as db:
            exists = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if exists:
            store.make_admin(email)
            print(f"{email} is now an administrator. Its password is unchanged.")
            return 0
        if args.generate_password:
            password = secrets.token_urlsafe(18)
        elif args.password_stdin:
            password = sys.stdin.readline().rstrip("\r\n")
        else:
            password = getpass.getpass("Password for the new administrator (12-128 characters): ")
            if getpass.getpass("Repeat the password: ") != password:
                raise ValueError("The passwords do not match.")
        store.make_admin(email, password, args.name)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"Created administrator {email}. Sign in and complete the profile to continue.")
    if args.generate_password:
        print(f"Password (shown once, store it safely): {password}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
