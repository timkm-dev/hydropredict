"""Small owner-scoped SQLite store for local SHP assessments and review history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class AssessmentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                "name TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews (run_id TEXT NOT NULL, episode TEXT NOT NULL, "
                "owner TEXT NOT NULL, created TEXT NOT NULL, action TEXT NOT NULL, "
                "comment TEXT NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save(self, owner: str, name: str, records: list[Any], result: dict[str, Any]) -> str:
        run_id = uuid4().hex
        payload = json.dumps({"records": records, "result": result}, allow_nan=False)
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (run_id, owner, name[:160], datetime.now(UTC).isoformat(), payload),
            )
        return run_id

    def list(self, owner: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,name,created FROM runs WHERE owner=? ORDER BY created DESC LIMIT 50",
                    (owner,),
                )
            ]

    def get(self, owner: str, run_id: str) -> dict[str, Any]:
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT * FROM runs WHERE owner=? AND id=?", (owner, run_id)
            ).fetchone()
            if row is None:
                raise ValueError("Assessment not found.")
            result: dict[str, Any] = json.loads(row["payload"])
            result.update(id=row["id"], name=row["name"], created=row["created"])
            result["reviews"] = [
                dict(r)
                for r in db.execute(
                    "SELECT episode,created,action,comment FROM reviews "
                    "WHERE run_id=? AND owner=? ORDER BY rowid",
                    (run_id, owner),
                )
            ]
            return result

    def review(self, owner: str, payload: dict[str, Any]) -> None:
        run = self.get(owner, str(payload.get("run_id", "")))
        episode = payload.get("episode")
        action = payload.get("action")
        comment = payload.get("comment", "")
        starts = {s["t"] for s in run["result"]["scores"] if s["is_alert"]}
        if not isinstance(episode, str) or episode not in starts:
            raise ValueError("Select a flagged reading from this assessment.")
        if not isinstance(action, str) or action not in {"acknowledged", "dismissed", "open"}:
            raise ValueError("Invalid review action.")
        if not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError("Notes must contain at most 2,000 characters.")
        if action == "dismissed" and not comment.strip():
            raise ValueError("Add a reason before dismissing an alert.")
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO reviews VALUES (?,?,?,?,?,?)",
                (run["id"], episode, owner, datetime.now(UTC).isoformat(), action, comment.strip()),
            )
