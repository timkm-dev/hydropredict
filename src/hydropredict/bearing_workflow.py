"""Owner-scoped local bearing assessment API operations."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from hydropredict.assessment_store import AssessmentStore
from hydropredict.models.bearing import BearingService


class BearingWorkflow:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.service = BearingService(root / "models/cwru-xgb-v1")
        self.store = AssessmentStore(root / "data/bearing_assessments.sqlite3")
        manifest = pd.read_csv(root / "data/cwru_manifest.csv")
        self.examples = {
            row.recording: {"label": row.label, "sha256": row.sha256}
            for row in manifest[manifest.load_hp.eq(3)].itertuples()
        }

    def info(self) -> dict[str, Any]:
        results = json.loads((self.root / "models/cwru-xgb-v1/results.json").read_text())
        return {
            "model_version": self.service.metadata["model_version"],
            "evaluation": results["primary"]["test"],
            "size_challenge": results["size_challenge"]["test"],
            "examples": [
                {"recording": name, "label": value["label"]}
                for name, value in self.examples.items()
            ],
        }

    def assess(self, owner: str, payload: dict[str, Any]) -> dict[str, Any]:
        reference = None
        if "example" in payload:
            name = payload["example"]
            if not isinstance(name, str) or name not in self.examples:
                raise ValueError("Choose an available benchmark recording.")
            content = (self.root / "data/raw/cwru" / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != self.examples[name]["sha256"]:
                raise ValueError("Benchmark recording checksum mismatch.")
            reference = self.examples[name]["label"]
        else:
            name = payload.get("name")
            if not isinstance(name, str) or not name.lower().endswith(".mat") or len(name) > 160:
                raise ValueError("Choose a MATLAB .mat recording.")
            if payload.get("sample_rate") != 48000:
                raise ValueError("This model requires 48,000 Hz drive-end recordings.")
            if not isinstance(payload.get("data"), str):
                raise ValueError("A recording is required.")
            try:
                content = base64.b64decode(payload["data"], validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError("Invalid recording encoding.") from error
        result = self.service.assess(content, name)
        digest = hashlib.sha256(content).hexdigest()
        result.update(source_sha256=digest, reference_condition=reference)
        # Keep exact source bytes for replay; paths are generated from hashes, not filenames.
        archive = self.root / "data/raw/cwru-imports"
        archive.mkdir(parents=True, exist_ok=True)
        path = archive / f"{digest}.mat"
        if not path.exists():
            path.write_bytes(content)
        run_id = self.store.save(owner, name, [], result)
        return self.store.get(owner, run_id)
