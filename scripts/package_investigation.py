"""Package committed investigation code and explicitly allowlisted local evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.models.lstm import SHPLSTMAutoencoder

ROOT = Path(__file__).resolve().parents[1]


def package_investigation(inputs: Path, diagnostics: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("Preserve prior packages; select a new output path.")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit tracked source changes before packaging.")
    # Untracked caches are irrelevant: git archive includes only committed paths.
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=zip",
            "HEAD",
            "pyproject.toml",
            "README.md",
            "src",
            "scripts",
            "configs",
            "notebooks",
            "docs",
        ],
        cwd=ROOT,
    )
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        if "notebooks/investigate_shp_colab.ipynb" not in archive.namelist():
            raise ValueError("Commit the investigation notebook first.")
    report = json.loads((diagnostics / "diagnostics.json").read_text())
    if compute_file_sha256(inputs) != report["input_package_sha256"]:
        raise ValueError("Original input package changed.")
    expected = json.loads((diagnostics / "checksums.json").read_text())
    payloads = {"project.zip": source, "inputs/original-lstm-input.zip": inputs.read_bytes()}
    verified = {}
    for name in (
        "diagnostics.json",
        "lstm-scores.csv",
        "if_original_raw-scores.csv",
        "if_30min-scores.csv",
    ):
        if compute_file_sha256(diagnostics / name) != expected[name]:
            raise ValueError(f"Diagnostic evidence changed: {name}")
        payloads[f"evidence/{name}"] = (diagnostics / name).read_bytes()
        verified[name] = expected[name]
    payloads["evidence/checksums.json"] = json.dumps(verified, indent=2).encode()
    bundle = diagnostics / "frozen_bundle_copy"
    model = SHPLSTMAutoencoder.load(bundle)  # Verifies all four trusted bundle members.
    for name in ("metadata.json", "scaler.json", "weights.pt", "checksums.json"):
        payloads[f"evidence/frozen_bundle_copy/{name}"] = (bundle / name).read_bytes()
    with zipfile.ZipFile(inputs) as archive:
        original = json.loads(archive.read("input_manifest.json"))
        if original["test_data_included"]:
            raise ValueError("Final-test observations must not be included.")
        for name, sha in original["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest().upper() != sha:
                raise ValueError(f"Original package member changed: {name}")
    manifest = {
        "source_commit": commit,
        "model_version": model.version,
        "test_data_included": False,
        "training_or_threshold_changes": False,
        "files_sha256": {
            name: hashlib.sha256(value).hexdigest().upper() for name, value in payloads.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("investigation_manifest.json", json.dumps(manifest, indent=2))
    return {
        "output": str(output),
        "source_commit": commit,
        "sha256": compute_file_sha256(output),
        "bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "artifacts/colab/hydropredict-lstm-input-2026-10-14.zip",
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=ROOT / "models/lstm-diagnostics-2026-10-15-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_investigation(args.inputs, args.diagnostics, args.output), indent=2))


if __name__ == "__main__":
    main()
