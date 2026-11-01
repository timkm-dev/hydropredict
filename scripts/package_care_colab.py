"""Create the small input ZIP for notebooks/train_care_autoencoder_colab.ipynb.

The CARE data is not packaged: Colab downloads it from Zenodo and checks its MD5.
A locally trained Farm A reference score file is included for a device parity check.
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/api/records/15846963/files/CARE_To_Compare.zip/content"
ZENODO_MD5 = "2547b58c21ac8c242d13232860cf500c"
ZENODO_BYTES = 5503439673
PARITY = ("Wind Farm A", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; prior packages are preserved.")
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip():
        raise ValueError("Commit source changes before packaging a reproducible snapshot.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=zip", "HEAD", "pyproject.toml", "README.md", "src"]
        + ["scripts", "notebooks", "docs/care-autoencoder-protocol.md"],
        cwd=root,
    )
    farm, event_id = PARITY
    reference = (args.reference_scores / f"{farm.replace(' ', '_')}_{event_id}.npz").read_bytes()
    payloads = {"project.zip": archive, "parity_reference.npz": reference}
    manifest = dict(
        source_commit=commit,
        files_sha256={k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
        zenodo_url=ZENODO_URL,
        zenodo_md5=ZENODO_MD5,
        zenodo_bytes=ZENODO_BYTES,
        parity=dict(farm=farm, event_id=event_id),
        train_farms=["Wind Farm C"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in payloads.items():
            bundle.writestr(name, content)
        bundle.writestr("input_manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps(dict(output=str(args.output), source_commit=commit), indent=2))


if __name__ == "__main__":
    main()
