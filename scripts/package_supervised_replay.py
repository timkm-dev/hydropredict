"""Package a verified supervised development run for local or Colab reproduction."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = json.loads((args.package / "checksums.json").read_text())
    for name, expected in checks.items():
        if hashlib.sha256((args.package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed package member: {name}")
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in [*checks, "checksums.json"]:
            archive.write(args.package / name, f"package/{name}")
        for name in ("lstm-scores.csv", "checksums.json"):
            archive.write(args.diagnostics / name, f"baseline/{name}")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"{args.output}: {digest}")


if __name__ == "__main__":
    main()
