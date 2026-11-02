"""Package, verify or restore the files the hosted dashboard needs beyond Git.

The site serves generated CARE replay data, the SHP evidence page data, two
frozen models and the held-out CWRU recordings; Git ignores all of them. The
zip mirrors repository paths and carries a SHA-256 manifest. Keep it in
private storage: it holds CC BY-SA 4.0 derived data and CWRU recordings whose
redistribution licence is unverified.
"""

import argparse
import csv
from pathlib import Path

from scripts.acquire_care import extract
from scripts.backup_care_models import MANIFEST, create, verify

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "frontend/care-data",
    "frontend/evidence-data.json",
    "models/lstm-diagnostics-2026-10-15-v1/frozen_bundle_copy",
    "models/cwru-xgb-v1",
)


def runtime_sources(root):
    """Fixed sources plus the held-out (3 hp) recordings the bearing page offers."""
    with (root / "data/cwru_manifest.csv").open(newline="") as stream:
        held_out = [row["recording"] for row in csv.DictReader(stream) if row["load_hp"] == "3"]
    sources = [*SOURCES, *(f"data/raw/cwru/{name}" for name in held_out)]
    missing = [s for s in sources if not (root / s).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} runtime files are missing, e.g. {missing[0]}")
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", type=Path, metavar="ZIP")
    group.add_argument("--restore", type=Path, metavar="ZIP")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/runtime/runtime.zip")
    args = parser.parse_args(argv)
    archive = args.verify or args.restore
    if archive:
        bad = verify(archive)
        if bad:
            raise SystemExit(f"{len(bad)} files fail verification, e.g. {bad[0]}")
        print(f"Verified: {archive}")
        if args.restore:
            extract(archive, ROOT)
            (ROOT / MANIFEST).unlink(missing_ok=True)
            print(f"Restored runtime files under {ROOT}")
        return
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose another --output.")
    manifest = create(ROOT, runtime_sources(ROOT), args.output)
    bad = verify(args.output)
    if bad:
        raise SystemExit(f"Runtime package failed verification: {bad[0]}")
    size = args.output.stat().st_size / 1e6
    print(f"Packaged {len(manifest['files'])} files ({size:.0f} MB): {args.output}")


if __name__ == "__main__":
    main()
