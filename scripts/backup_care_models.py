"""Back up, verify or restore the trained CARE autoencoders and their scores.

The models live only in ignored .local/ folders (and Google Drive for Farm C).
A backup is one zip with a manifest of SHA-256 checksums and the code commit.
Archive paths mirror the repository, so --restore puts files back in place.
Store the zip privately: the models are derived from CC BY-SA 4.0 data.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from scripts.acquire_care import extract

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (".local/care-autoencoder-v1-scores", ".local/care-autoencoder-v1-C-colab")
MANIFEST = "backup-manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create(root, sources, output):
    """Zip each source, a folder (recursively) or a single file, relative to root."""
    paths = [root / source for source in sources]
    files = sorted(p for s in paths for p in (s.rglob("*") if s.is_dir() else [s]) if p.is_file())
    if not files:
        raise SystemExit("No model files found to back up.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    manifest = dict(
        created=datetime.datetime.now(datetime.UTC).isoformat(),
        source_commit=commit or None,
        files={p.relative_to(root).as_posix(): sha256(p) for p in files},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for n, path in enumerate(files, start=1):
            bundle.write(path, path.relative_to(root).as_posix())
            print(f"\rBackup: {n}/{len(files)} files", end="", flush=True)
        bundle.writestr(MANIFEST, json.dumps(manifest, indent=1))
    print()
    return manifest


def verify(archive):
    """Return the names whose contents no longer match the manifest."""
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(MANIFEST))
        names = set(bundle.namelist()) - {MANIFEST}
        bad = sorted(names ^ set(manifest["files"]))
        for name in sorted(names & set(manifest["files"])):
            if hashlib.sha256(bundle.read(name)).hexdigest() != manifest["files"][name]:
                bad.append(name)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", type=Path, metavar="ZIP")
    group.add_argument("--restore", type=Path, metavar="ZIP")
    stamp = datetime.date.today().isoformat()
    parser.add_argument(
        "--output", type=Path, default=ROOT / f".local/backups/care-autoencoder-v1-{stamp}.zip"
    )
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
            print("Restored model files under .local/")
        return
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose another --output.")
    manifest = create(ROOT, SOURCES, args.output)
    bad = verify(args.output)
    if bad:
        raise SystemExit(f"Backup failed verification: {bad[0]}")
    size = args.output.stat().st_size / 1e6
    print(f"Backed up {len(manifest['files'])} files ({size:.0f} MB): {args.output}")


if __name__ == "__main__":
    main()
