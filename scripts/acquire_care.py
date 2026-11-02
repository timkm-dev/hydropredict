"""Download, verify and extract the CARE to Compare dataset from Zenodo.

Resumes partial downloads, checks the published MD5 and recorded SHA-256, and
extracts to <dest>/CARE_To_Compare. Every long step shows progress. Afterwards set
CARE_DATA_DIR=<dest>/CARE_To_Compare in .env (the script prints the line).
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from hydropredict.data.care import ENV_VAR, ROOT, care_root
from scripts.package_care_colab import ZENODO_BYTES, ZENODO_MD5, ZENODO_URL

ZENODO_SHA256 = "ca61379e98956d891041ad45c885109bd8a14199fde0688d0184a11c2d4194f1"
CHUNK = 8 * 1024 * 1024


class Progress:
    """One-line progress with percentage, size, speed and time remaining."""

    def __init__(self, label, total, done=0):
        self.label, self.total, self.done = label, total, done
        self.start, self.base, self.shown = time.monotonic(), done, 0.0

    def update(self, n):
        self.done += n
        now = time.monotonic()
        if now - self.shown >= 0.5 or self.done >= self.total:
            self.shown = now
            rate = (self.done - self.base) / max(now - self.start, 1e-6)
            left = (self.total - self.done) / rate if rate else 0
            sys.stdout.write(
                f"\r{self.label}: {self.done / self.total:6.1%}  "
                f"{self.done / 1e9:5.2f}/{self.total / 1e9:.2f} GB  "
                f"{rate / 1e6:6.1f} MB/s  {left / 60:4.0f} min left "
            )
            sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")


def download(url, path, total):
    """Download to path, resuming when the server honours a byte range."""
    done = path.stat().st_size if path.exists() else 0
    if done >= total:
        return
    headers = {"Range": f"bytes={done}-"} if done else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        resumed = getattr(response, "status", None) == 206
        progress = Progress("Download", total, done if resumed else 0)
        with path.open("ab" if resumed else "wb") as out:
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                progress.update(len(block))
        progress.close()


def digests(path, total):
    md5, sha = hashlib.md5(), hashlib.sha256()
    progress = Progress("Verify", total)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            md5.update(block)
            sha.update(block)
            progress.update(len(block))
    progress.close()
    return md5.hexdigest(), sha.hexdigest()


def extract(archive, dest):
    """Extract with a byte progress bar, refusing paths that escape dest."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        progress = Progress("Extract", sum(m.file_size for m in members) or 1)
        for member in members:
            bundle.extract(member, dest)
            progress.update(member.file_size)
        progress.close()


def extracted(target):
    try:
        care_root(target)
    except ValueError:
        return False
    return True


def main(argv=None):
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ[ENV_VAR]).parent if os.environ.get(ENV_VAR) else Path.home() / "data"
    parser.add_argument("--dest", type=Path, default=default, help=f"Parent folder ({default})")
    args = parser.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    archive = args.dest / "CARE_To_Compare.zip"
    target = args.dest / "CARE_To_Compare"
    if extracted(target):
        print(f"Already extracted: {target}")
    else:
        download(ZENODO_URL, archive, ZENODO_BYTES)
        if digests(archive, ZENODO_BYTES) != (ZENODO_MD5, ZENODO_SHA256):
            raise SystemExit(
                f"{archive} does not match the published checksums. Delete it and rerun."
            )
        extract(archive, args.dest)
        care_root(target)
    print(f"Ready. In .env set:\n{ENV_VAR}={target}")


if __name__ == "__main__":
    main()
