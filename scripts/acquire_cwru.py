"""Acquire the official 48 kHz DE subset, preserving source provenance (#9)."""

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = "https://engineering.case.edu/bearingdatacenter/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")

    def handle_data(self, data):
        if self.href and ".mat" in self.href:
            self.links.append((self.href, data.strip()))

    def handle_endtag(self, tag):
        if tag == "a":
            self.href = None


def run():
    root = Path("data/raw/cwru")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for page in ["normal-baseline-data", "48k-drive-end-bearing-fault-data"]:
        response = requests.get(BASE + page, timeout=45)
        response.raise_for_status()
        (root / f"{page}.html").write_text(response.text, encoding="utf-8")
        parser = Links()
        parser.feed(response.text)
        for href, name in parser.links:
            match = re.fullmatch(
                r"(Normal|IR007|IR014|IR021|B007|B014|B021|OR007@6|OR014@6|OR021@6)_(\d)", name
            )
            if not match:
                continue
            condition, load = match.groups()
            label = (
                "Healthy"
                if condition == "Normal"
                else "Inner race fault"
                if condition.startswith("IR")
                else "Outer race fault"
                if condition.startswith("OR")
                else "Rolling element fault"
            )
            url = urljoin(BASE, href)
            entries.append(
                dict(
                    recording=Path(href).name,
                    source_name=name,
                    label=label,
                    load_hp=int(load),
                    diameter_mils=0
                    if condition == "Normal"
                    else int(re.search(r"\d{3}", condition)[0]),
                    sample_rate=48000,
                    source_url=url,
                    source_page=BASE + page,
                    access_terms=(
                        "Public research download; "
                        "no explicit dataset redistribution license verified"
                    ),
                )
            )
    assert len(entries) == 40, f"Expected 40 source links; found {len(entries)}"

    def download(entry):
        path = root / entry["recording"]
        if not path.exists():
            for attempt in range(4):
                try:
                    r = requests.get(entry["source_url"], timeout=90)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
            if not r.content.startswith(b"MATLAB"):
                raise ValueError(f"Not a MATLAB file: {entry['source_url']}")
            path.write_bytes(r.content)
        entry.update(
            sha256=sha256(path.read_bytes()).hexdigest(),
            retrieved_at=datetime.now(UTC).isoformat(),
            local_filename=str(path),
        )
        print(entry["source_name"], flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(download, entries))
    with Path("data/cwru_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


if __name__ == "__main__":
    run()
