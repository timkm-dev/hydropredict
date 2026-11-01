"""Location of the CARE to Compare wind farm dataset, kept outside the repository."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_VAR = "CARE_DATA_DIR"
ROOT = Path(__file__).resolve().parents[3]
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")


def care_root(explicit: Path | None = None) -> Path:
    """Dataset folder from an explicit path, else CARE_DATA_DIR (environment or .env).

    The folder must contain the publisher's farm folders, e.g. ``Wind Farm A``.
    """
    load_dotenv(ROOT / ".env", override=False)
    value = explicit or (Path(os.environ[ENV_VAR]) if os.environ.get(ENV_VAR) else None)
    if value is None:
        raise ValueError(
            f"CARE data location unknown. Set {ENV_VAR} in .env or pass the folder; "
            "download it with scripts/acquire_care.py."
        )
    if not any((value / farm / "event_info.csv").is_file() for farm in FARMS):
        raise ValueError(
            f"{value} is not a CARE to Compare folder (no 'Wind Farm A/event_info.csv'). "
            "Download it with scripts/acquire_care.py."
        )
    return value
