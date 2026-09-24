"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded without requiring a third-party package."""

    data_dir: Path
    log_level: str

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("HYDROPREDICT_DATA_DIR", "data")),
            log_level=os.getenv("HYDROPREDICT_LOG_LEVEL", "INFO").upper(),
        )
