"""Six-channel pointwise baseline with disjoint normal-reference calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.preprocessing import StandardTelemetryScaler
from hydropredict.data.schema import MEASUREMENT_COLUMNS


class SHPIsolationForest:
    """No labels, outcome metadata or engineered physical assumptions enter predictors."""

    def __init__(self, *, n_estimators: int = 200, random_state: int = 42) -> None:
        self.estimator = IsolationForest(
            n_estimators=n_estimators,
            max_samples=256,
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        )
        self.scaler = StandardTelemetryScaler()
        self.threshold: float | None = None
        self.version = "untrained"

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        if frame.empty or not set(MEASUREMENT_COLUMNS).issubset(frame.columns):
            raise ValueError("Nonempty complete V1..V6 input required.")
        values = frame[list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model measurements must be finite.")
        times = pd.to_datetime(frame["t"], utc=True)
        if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("Model timestamps must be valid, unique and chronological.")

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        self._validate(frame)
        return self.scaler.transform(frame)[list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float)

    def fit(
        self,
        training: pd.DataFrame,
        calibration: pd.DataFrame,
        *,
        version: str,
        calibration_quantile: float = 0.99,
    ) -> SHPIsolationForest:
        self._validate(training)
        self._validate(calibration)
        if len(training) < 256 or len(calibration) < 100:
            raise ValueError(
                "Baseline needs at least 256 fitting and 100 calibration observations."
            )
        if not 0 < calibration_quantile < 1:
            raise ValueError("Calibration quantile must be between zero and one.")
        if (
            pd.to_datetime(training.t, utc=True).max()
            >= pd.to_datetime(calibration.t, utc=True).min()
        ):
            raise ValueError("Fitting must strictly precede calibration.")
        self.scaler.fit(training)
        self.estimator.fit(self._matrix(training))
        scores = -self.estimator.score_samples(self._matrix(calibration))
        self.threshold = float(np.quantile(scores, calibration_quantile))
        self.version = version
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.threshold is None:
            raise RuntimeError("Model must be fitted or loaded before scoring.")
        scores = -self.estimator.score_samples(self._matrix(frame))
        return pd.DataFrame(
            {
                "t": pd.to_datetime(frame.t, utc=True).to_numpy(),
                "anomaly_score": scores,
                "is_alert": scores > self.threshold,
                "threshold": self.threshold,
                "model_version": self.version,
            }
        )

    def save(self, directory: Path | str, provenance: dict[str, Any]) -> None:
        if self.threshold is None:
            raise RuntimeError("Cannot save an unfitted model.")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        joblib.dump(self.estimator, target / "estimator.joblib")
        self.scaler.save(target / "scaler.json")
        metadata = {
            "format_version": 1,
            "model_version": self.version,
            "feature_columns": list(MEASUREMENT_COLUMNS),
            "threshold": self.threshold,
            "score_definition": "negative sklearn score_samples; higher is more anomalous",
            "alert_rule": "score > threshold; no persistence or cooldown",
            "sklearn_version": sklearn.__version__,
            "parameters": self.estimator.get_params(),
            "provenance": provenance,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        checksums = {
            name: compute_file_sha256(target / name)
            for name in ("estimator.joblib", "scaler.json", "metadata.json")
        }
        (target / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> SHPIsolationForest:
        """Load only trusted local bundles; joblib is executable Python serialization."""
        target = Path(directory)
        checksums = json.loads((target / "checksums.json").read_text(encoding="utf-8"))
        for name in ("estimator.joblib", "scaler.json", "metadata.json"):
            if compute_file_sha256(target / name) != checksums[name]:
                raise ValueError(f"Model bundle checksum mismatch: {name}")
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if metadata["feature_columns"] != list(MEASUREMENT_COLUMNS):
            raise ValueError("Incompatible model feature schema.")
        if metadata["sklearn_version"] != sklearn.__version__:
            raise ValueError("Load using the recorded scikit-learn version.")
        threshold = float(metadata["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Model threshold must be finite.")
        model = cls()
        model.estimator = joblib.load(target / "estimator.joblib")
        model.scaler = StandardTelemetryScaler.load(target / "scaler.json")
        if set(model.scaler.means) != set(MEASUREMENT_COLUMNS):
            raise ValueError("Incompatible scaler schema.")
        model.threshold = threshold
        model.version = str(metadata["model_version"])
        return model
