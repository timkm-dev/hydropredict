"""Fixed normal-only novelty candidates for the missed-fault development study."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from hydropredict.data.preprocessing import StandardTelemetryScaler
from hydropredict.features.causal import causal_feature_frame
from hydropredict.models.isolation_forest import SHPIsolationForest

CANDIDATES = ("pca_residual", "local_level", "regime_covariance", "local_temporal")
Matrix = np.ndarray[Any, np.dtype[np.float64]]


class NormalNovelty:
    """One predeclared mechanism, never tuned on fault outcomes."""

    def __init__(self, method: str) -> None:
        if method not in CANDIDATES:
            raise ValueError("Unknown fixed novelty candidate.")
        self.method = method
        self.scaler = StandardTelemetryScaler()
        self.estimator: Any = None
        self.covariances: list[Any] = []
        self.threshold: float | None = None
        self.fit_rows = 0
        self.calibration_rows = 0

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        SHPIsolationForest._validate(frame)
        mode = "temporal" if self.method == "local_temporal" else "raw"
        return causal_feature_frame(frame, mode, 30)

    def matrix(self, features: pd.DataFrame) -> Matrix:
        return self.scaler.transform(features)[list(self.scaler.means)].to_numpy(dtype=float)

    def values(self, matrix: Matrix) -> Matrix:
        if self.estimator is None:
            raise RuntimeError("Fit the normal-reference model first.")
        if not len(matrix):
            return np.array([], dtype=float)
        if self.method == "pca_residual":
            projections = self.estimator.transform(matrix)[:, -3:]
            variances = np.maximum(self.estimator.explained_variance_[-3:], 1e-8)
            scores = (projections**2 / variances).sum(axis=1)
        elif self.method == "regime_covariance":
            scores = np.stack([model.mahalanobis(matrix) for model in self.covariances]).min(axis=0)
        else:
            distances, _ = self.estimator.kneighbors(matrix)
            scores = distances.mean(axis=1)
        result = np.asarray(scores, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite novelty scores.")
        return result

    def fit(self, fitting: pd.DataFrame, calibration: pd.DataFrame) -> NormalNovelty:
        if (
            pd.to_datetime(fitting.t, utc=True).max()
            >= pd.to_datetime(calibration.t, utc=True).min()
        ):
            raise ValueError("Fitting must strictly precede calibration.")
        fit_features, cal_features = self.features(fitting), self.features(calibration)
        if len(fit_features) < 256 or len(cal_features) < 100:
            raise ValueError("Require 256 fitting and 100 calibration feature rows.")
        self.scaler.fit(fit_features, columns=tuple(str(c) for c in fit_features if c != "t"))
        matrix = self.matrix(fit_features)
        self.fit_rows, self.calibration_rows = len(matrix), len(cal_features)
        if self.method == "pca_residual":
            self.estimator = PCA(n_components=6, svd_solver="full").fit(matrix)
        elif self.method == "regime_covariance":
            self.estimator = KMeans(n_clusters=8, random_state=42, n_init=10).fit(matrix)
            self.covariances = []
            for label in range(8):
                members = matrix[self.estimator.labels_ == label]
                if len(members) < 10:
                    raise ValueError("Insufficient fitting support in a declared regime.")
                self.covariances.append(LedoitWolf().fit(members))
        else:
            self.estimator = NearestNeighbors(n_neighbors=20, n_jobs=1).fit(matrix)
        self.threshold = float(np.quantile(self.values(self.matrix(cal_features)), 0.99))
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.threshold is None:
            raise RuntimeError("A calibrated model is required.")
        features = self.features(frame)
        result = features[["t"]].copy()
        result["anomaly_score"] = self.values(self.matrix(features))
        result["threshold"] = self.threshold
        result["is_alert"] = result.anomaly_score > self.threshold
        result["model_version"] = self.method
        return result.reset_index(drop=True)
