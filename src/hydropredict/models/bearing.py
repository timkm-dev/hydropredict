"""Fixed CWRU DE recording features and portable XGBoost classification."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, whosmat  # type: ignore[import-untyped]
from scipy.signal import hilbert, resample_poly  # type: ignore[import-untyped]
from xgboost import DMatrix, XGBClassifier

CLASSES = ["Healthy", "Inner race fault", "Outer race fault", "Rolling element fault"]
WINDOW = 4096
PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=3,
    reg_lambda=2,
    random_state=42,
    n_jobs=2,
    objective="multi:softprob",
    num_class=4,
)
FEATURES = [
    "RMS amplitude",
    "Peak amplitude",
    "Peak-to-peak amplitude",
    "Crest factor",
    "Kurtosis",
    "Skewness",
    "Shape factor",
    "Impulse factor",
    "Spectral centroid",
    "Spectral entropy",
]
FEATURES += [f"Spectrum {i * 750}–{(i + 1) * 750} Hz" for i in range(8)]
FEATURES += [f"Envelope {i * 200}–{(i + 1) * 200} Hz" for i in range(6)]


def read_signal(content: bytes, recording_name: str = "") -> NDArray[np.float64]:
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Choose a MATLAB recording smaller than 20 MB.")
    try:
        variables = whosmat(io.BytesIO(content))
        matches = [(name, shape) for name, shape, kind in variables if name.endswith("_DE_time")]
        if len(matches) > 1 and Path(recording_name).stem.isdigit():
            expected = f"X{int(Path(recording_name).stem):03d}_DE_time"
            matches = [(name, shape) for name, shape in matches if name == expected]
        if len(matches) != 1:
            raise ValueError(
                "Ambiguous or missing drive-end signal. Keep the original numeric filename "
                "(for example 175.mat) when a file contains multiple *_DE_time signals."
            )
        name, shape = matches[0]
        if len(shape) != 2 or min(shape) != 1 or not WINDOW * 4 <= np.prod(shape) <= 5_000_000:
            raise ValueError("Drive-end signal must be a vector with 16,384–5,000,000 samples.")
        raw = loadmat(io.BytesIO(content), variable_names=[name])[name]
        if np.iscomplexobj(raw):
            raise ValueError("Complex vibration measurements are unsupported.")
        values = np.asarray(raw, dtype=np.float64).ravel()
    except (OSError, TypeError, IndexError, NotImplementedError) as error:
        raise ValueError("Use a supported CWRU MATLAB recording (not MATLAB v7.3).") from error
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e8:
        raise ValueError("Vibration measurements must be finite and within numeric limits.")
    if np.std(values) < 1e-12:
        raise ValueError("The recording is constant; a vibration signal is required.")
    return values


def extract(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict[str, int]]:
    signal = resample_poly(values, 1, 4)
    count = len(signal) // WINDOW
    if count < 1:
        raise ValueError("Recording is too short for one complete analysis window.")
    x = signal[: count * WINDOW].reshape(count, WINDOW)
    x = x - x.mean(axis=1, keepdims=True)
    eps = 1e-15
    rms = np.sqrt((x * x).mean(axis=1)) + eps
    peak = np.abs(x).max(axis=1)
    meanabs = np.abs(x).mean(axis=1) + eps
    normalized = x / rms[:, None]
    power = np.abs(np.fft.rfft(x * np.hanning(WINDOW), axis=1)) ** 2
    power /= power.sum(axis=1, keepdims=True) + eps
    freq = np.fft.rfftfreq(WINDOW, 1 / 12000)
    envelope = np.abs(hilbert(x, axis=1))
    envelope -= envelope.mean(axis=1, keepdims=True)
    epower = np.abs(np.fft.rfft(envelope * np.hanning(WINDOW), axis=1)) ** 2
    epower /= epower.sum(axis=1, keepdims=True) + eps
    columns = [
        rms,
        peak,
        np.ptp(x, axis=1),
        peak / rms,
        (normalized**4).mean(axis=1),
        (normalized**3).mean(axis=1),
        rms / meanabs,
        peak / meanabs,
        (power * freq).sum(axis=1),
        -(power * np.log(power + eps)).sum(axis=1),
    ]
    columns += [power[:, (freq >= i * 750) & (freq < (i + 1) * 750)].sum(axis=1) for i in range(8)]
    columns += [epower[:, (freq >= i * 200) & (freq < (i + 1) * 200)].sum(axis=1) for i in range(6)]
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("Unable to derive finite vibration features.")
    return features, dict(
        original_samples=len(values),
        resampled_samples=len(signal),
        windows=count,
        unused_resampled_samples=len(signal) % WINDOW,
    )


class BearingService:
    def __init__(self, bundle: Path):
        self.model = XGBClassifier()
        self.model.load_model(bundle / "model.ubj")
        self.metadata = json.loads((bundle / "metadata.json").read_text())

    def assess(self, content: bytes, recording_name: str = "") -> dict[str, Any]:
        features, accounting = extract(read_signal(content, recording_name))
        probabilities = self.model.predict_proba(features)
        support = probabilities.mean(axis=0)
        winner = int(support.argmax())
        contributions = self.model.get_booster().predict(DMatrix(features), pred_contribs=True)
        effects = contributions[:, winner, :-1].mean(axis=0)
        top = np.argsort(np.abs(effects))[-3:][::-1]
        return dict(
            workflow="cwru",
            predicted_condition=CLASSES[winner],
            support={name: float(support[i]) for i, name in enumerate(CLASSES)},
            window_agreement=float((probabilities.argmax(axis=1) == winner).mean()),
            explanation=[dict(feature=FEATURES[i], contribution=float(effects[i])) for i in top],
            accounting=accounting,
            model_version=self.metadata["model_version"],
            evaluation=self.metadata["evaluation"],
            scores=[],
        )
