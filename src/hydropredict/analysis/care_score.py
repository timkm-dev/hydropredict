"""CARE score for early fault detection on the CARE to Compare wind-turbine benchmark.

Reimplements the reference ``CAREScore`` of AEFDI/EnergyFaultDetector (Gück, Roelofs
and Faulstich, 2024). Where the reference code and paper differ, the code is followed
by default: earliness descends after 1/4 of the event, alarms need criticality >= 72,
and mean normal accuracy <= 0.5 replaces the combined score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Flags = np.ndarray[Any, np.dtype[np.bool_]]


@dataclass(frozen=True)
class EventScore:
    """Per-event evaluation in the reference implementation's terms."""

    event_id: int
    label: str
    f_beta: float
    accuracy: float
    weighted_score: float
    max_criticality: int
    first_detection: pd.Timestamp | None
    tp: int
    fp: int
    fn: int
    tn: int


def f_beta(tp: int, fp: int, fn: int, beta: float) -> float:
    """Standard F-beta; zero when undefined, as scikit-learn returns by default."""
    b2 = beta * beta
    denominator = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denominator) if denominator else 0.0


def criticality(predicted: Flags, normal: Flags, limit: int = 1000) -> np.ndarray[Any, Any]:
    """+1 for an anomaly during normal status, -1 for no anomaly, unchanged otherwise."""
    deltas = np.where(normal & predicted, 1, np.where(normal & ~predicted, -1, 0))
    result = np.empty(len(deltas), dtype=np.int64)
    value = 0
    for i, delta in enumerate(deltas):
        value = max(0, min(limit, value + int(delta)))
        result[i] = value
    return result


def earliness_weights(length: int, descend: tuple[int, int] = (1, 4)) -> np.ndarray[Any, Any]:
    """Event-window weights: 1 until ``descend`` of the event, then linear to 0.

    Mirrors the reference upsampling arithmetic exactly so scores are comparable.
    """
    numerator, denominator = descend
    start = numerator / denominator
    if denominator <= 0 or not 0 <= start < 1:
        raise ValueError("descend must satisfy 0 <= numerator/denominator < 1")
    scale = int(denominator)
    scaled = length * scale
    cut = int(scaled * start)
    x = np.linspace(0, scale, scaled)
    weights = np.zeros(scaled)
    weights[:cut] = scale
    weights[cut:] = scale / (1 - start) - x[cut:] / (1 - start)
    return np.asarray(weights[::scale] / scale)


def score_event(
    event_id: int,
    label: str,
    times: pd.DatetimeIndex,
    predicted: Flags,
    normal: Flags,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    *,
    threshold: int = 72,
    coverage_beta: float = 0.5,
    descend: tuple[int, int] = (1, 4),
) -> EventScore:
    """Evaluate one prediction period; ``normal`` marks status-normal timestamps."""
    if label not in ("anomaly", "normal"):
        raise ValueError("label must be 'anomaly' or 'normal'")
    if not times.is_monotonic_increasing or not len(times) == len(predicted) == len(normal):
        raise ValueError("times, predictions and status must align in time order")
    predicted = np.asarray(predicted, dtype=bool)
    normal = np.asarray(normal, dtype=bool)
    window = np.asarray((times >= event_start) & (times <= event_end))
    if not window.any():
        raise ValueError("No prediction data inside the event window.")
    truth = ~normal | (window if label == "anomaly" else False)
    y, p = truth[normal], predicted[normal]
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    tn = int((~y & ~p).sum())
    total = tp + fp + fn + tn
    upto = np.asarray(times <= event_end)
    crit = criticality(predicted[upto], normal[upto])
    hits = np.flatnonzero(crit >= threshold)
    weighted = float("nan")
    if label == "anomaly":
        weights = earliness_weights(int(window.sum()), descend)
        weighted = float(np.sum(predicted[window] * weights) / np.sum(weights))
    return EventScore(
        event_id=event_id,
        label=label,
        f_beta=f_beta(tp, fp, fn, coverage_beta) if label == "anomaly" else float("nan"),
        accuracy=(tp + tn) / total if total else float("nan"),
        weighted_score=weighted,
        max_criticality=int(crit.max()) if len(crit) else 0,
        first_detection=times[upto][hits[0]] if label == "anomaly" and len(hits) else None,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def care_score(
    events: list[EventScore],
    *,
    threshold: int = 72,
    reliability_beta: float = 0.5,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 2.0),
) -> dict[str, float]:
    """Combine events into coverage, earliness, reliability, accuracy and CARE.

    ``weights`` order is coverage, earliness, reliability, accuracy.
    """
    labels = {e.label for e in events}
    if labels != {"anomaly", "normal"}:
        raise ValueError("CARE needs at least one normal and one anomaly event.")
    anomalies = [e for e in events if e.label == "anomaly"]
    normals = [e for e in events if e.label == "normal"]
    detected = np.array([e.max_criticality >= threshold for e in events])
    actual = np.array([e.label == "anomaly" for e in events])
    parts = dict(
        coverage=float(np.mean([e.f_beta for e in anomalies])),
        earliness=float(np.mean([e.weighted_score for e in anomalies])),
        reliability=f_beta(
            int((actual & detected).sum()),
            int((~actual & detected).sum()),
            int((actual & ~detected).sum()),
            reliability_beta,
        ),
        accuracy=float(np.nanmean([e.accuracy for e in normals])),
        anomalies_detected=float((actual & detected).sum()),
        normals_alarmed=float((~actual & detected).sum()),
    )
    if not detected.any():
        care = 0.0
    elif parts["accuracy"] <= 0.5:
        care = parts["accuracy"]
    else:
        names = ("coverage", "earliness", "reliability", "accuracy")
        care = sum(w * parts[n] for w, n in zip(weights, names, strict=True)) / sum(weights)
    return {**parts, "care": float(care)}
