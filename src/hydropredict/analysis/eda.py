"""Exploratory Data Analysis (EDA) comparing normal operation vs fault precursors."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hydropredict.data.faults import FaultState
from hydropredict.data.schema import MEASUREMENT_COLUMNS


def compute_health_distribution_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Compare telemetry distributions between normal operation and precursor states."""
    if "fault_state" not in df.columns:
        raise ValueError("DataFrame must contain 'fault_state' column from fault alignment.")

    state_counts = df["fault_state"].value_counts().to_dict()
    total_records = len(df)

    normal_mask = df["fault_state"] == FaultState.NORMAL.value
    precursor_mask = df["fault_state"] == FaultState.PRECURSOR.value

    # Target numeric columns for comparison
    candidate_cols = [
        col
        for col in df.columns
        if col not in ("t", "fault_state") and pd.api.types.is_numeric_dtype(df[col])
    ]

    metrics: dict[str, dict[str, float]] = {}

    for col in candidate_cols:
        norm_series = df.loc[normal_mask, col].dropna()
        prec_series = df.loc[precursor_mask, col].dropna()

        if norm_series.empty or prec_series.empty:
            continue

        norm_mean = float(norm_series.mean())
        norm_std = float(norm_series.std())
        prec_mean = float(prec_series.mean())
        prec_std = float(prec_series.std())

        # Percentage shift in mean
        denom = norm_mean if abs(norm_mean) > 1e-6 else 1.0
        pct_shift = ((prec_mean - norm_mean) / abs(denom)) * 100.0

        metrics[col] = {
            "normal_mean": norm_mean,
            "normal_std": norm_std,
            "precursor_mean": prec_mean,
            "precursor_std": prec_std,
            "percentage_shift": pct_shift,
        }

    return {
        "total_records": total_records,
        "normal_records": state_counts.get(FaultState.NORMAL.value, 0),
        "precursor_records": state_counts.get(FaultState.PRECURSOR.value, 0),
        "fault_records": state_counts.get(FaultState.FAULT.value, 0),
        "metrics": metrics,
    }


def format_eda_summary(profile: dict[str, Any]) -> str:
    """Format EDA distribution metrics into a clean text summary table."""
    total = profile.get("total_records", 0)
    normal = profile.get("normal_records", 0)
    precursor = profile.get("precursor_records", 0)
    fault = profile.get("fault_records", 0)

    lines: list[str] = [
        "HydroPredict - Telemetry Exploratory Distribution Analysis",
        "=" * 60,
        f"Total Records     : {total}",
        f"Normal Records    : {normal} ({normal / max(total, 1) * 100:.1f}%)",
        f"Precursor Records : {precursor} ({precursor / max(total, 1) * 100:.1f}%)",
        f"Active Faults     : {fault} ({fault / max(total, 1) * 100:.1f}%)",
        "-" * 60,
        f"{'Variable':<18} | {'Normal Mean':<12} | {'Precursor Mean':<14} | {'Shift (%)':<10}",
        "-" * 60,
    ]

    metrics: dict[str, dict[str, float]] = profile.get("metrics", {})
    # Prioritize canonical measurements first, then others
    sorted_cols = [c for c in MEASUREMENT_COLUMNS if c in metrics] + [
        c for c in metrics if c not in MEASUREMENT_COLUMNS
    ]

    for col in sorted_cols:
        m = metrics[col]
        norm_str = f"{m['normal_mean']:<12.4f}"
        prec_str = f"{m['precursor_mean']:<14.4f}"
        shift_str = f"{m['percentage_shift']:>+8.2f}%"
        lines.append(f"{col:<18} | {norm_str} | {prec_str} | {shift_str}")

    lines.append("=" * 60)
    return "\n".join(lines)
