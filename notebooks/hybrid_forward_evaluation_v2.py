"""Run after investigation notebook cell 19 with %run -i hybrid_forward_evaluation_v2.py.

Requires timeline, frozen_scores, nominal, faults, CHANNELS and model from that
notebook. Does not use the previously fitted hybrid classifier or its thresholds.
"""

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def trend_features(frame, channels):
    """Six elapsed hours, no interpolation, reset after a gap over 450 seconds."""
    frame = frame.sort_values("t").reset_index(drop=True).copy()
    frame["t"] = pd.to_datetime(frame["t"], utc=True)
    if frame.t.isna().any() or frame.t.duplicated().any():
        raise ValueError("Invalid or duplicate timeline timestamps")
    if not np.isfinite(frame[channels].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite telemetry")
    segments = frame.t.diff().dt.total_seconds().gt(450).cumsum()
    output = []
    for _, group in frame.groupby(segments):
        times = pd.DatetimeIndex(group.t)
        hours = (times - times[0]).total_seconds().to_numpy() / 3600
        values = group[channels].to_numpy(dtype=float)
        for end in range(len(group)):
            if hours[end] < 6:
                continue
            begin = times.searchsorted(times[end] - pd.Timedelta(hours=6))
            x, y = hours[begin : end + 1], values[begin : end + 1]
            # Coverage in distinct five-minute slots, not raw row counts.
            slots = np.floor((hours[end] - x) * 12 + 1e-8).astype(int)
            if len(np.unique(slots)) < 58:
                continue
            centered = x - x.mean()
            slope = centered @ (y - y.mean(axis=0)) / (centered @ centered)
            row = {"t": times[end]}
            for j, channel in enumerate(channels):
                row[f"{channel}_slope_per_hour"] = slope[j]
                row[f"{channel}_std_6h"] = y[:, j].std(ddof=1)
            output.append(row)
    columns = ["t"] + [
        name for channel in channels for name in (f"{channel}_slope_per_hour", f"{channel}_std_6h")
    ]
    return pd.DataFrame(output, columns=columns)


def persistent(frame, threshold):
    """Three readings, restarting after unavailable observations or fold changes."""
    above = frame.score.ge(threshold)
    segments = frame.t.diff().dt.total_seconds().gt(450).cumsum()
    return above.groupby(segments).transform(lambda x: x.rolling(3, min_periods=3).sum().eq(3))


def coverage(times, fault):
    times = pd.DatetimeIndex(times)
    times = times[(times >= fault - pd.Timedelta(hours=6)) & (times < fault)]
    if not len(times):
        return 0.0, False
    slots = np.floor((fault - times).total_seconds() / 300).astype(int)
    fraction = min(len(np.unique(slots)) / 72, 1.0)
    return fraction, fraction >= 0.8 and fault - times.max() <= pd.Timedelta(minutes=10)


def evaluate(namespace):
    import xgboost as xgb

    required = ["timeline", "frozen_scores", "nominal", "faults", "CHANNELS", "model"]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError(f"Run the original investigation cells 0–19 first: missing {missing}")
    timeline = namespace["timeline"].copy()
    timeline["t"] = pd.to_datetime(timeline.t, utc=True)
    faults = sorted(set(pd.to_datetime(namespace["faults"], utc=True)))
    channels = list(namespace["CHANNELS"])
    scores = namespace["frozen_scores"][["t", "anomaly_score", "is_alert"]].copy()
    scores["t"] = pd.to_datetime(scores.t, utc=True)
    if scores.is_alert.isna().any() or not scores.is_alert.isin([True, False]).all():
        raise ValueError("Expected boolean frozen LSTM decisions")
    features = trend_features(timeline, channels)
    if features.empty:
        raise ValueError("No eligible six-hour trend history; no model was trained.")
    data = features.merge(scores, on="t", validate="one_to_one").sort_values("t")
    feature_cols = [c for c in features if c != "t"] + ["anomaly_score"]
    data["label"] = 0
    for fault in faults:
        data.loc[data.t.ge(fault - pd.Timedelta(hours=6)) & data.t.lt(fault), "label"] = 1
    nominal_times = pd.to_datetime(namespace["nominal"].t, utc=True)
    data["nominal"] = data.t.isin(nominal_times) & data.label.eq(0)

    # Overlapping precursor windows form a single evaluation group.
    groups = []
    for fault in faults:
        start = fault - pd.Timedelta(hours=6)
        if groups and start <= groups[-1]["end"]:
            groups[-1]["end"] = fault
            groups[-1]["faults"].append(fault)
        else:
            groups.append({"start": start, "end": fault, "faults": [fault]})

    output = Path(tempfile.mkdtemp(prefix="hydropredict-hybrid-forward-v2-"))
    protocol = {
        "purpose": "exploratory development; not independent final evaluation",
        "features": feature_cols,
        "calibration_rule": "latest up to 500 earlier nominal readings; minimum 100",
        "purge_hours": 6,
        "nominal_point_budget": 0.02,
        "persistence_readings": 3,
        "gap_reset_seconds": 450,
        "minimum_calibration_nominal_rows": 100,
        "minimum_training_rows": 100,
        "seed": 42,
        "lstm_version": str(namespace["model"].version),
        "test_evaluated": False,
        "model_promoted": False,
        "versions": {p: importlib.metadata.version(p) for p in ["xgboost", "numpy", "pandas"]},
        "label_caveat": "Outside known precursor windows is not verified healthy",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    folds, predictions, event_rows = [], [], []
    for fold, group in enumerate(groups):
        start = group["start"]
        stop = (
            groups[fold + 1]["start"]
            if fold + 1 < len(groups)
            else max(
                timeline.t.max() + pd.Timedelta(nanoseconds=1),
                group["end"] + pd.Timedelta(nanoseconds=1),
            )
        )
        cal_stop = start - pd.Timedelta(hours=6)
        # Reserve the latest 500 available earlier nominal observations, or all
        # available observations if fewer exist. The unchanged minimum is 100.
        # This rule depends only on history available before evaluation.
        earlier_nominal = data.loc[data.t.lt(cal_stop) & data.nominal, "t"]
        cal_start = (
            earlier_nominal.iloc[max(0, len(earlier_nominal) - 500)]
            if len(earlier_nominal)
            else cal_stop
        )
        train_stop = cal_start - pd.Timedelta(hours=6)
        train = data[data.t.lt(train_stop)].copy()
        # Do not train on only part of an event group crossing the training cutoff.
        for earlier in groups:
            if earlier["end"] >= train_stop:
                train = train[~train.t.between(earlier["start"], earlier["end"])]
        cal = data[data.t.ge(cal_start) & data.t.lt(cal_stop)].copy()
        test = data[data.t.ge(start) & data.t.lt(stop)].copy()
        record = {
            "fold": fold,
            "evaluation_start": start,
            "evaluation_end": stop,
            "training_end_exclusive": train_stop,
            "calibration_start": cal_start,
            "calibration_end_exclusive": cal_stop,
            "train_rows": len(train),
            "train_positive_rows": int(train.label.sum()),
            "calibration_rows": len(cal),
            "calibration_nominal_rows": int(cal.nominal.sum()),
            "evaluation_rows": len(test),
        }
        reason = ""
        if len(train) < 100 or train.label.nunique() < 2:
            reason = "insufficient earlier training rows or positive/negative classes"
        elif cal.nominal.sum() < 100:
            reason = "fewer than 100 earlier nominal calibration readings"
        elif test.empty:
            reason = "no matched evaluation scores"
        if not reason:
            classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=float(train.label.eq(0).sum() / train.label.eq(1).sum()),
                random_state=42,
                eval_metric="logloss",
                n_jobs=2,
            )
            classifier.fit(train[feature_cols], train.label)
            cal["score"] = classifier.predict_proba(cal[feature_cols])[:, 1]
            threshold = None
            # Strictly above 1 permits a valid no-alert calibration policy.
            for candidate in np.r_[np.linspace(0, 1, 1001), np.nextafter(1.0, 2.0)]:
                alerts = persistent(cal, candidate)
                if alerts[cal.nominal].mean() <= 0.02:
                    threshold = float(candidate)
                    break
            record["threshold"] = threshold
            record["calibration_alert_fraction"] = float(
                persistent(cal, threshold)[cal.nominal].mean()
            )
            test["score"] = classifier.predict_proba(test[feature_cols])[:, 1]
            test["hybrid_alert"] = persistent(test, threshold)
            test["fold"] = fold
            predictions.append(test)
            classifier.save_model(output / f"fold-{fold}-xgboost.json")
            cal.to_csv(output / f"fold-{fold}-calibration.csv", index=False)
        record["status"] = reason or "evaluated"
        folds.append(record)
        for fault in group["faults"]:
            raw_fraction, raw_ok = coverage(timeline.t, fault)
            fraction, eligible = coverage(test.t, fault)
            row = {
                "fold": fold,
                "fault": fault,
                "status": record["status"],
                "raw_coverage": raw_fraction,
                "raw_eligible": raw_ok,
                "matched_coverage": fraction,
                "eligible": eligible and not reason,
            }
            window = test[test.t.ge(fault - pd.Timedelta(hours=6)) & test.t.lt(fault)]
            for name, column in [("hybrid", "hybrid_alert"), ("lstm", "is_alert")]:
                alerts = (
                    window.loc[window[column].astype(bool), "t"]
                    if not reason
                    else pd.Series(dtype="datetime64[ns, UTC]")
                )
                row[f"{name}_warning_observed"] = bool(len(alerts)) if not reason else None
                row[f"{name}_lead_minutes"] = (
                    (fault - alerts.min()).total_seconds() / 60 if len(alerts) else None
                )
            event_rows.append(row)

    fold_table, events = pd.DataFrame(folds), pd.DataFrame(event_rows)
    fold_table.to_csv(output / "folds.csv", index=False)
    events.to_csv(output / "events.csv", index=False)
    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary = []
    if not combined.empty:
        assert not combined.t.duplicated().any()
        combined.to_csv(output / "predictions.csv", index=False)
        # Exposure is limited to adjacent observed intervals within each fold.
        for name, column in [("hybrid", "hybrid_alert"), ("lstm", "is_alert")]:
            nominal = combined[combined.nominal]
            episodes, hours = 0, 0.0
            for _, block in combined.groupby("fold"):
                dt = block.t.diff().dt.total_seconds()
                pair = block.nominal & block.nominal.shift(fill_value=False) & dt.le(450)
                hours += float(dt[pair].sum() / 3600)
                active = block[column].astype(bool) & block.nominal
                reset = ~pair
                episodes += int((active & (~active.shift(fill_value=False) | reset)).sum())
            eligible = events[events.eligible]
            summary.append(
                {
                    "model": name,
                    "nominal_points": len(nominal),
                    "nominal_alert_fraction": float(nominal[column].mean())
                    if len(nominal)
                    else None,
                    "nominal_episodes": episodes,
                    "observed_nominal_hours": hours,
                    "episodes_per_24_observed_hours": episodes * 24 / hours if hours else None,
                    "eligible_faults": len(eligible),
                    "eligible_faults_warned": int(
                        eligible[f"{name}_warning_observed"].fillna(False).sum()
                    ),
                    "eligible_groups": int(eligible.fold.nunique()),
                    "eligible_groups_warned": int(
                        eligible.groupby("fold")[f"{name}_warning_observed"].any().sum()
                    ),
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    from IPython.display import display

    print("Development evaluation only. Frozen LSTM uses its original point-alert policy.")
    print("Hybrid uses three-reading persistence. Nominal fractions are not operational FPR.")
    display(fold_table)
    display(events)
    display(summary)
    if not predictions or not events.eligible.any():
        print("INSUFFICIENT EVIDENCE: no adequately covered forward-evaluated faults.")
    print("Do not retune based on these evaluation results. Export:", output)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    archive = shutil.make_archive(str(output), "zip", output)
    try:
        from google.colab import files
    except ImportError:
        print("Results archive:", archive)
    else:
        files.download(archive)
    return output


if __name__ == "__main__":
    HYBRID_FORWARD_OUTPUT = evaluate(globals())
