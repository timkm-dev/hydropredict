"""Build corrected SHP history and run the single preregistered comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.context import reconstruct_context
from hydropredict.data.sequences import build_sequences
from hydropredict.features.causal import causal_feature_frame

CHANNELS = [f"V{i}" for i in range(1, 7)]
BOUNDS = pd.to_datetime(["2018-08-01", "2019-01-01", "2019-02-01", "2019-05-01"], utc=True)
PARAMS = dict(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    max_depth=4,
    min_samples_leaf=20,
    l2_regularization=1,
    class_weight="balanced",
    early_stopping=False,
    random_state=42,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    frame = pd.read_csv(path)
    frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
    return frame


def groups(faults):
    result = []
    for fault in sorted(faults):
        if not result or fault - result[-1][-1] > pd.Timedelta(hours=6):
            result.append([])
        result[-1].append(fault)
    return result


def labels(times, faults):
    future = np.asarray(pd.DatetimeIndex(faults).as_unit("ns").asi8)
    values = pd.DatetimeIndex(times).as_unit("ns").asi8
    positions = np.searchsorted(future, values, side="right")
    y = np.zeros(len(values), dtype=int)
    valid = positions < len(future)
    y[valid] = (future[positions[valid]] - values[valid]) <= pd.Timedelta(hours=6).value
    return y


def split_rows(timeline, faults, start, end):
    selected = timeline[timeline.t.ge(start) & timeline.t.lt(end - pd.Timedelta(hours=6))]
    for group in groups(faults):
        if any(group[0] < boundary <= group[-1] for boundary in (start, end)):
            selected = selected[~selected.t.between(group[0] - pd.Timedelta(hours=6), group[-1])]
    # At-event readings are not precursor observations.
    return selected[~selected.t.isin(faults)].reset_index(drop=True)


def features(frame):
    # Split also on unexpectedly short intervals; the shared causal builder resets long gaps.
    segment = frame.t.diff().dt.total_seconds().lt(150).cumsum()
    parts = [
        causal_feature_frame(g.reset_index(drop=True), "temporal", 30)
        for _, g in frame.groupby(segment)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def persistent(times, above):
    dt = times.diff().dt.total_seconds()
    segment = (~dt.between(150, 450)).cumsum()
    return above.groupby(segment).transform(lambda s: s.rolling(3, min_periods=3).sum().eq(3))


def coverage(times, faults):
    return fault_window_coverage(times, faults)


def metrics(frame, faults, nominal):
    events = coverage(frame.t, faults)
    group_ids = {f: i for i, group in enumerate(groups(faults)) for f in group}
    for event, fault in zip(events, faults, strict=True):
        alerts = frame[
            frame.t.ge(fault - pd.Timedelta(hours=6)) & frame.t.lt(fault) & frame.is_alert
        ].t
        event.update(
            group=group_ids[fault],
            warning_observed=bool(len(alerts)),
            lead_minutes=(fault - alerts.min()).total_seconds() / 60 if len(alerts) else None,
        )
    selected = frame[frame.t.isin(nominal)]
    dt = selected.t.diff().dt.total_seconds()
    episodes = int(
        (
            selected.is_alert & (~selected.is_alert.shift(fill_value=False) | ~dt.between(150, 450))
        ).sum()
    )
    hours = float(dt[dt.between(150, 450)].sum() / 3600)
    return dict(
        score_rows=len(frame),
        events=events,
        eligible_groups=sorted({e["group"] for e in events if e["eligible"]}),
        warned_groups=sorted(
            {e["group"] for e in events if e["eligible"] and e["warning_observed"]}
        ),
        nominal_points=len(selected),
        nominal_alerts=int(selected.is_alert.sum()),
        nominal_alert_fraction=float(selected.is_alert.mean()) if len(selected) else None,
        nominal_episodes=episodes,
        observed_nominal_hours=hours,
        episodes_per_observed_day=episodes * 24 / hours if hours else None,
    )


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for row in pd.read_csv(root / "data/dataset_manifest.csv").iloc[:3].itertuples():
        path = root / row.local_filename
        hashes[row.local_filename] = sha(path)
        if hashes[row.local_filename].upper() != row.sha256.upper():
            raise ValueError(f"Source checksum mismatch: {path.name}")
    sources = {
        name: read(root / f"data/raw/{filename}")
        for name, filename in (("normal", "shp_train.csv"), ("prefault", "shp_test.csv"))
    }
    all_faults = read(root / "data/raw/shp_faults.csv").t.sort_values().tolist()
    faults = [f for f in all_faults if BOUNDS[0] <= f < BOUNDS[-1]]
    timeline, provenance, union = reconstruct_context(sources, BOUNDS[0], BOUNDS[-1])
    provenance.to_csv(output / "provenance.csv", index=False)
    pd.DataFrame({"t": faults}).to_csv(output / "faults.csv", index=False)
    report = dict(source_sha256=hashes, union=union, final_measurements_included=False, splits={})
    for name, start, end in zip(
        ("fit", "calibration", "development"), BOUNDS[:-1], BOUNDS[1:], strict=True
    ):
        raw = split_rows(timeline, faults, start, end)
        feat = features(raw)
        feat["target"] = labels(feat.t, faults)
        feat["selected_nominal"] = feat.t.isin(sources["normal"].t) & feat.target.eq(0)
        raw.to_csv(output / f"{name}-observations.csv", index=False)
        feat.to_csv(output / f"{name}-features.csv", index=False)
        period_faults = [f for f in faults if start <= f < end]
        seq = build_sequences(raw)
        report["splits"][name] = dict(
            start=start.isoformat(),
            end=end.isoformat(),
            before_purge_rows=int(timeline.t.between(start, end, inclusive="left").sum()),
            retained_rows=len(raw),
            feature_rows=len(feat),
            positive_rows=int(feat.target.sum()),
            selected_nominal_rows=int(feat.selected_nominal.sum()),
            fault_count=len(period_faults),
            raw_coverage=coverage(raw.t, period_faults),
            sequence_coverage=coverage(seq.endpoints.t, period_faults),
            feature_coverage=coverage(feat.t, period_faults),
        )
    dump(output / "report.json", report)
    # Archive an explicit code allowlist, never the working tree or later measurements.
    code = [
        "scripts/supervised_precursor.py",
        "docs/supervised-precursor-protocol.md",
        "pyproject.toml",
    ]
    code += [
        str(p.relative_to(root)).replace("\\", "/")
        for p in (root / "src/hydropredict").rglob("*.py")
    ]
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            archive.write(root / name, name)
    dump(output / "checksums.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return report


def run(package, diagnostics, output):
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in json.loads((package / "checksums.json").read_text()).items():
        if sha(package / name) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    frames = {
        name: read(package / f"{name}-features.csv")
        for name in ("fit", "calibration", "development")
    }
    predictors = [c for c in frames["fit"].columns if c not in ("t", "target", "selected_nominal")]
    if len(predictors) != 36:
        raise ValueError("Unexpected feature schema")
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(frames["fit"][predictors], frames["fit"].target)
    cal = frames["calibration"].copy()
    cal["score"] = model.predict_proba(cal[predictors])[:, 1]
    threshold = float(np.quantile(cal.loc[cal.selected_nominal, "score"], 0.98, method="higher"))
    frozen = dict(
        parameters=PARAMS,
        predictors=predictors,
        threshold=threshold,
        persistence=3,
        python=platform.python_version(),
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        package_manifest_sha256=sha(package / "checksums.json"),
        script_sha256=sha(__file__),
    )
    dump(output / "frozen.json", frozen)
    joblib.dump(model, output / "classifier.joblib")
    dev = frames["development"].copy()
    dev["score"] = model.predict_proba(dev[predictors])[:, 1]
    restored = joblib.load(output / "classifier.joblib")
    if not np.array_equal(dev.score.to_numpy(), restored.predict_proba(dev[predictors])[:, 1]):
        raise AssertionError("Serialized model replay mismatch")
    for frame in (cal, dev):
        frame["is_alert"] = persistent(frame.t, frame.score.gt(threshold))
    cal.to_csv(output / "calibration-scores.csv", index=False)
    dev.to_csv(output / "development-scores.csv", index=False)
    checks = json.loads((diagnostics / "checksums.json").read_text())
    if sha(diagnostics / "lstm-scores.csv").upper() != checks["lstm-scores.csv"].upper():
        raise ValueError("Frozen LSTM scores changed")
    baseline = read(diagnostics / "lstm-scores.csv")
    baseline = baseline[baseline.t.ge(BOUNDS[2]) & baseline.t.lt(BOUNDS[3])]
    faults = read(package / "faults.csv").t
    faults = faults[faults.ge(BOUNDS[2]) & faults.lt(BOUNDS[3])].tolist()
    # Nominal membership must include baseline-only endpoints, not just classifier coverage.
    provenance = read(package / "provenance.csv")
    normal = provenance.t[
        provenance.source_records.map(
            lambda s: any(r["source_id"] == "normal" for r in json.loads(s))
        )
    ]
    nominal = normal[labels(normal, faults) == 0]
    common = dev.t[dev.t.isin(baseline.t)]
    result = dict(
        classifier_native=metrics(dev, faults, nominal),
        lstm_native=metrics(baseline, faults, nominal),
        classifier_matched=metrics(dev[dev.t.isin(common)], faults, nominal),
        lstm_matched=metrics(baseline[baseline.t.isin(common)], faults, nominal),
        calibration_nominal_alert_fraction=float(cal.loc[cal.selected_nominal, "is_alert"].mean()),
        baseline_scores_sha256=sha(diagnostics / "lstm-scores.csv"),
        serialized_replay_exact=True,
        development_only=True,
    )
    ours = set(result["classifier_matched"]["warned_groups"])
    theirs = set(result["lstm_matched"]["warned_groups"])
    result["promising"] = bool(
        ours > theirs and result["classifier_native"]["nominal_alert_fraction"] <= 0.02
    )
    dump(output / "results.json", result)
    dump(output / "checksums.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.root, args.output)
        print(
            json.dumps(
                {
                    k: {c: v for c, v in s.items() if not c.endswith("coverage")}
                    for k, s in result["splits"].items()
                },
                indent=2,
            )
        )
    else:
        result = run(args.package, args.diagnostics, args.output)
        print(
            json.dumps(
                {
                    k: {c: v for c, v in s.items() if c != "events"} if isinstance(s, dict) else s
                    for k, s in result.items()
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
