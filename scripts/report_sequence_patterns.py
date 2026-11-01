"""Summarize the fixed experiment and test alert-timing chance explanations."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.supervised_precursor import dump, read, sha


def timing_control(frame, result, repeats=2000):
    """Shift alarms inside observed cadence segments, retaining their local count."""
    rng = np.random.default_rng(20261024)
    t = pd.DatetimeIndex(frame.t).as_unit("ns").asi8
    a = frame.is_alert.to_numpy(dtype=bool)
    cuts = np.r_[0, np.flatnonzero((np.diff(t) < 150e9) | (np.diff(t) > 450e9)) + 1, len(t)]
    masks = []
    for group in result["eligible_groups"]:
        mask = np.zeros(len(t), dtype=bool)
        for event in result["events"]:
            if event["eligible"] and event["group"] == group:
                fault = pd.Timestamp(event["fault_time"]).value
                mask |= (t >= fault - pd.Timedelta(hours=6).value) & (t < fault)
        masks.append(mask)
    counts = []
    for _ in range(repeats):
        shifted = np.empty_like(a)
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            shifted[start:end] = np.roll(a[start:end], rng.integers(end - start))
        counts.append(sum(bool(shifted[m].any()) for m in masks))
    observed = len(result["warned_groups"])
    rate = result["nominal_alert_fraction"]
    chance_probs = [1 - (1 - rate) ** int(m.sum()) for m in masks]
    return dict(
        observed=observed,
        eligible=len(masks),
        segment_shift_mean=float(np.mean(counts)),
        segment_shift_fraction_at_least_observed=float(np.mean(np.array(counts) >= observed)),
        iid_nominal_rate_expected=float(sum(chance_probs)),
        repeats=repeats,
        method="Independent circular shift per observed 150–450-second cadence segment; local alarm counts retained",
        caveat="Post-selection diagnostic, not a confirmatory p-value; short segments and local fault enrichment constrain shifts",
    )


def run():
    root = Path("models/sequence-patterns-v1-complete")
    public = Path("docs/results/sequence-patterns-v1")
    results = json.loads((root / "results.json").read_text())
    controls = {}
    for r in results:
        if r["candidate"] == "random_convolution-shape" and r["rule"] == "single":
            scores = read(root / f"{r['fold']}-{r['candidate']}-{r['rule']}-scores.csv")
            controls[r["fold"]] = timing_control(scores, r["native"])
    dump(public / "timing-controls.json", controls)
    print(json.dumps(controls, indent=2))
    table = []
    for name in sorted({r["candidate"] for r in results}):
        for rule in ["single", "three"]:
            rows = [
                next(
                    r
                    for r in results
                    if r["candidate"] == name and r["rule"] == rule and r["fold"] == fold
                )
                for fold in ["november", "december", "primary"]
            ]
            cells = [
                f"{len(r['native']['warned_groups'])}/{len(r['native']['eligible_groups'])}; {100 * r['native']['nominal_alert_fraction']:.2f}%"
                for r in rows
            ]
            table.append(f"| {name} | {rule} | " + " | ".join(cells) + " |")
    best = next(
        r
        for r in results
        if r["candidate"] == "random_convolution-shape"
        and r["rule"] == "single"
        and r["fold"] == "primary"
    )
    events = "\n".join(
        f"| {e['fault_time']} | {e['lead_minutes']:.1f} min |"
        for e in best["native"]["events"]
        if e["eligible"]
    )
    control_table = "\n".join(
        f"| {fold} | {c['observed']}/{c['eligible']} | {c['segment_shift_mean']:.2f} | {100 * c['segment_shift_fraction_at_least_observed']:.1f}% | {c['iid_nominal_rate_expected']:.2f} |"
        for fold, c in controls.items()
    )
    report = f"""# Sequence-pattern experiment results — issue #20

The strongest exploratory candidate is **random-convolution shape features with a single-reading alert**. It warned 10 of 11 eligible fault groups across the three chronological evaluation periods, including all three February–April groups. This is the most promising candidate from this experiment for a research proof of concept. It is not established as a reliable predictive model: its alert burden exceeds the previous 2% ceiling, warnings do not survive three-reading persistence, and the timing controls below show how much apparent success can arise from frequent alerts.

## Complete comparison

Cells show eligible fault groups warned / eligible groups; nominal point-alert fraction. Repeated timestamps in a six-hour fault group are not counted as separate independent successes.

| Model | Alert rule | November 2018 | December 2018 | February–April 2019 |
|---|---|---|---|---|
{chr(10).join(table)}

## Strongest candidate versus the saved LSTM

On **exactly the same 6,469 February–April endpoints**, random-convolution shape features warned **3/3** eligible groups versus **1/3** for the saved LSTM. Nominal alerts increased from **80/5,904 (1.36%)** to **149/5,904 (2.52%)**. Separate nominal alert episodes increased from **26 to 117** over 488.75 observed nominal hours: **1.28 to 5.75 per observed day**. These are observed hours, not full calendar days, and nominal labels are publisher labels, not verified operational false-alarm adjudications. The three eligible events are a subset of nine fault timestamps in that period; missing coverage excludes the others.

| Eligible fault timestamp (UTC as parsed) | First warning lead within six hours |
|---|---|
{events}

Three consecutive exceedances remove every eligible warning for this candidate in all three periods. The new signals are short. Counting a single alert in a six-hour window makes chance hits likely, so 3/3 must not be presented as 100% predictive accuracy.

## Chance-timing controls

After selecting the strongest candidate, 2,000 circular-shift controls per fold moved its alarms within each continuous observed segment. These preserve the local alarm count and most clustering, while changing timing relative to faults. Short segments and local alarm enrichment limit how much this control can move alarms. A second simple comparator assumes independent random alerts at the observed nominal point-alert rate.

| Period | Actual groups warned | Mean groups warned after segment shifts | Shifts equalling/exceeding actual | Expected groups warned by independent random alerts |
|---|---|---|---|---|
{control_table}

These are descriptive post-selection checks, not confirmatory significance tests. They must be considered alongside the raw detection count. Eleven groups and repeatedly inspected development periods cannot establish generalization.

## What was actually run

Four fixed models × three chronological folds = **12 fits**, each evaluated with two predeclared alarm rules = **24 evaluations**. No hyperparameter or threshold search against evaluation labels was performed. The best candidate was selected retrospectively from these evaluations.

- Input: seven observed readings (about 30 minutes) from six channels; reset at gaps outside 150–450 seconds; no filling or future context.
- Representations: training-standardized levels, or per-window centered/scaled channel shapes. Shape standard deviation floor 0.05 in training-standardized units.
- Dictionary: at most 24 positive prototypes per training fault group, 2,048 sampled negative prototypes; log nearest-negative / nearest-positive squared-distance ratio. Negative examples use the six-hour future-fault label, including earlier portions of publisher pre-fault windows.
- Random convolution: 256 fixed random multichannel kernels with lengths/dilations (3,1), (3,2), (5,1), (7,1); maximum and proportion-positive outputs; standardized features and ridge classifier alpha 10. Training positives weighted equally across fault groups and collectively balanced with negatives. This is an independent ROCKET-inspired implementation, not MiniRocket or a reproduction of its published scores.
- Train/calibrate/evaluate: August–September / October / November; August–October / November / December; August–December / January / February–April. Existing six-hour boundary purges and crossing-group exclusions retained.
- Threshold: strict exceedance of each fold's calibration-normal 98th percentile, frozen before evaluation. Alert rules: one exceedance or three consecutive exceedances. Evaluation fault labels do not enter fitting, scaling or calibration.
- All periods were previously used for project development. May onward was not accessed. Earlier folds reuse data in later training, so the 10/11 total is a chronological backtest summary, not eleven independent trials of one frozen model.
- Input checksums verified; saved-model replay exact on the same batch; prefix predictions agree within 1e-6. The initial incomplete attempt stopped on an overly strict cross-batch floating-point check (maximum difference about 5.3e-7); the complete run uses exact same-batch replay plus a separate numerical-tolerance prefix check. No model settings changed after that attempt.

## Decision and artifacts

Retain this shape-based convolution model as the **leading experimental candidate**, alongside the LSTM on the proof-of-concept evidence track. Do not replace production alerts or claim the earlier acceptance criteria are met. If a quieter candidate is required, the shape dictionary stayed below 2% nominal readings in all three periods but warned only 1/3, 4/5 and 1/3 groups respectively.

Runner: `scripts/experiment_sequence_patterns.py`. Re-run requires a fresh output directory (the script refuses to overwrite the completed run). Local models, scores, calibration scores and hashes: `models/sequence-patterns-v1-complete/`. Aggregate results and timing controls: `docs/results/sequence-patterns-v1/`. Restricted measurements and per-reading scores remain in ignored directories. No deployment, GitHub issue mutation, commit or push was performed.
"""
    Path("docs/sequence-pattern-results.md").write_text(report, encoding="utf-8")
    with zipfile.ZipFile(root / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in [
            "scripts/experiment_sequence_patterns.py",
            "scripts/report_sequence_patterns.py",
            "scripts/supervised_precursor.py",
            "scripts/compare_eif_context.py",
            "src/hydropredict/data/sequences.py",
            "src/hydropredict/analysis/readiness.py",
        ]:
            z.write(name, name)
    dump(
        root / "checksums.json",
        {p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != "checksums.json"},
    )


if __name__ == "__main__":
    run()
