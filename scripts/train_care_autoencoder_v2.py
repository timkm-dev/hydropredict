"""Cleaned-input autoencoder candidates; see docs/care-autoencoder-v2-protocol.md.

Each dataset is read once and every requested candidate is trained from it. Outputs
go to ``OUTPUT/<candidate>/`` in the v1 score-file format plus ``scores_per_sensor``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydropredict.data.care import care_root
from scripts.train_care_autoencoder import prepare, train_autoencoder
from scripts.train_care_baseline import digest

MIN_DISTINCT, CLIP, FLOOR, CAP = 10, 10.0, 1e-2, 100.0
AMBIENT = {"Wind Farm A": "sensor_0", "Wind Farm B": "sensor_8", "Wind Farm C": "sensor_7"}
CANDIDATES = {
    "clean": dict(epochs=40, patience=5, delta=False),
    "clean_long": dict(epochs=200, patience=10, delta=False),
    "clean_long_delta": dict(epochs=200, patience=10, delta=True),
}


def inputs(x, fit, descriptions, ambient, delta):
    """Model inputs: optional temperature-above-ambient, then quasi-constant channels dropped."""
    x = x.copy()
    if delta and ambient in x:
        temps = [
            c
            for c in x.columns
            if c != ambient and "temperature" in str(descriptions.get(c[:-4], "")).lower()
        ]
        x[temps] = x[temps].sub(x[ambient], axis=0)
    return x.loc[:, x.iloc[fit].nunique().ge(MIN_DISTINCT).to_numpy()]


def standardise(x, fit):
    mean = x.iloc[fit].mean().to_numpy()
    std = np.maximum(x.iloc[fit].std().to_numpy(), 1e-6)
    z = np.clip(np.nan_to_num(((x.to_numpy() - mean) / std).astype(np.float32)), -CLIP, CLIP)
    return z, mean, std


def scores(model, z, rows, n_stop, valid):
    """Plain and per-sensor scores for ``rows`` after the first ``n_stop`` stopping rows."""
    device = next(model.parameters()).device
    errors = []
    with torch.no_grad():
        for start in range(0, len(rows), 8192):
            chunk = torch.as_tensor(z[rows[start : start + 8192]], device=device)
            errors.append(((model(chunk) - chunk) ** 2).cpu().numpy().astype(float))
    err = np.concatenate(errors)
    stop, err = err[:n_stop], err[n_stop:]
    err[~valid] = np.nan
    typical = np.maximum(stop.mean(axis=0), FLOOR)
    plain = np.sqrt(err.mean(axis=1))
    per_sensor = np.sqrt(np.minimum(err / typical, CAP).mean(axis=1))
    return plain, per_sensor


def run_file(path, folder, out_dir, candidates, seed=42, device="cpu"):
    farm = path.parent.parent.name
    name = f"{farm.replace(' ', '_')}_{path.stem}"
    todo = [c for c in candidates if not (out_dir / c / f"{name}.npz").exists()]
    if not todo:
        return []
    data = prepare(path, folder)
    descriptions = pd.read_csv(folder / "feature_description.csv", sep=";").set_index(
        "sensor_name"
    )["description"]
    frame, times, fit, stop = data["frame"], data["times"], data["fit"], data["stop"]
    span = np.concatenate([data["tail"], data["pred"]])
    rows = []
    for candidate in todo:
        started = time.monotonic()
        settings = CANDIDATES[candidate]
        x = inputs(data["x"], fit, descriptions, f"{AMBIENT[farm]}_avg", settings["delta"])
        z, mean, std = standardise(x, fit)
        model, history = train_autoencoder(
            z[fit],
            z[stop],
            seed,
            epochs=settings["epochs"],
            patience=settings["patience"],
            device=device,
        )
        plain, per_sensor = scores(
            model, z, np.concatenate([stop, span]), len(stop), data["valid"][span]
        )
        target = out_dir / candidate
        np.savez_compressed(
            target / f"{name}.npz",
            times=times.iloc[span].to_numpy().astype("datetime64[ns]").astype(np.int64),
            scores=plain,
            scores_per_sensor=per_sensor,
            calibration=np.isin(span, data["calibration"]),
            prediction=np.isin(span, data["pred"]),
            status=frame.status_type_id.iloc[span].to_numpy(),
        )
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(
            dict(state=state, mean=mean, std=std, columns=list(x.columns)),
            target / f"{name}.pt",
        )
        rows.append(
            dict(
                candidate=candidate,
                farm=farm,
                event_id=int(path.stem),
                source_sha256=digest(path),
                features=int(x.shape[1]),
                dropped=int(data["x"].shape[1] - x.shape[1]),
                fit_rows=len(fit),
                epochs=len(history),
                best_stop_mse=float(min(history)),
                seconds=time.monotonic() - started,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", help="CARE folder (default: CARE_DATA_DIR in .env)"
    )
    parser.add_argument("--farms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--events", nargs="*", type=int, help="Optional subset of event ids")
    args = parser.parse_args()
    args.root = care_root(args.root)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    for candidate in args.candidates:
        (args.output / candidate).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    audit_path = args.output / "training.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    for farm in args.farms:
        folder = args.root / farm
        for path in sorted((folder / "datasets").glob("*.csv"), key=lambda p: int(p.stem)):
            if args.events and int(path.stem) not in args.events:
                continue
            started = time.monotonic()
            rows = run_file(path, folder, args.output, args.candidates, device=args.device)
            if not rows:
                continue
            audit += rows
            audit_path.write_text(json.dumps(audit, indent=2))
            done = "; ".join(f"{r['candidate']} {r['epochs']}" for r in rows)
            print(
                f"{farm}/{path.stem}: {done} epochs, {time.monotonic() - started:.0f}s", flush=True
            )
    config = dict(
        protocol_sha256=digest(Path("docs/care-autoencoder-v2-protocol.md")),
        script_sha256=digest(Path(__file__)),
        candidates={c: CANDIDATES[c] for c in args.candidates},
        min_distinct=MIN_DISTINCT,
        clip=CLIP,
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        seed=42,
        device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
