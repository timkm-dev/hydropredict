"""Per-dataset autoencoder normal-behaviour models; see docs/care-autoencoder-protocol.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.train_care_baseline import digest

NORMAL_STATUS = (0, 2)


def allowed_features(folder, header):
    desc = pd.read_csv(folder / "feature_description.csv", sep=";")
    flag = desc[["is_angle", "is_counter"]].astype(str).apply(lambda c: c.str.lower().eq("true"))
    allowed = set(desc.loc[~flag.any(axis=1), "sensor_name"])
    return [c for c in header if c.endswith("_avg") and c[:-4] in allowed]


def splits(frame, valid):
    """Chronological 70/10/20 split of training rows, keeping eligible normal rows."""
    train = np.flatnonzero(frame.train_test.eq("train"))
    pred = np.flatnonzero(frame.train_test.eq("prediction"))
    assert len(train) and len(pred) and train.max() < pred.min()
    eligible = frame.status_type_id.isin(NORMAL_STATUS).to_numpy() & valid
    a, b = int(len(train) * 0.7), int(len(train) * 0.8)
    parts = train[:a], train[a:b], train[b:]
    fit, stop, calibration = (p[eligible[p]] for p in parts)
    return fit, stop, calibration, train[b:], pred


class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        wide, narrow = max(8, d // 2), max(4, d // 8)
        self.net = nn.Sequential(
            nn.Linear(d, wide),
            nn.ELU(),
            nn.Linear(wide, narrow),
            nn.ELU(),
            nn.Linear(narrow, wide),
            nn.ELU(),
            nn.Linear(wide, d),
        )

    def forward(self, x):
        return self.net(x)


def train_autoencoder(x_fit, x_stop, seed=42, epochs=40, patience=5, batch=256, device="cpu"):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    # Initialise on CPU so weights match across devices for the same seed.
    model = AutoEncoder(x_fit.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit, stop = torch.as_tensor(x_fit, device=device), torch.as_tensor(x_stop, device=device)
    best, best_state, waited, history = float("inf"), None, 0, []
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(fit), generator=generator).split(batch):
            idx = idx.to(device)
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(fit[idx]), fit[idx])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.mse_loss(model(stop), stop))
        history.append(value)
        if value < best:
            best, waited = value, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def reconstruction_scores(model, x, valid):
    device = next(model.parameters()).device
    scores = np.full(len(x), np.nan)
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            chunk = torch.as_tensor(x[start : start + 8192], device=device)
            error = (model(chunk) - chunk).pow(2).mean(dim=1).sqrt().cpu().numpy()
            scores[start : start + len(error)] = error
    scores[~valid] = np.nan
    return scores


def run_file(path, folder, out_dir, seed=42, device="cpu"):
    header = pd.read_csv(path, sep=";", nrows=0).columns
    features = allowed_features(folder, header)
    keys = ["time_stamp", "train_test", "status_type_id"]
    frame = pd.read_csv(
        path, sep=";", usecols=keys + features, dtype={c: "float32" for c in features}
    )
    times = pd.to_datetime(frame.time_stamp)
    assert times.is_monotonic_increasing and times.is_unique
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    valid = x.isna().mean(axis=1).le(0.2).to_numpy()
    fit, stop, calibration, tail, pred = splits(frame, valid)
    kept = x.iloc[fit].nunique().gt(1).to_numpy()
    x = x.loc[:, kept]
    valid = x.isna().mean(axis=1).le(0.2).to_numpy()
    fit, stop, calibration, tail, pred = splits(frame, valid)
    if len(fit) < 1000 or len(stop) < 200 or len(calibration) < 500 or x.shape[1] == 0:
        raise ValueError(f"Insufficient normal history: {path}")
    mean = x.iloc[fit].mean().to_numpy()
    std = np.maximum(x.iloc[fit].std().to_numpy(), 1e-6)
    z = np.nan_to_num(((x.to_numpy() - mean) / std).astype(np.float32), nan=0.0)
    model, history = train_autoencoder(z[fit], z[stop], seed, device=device)
    span = np.concatenate([tail, pred])
    scores = reconstruction_scores(model, z[span], valid[span])
    name = f"{path.parent.parent.name.replace(' ', '_')}_{path.stem}"
    # Derived scores stay local, like earlier CARE score files.
    np.savez_compressed(
        out_dir / f"{name}.npz",
        times=times.iloc[span].to_numpy().astype("datetime64[ns]").astype(np.int64),
        scores=scores,
        calibration=np.isin(span, calibration),
        prediction=np.isin(span, pred),
        status=frame.status_type_id.iloc[span].to_numpy(),
    )
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=state, mean=mean, std=std), out_dir / f"{name}.pt")
    return dict(
        farm=path.parent.parent.name,
        event_id=int(path.stem),
        source_sha256=digest(path),
        features=int(x.shape[1]),
        fit_rows=len(fit),
        stop_rows=len(stop),
        calibration_rows=len(calibration),
        prediction_rows=len(pred),
        prediction_valid_fraction=float(valid[pred].mean()),
        epochs=len(history),
        best_stop_mse=float(min(history)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--farms", nargs="+", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--events", nargs="*", type=int, help="Optional subset of event ids")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.scores.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    audit_path = args.output / "training.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    done = {(a["farm"], a["event_id"]) for a in audit}
    for farm in args.farms:
        folder = args.root / farm
        for path in sorted((folder / "datasets").glob("*.csv"), key=lambda p: int(p.stem)):
            if (farm, int(path.stem)) in done or (
                args.events and int(path.stem) not in args.events
            ):
                continue
            started = time.monotonic()
            row = run_file(path, folder, args.scores, device=args.device)
            row["seconds"] = time.monotonic() - started
            audit.append(row)
            audit_path.write_text(json.dumps(audit, indent=2))
            print(f"{farm}/{path.stem}: {row['epochs']} epochs, {row['seconds']:.0f}s", flush=True)
    config = dict(
        protocol_sha256=digest(Path("docs/care-autoencoder-protocol.md")),
        script_sha256=digest(Path(__file__)),
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        seed=42,
        device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
    )
    (args.output / f"config-{'-'.join(f[-1] for f in args.farms)}.json").write_text(
        json.dumps(config, indent=2)
    )


if __name__ == "__main__":
    main()
