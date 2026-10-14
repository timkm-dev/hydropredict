"""Compact LSTM autoencoder, persistent training checkpoints and CPU inference."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.preprocessing import StandardTelemetryScaler
from hydropredict.data.schema import MEASUREMENT_COLUMNS
from hydropredict.data.sequences import SequenceBatch, build_sequences


@dataclass(frozen=True)
class LSTMConfig:
    sequence_length: int = 7
    minimum_step_seconds: float = 150
    maximum_step_seconds: float = 450
    hidden_size: int = 32
    latent_size: int = 16
    batch_size: int = 128
    learning_rate: float = 0.001
    maximum_epochs: int = 30
    patience: int = 5
    minimum_improvement: float = 0.00001
    calibration_quantile: float = 0.99
    random_seed: int = 42

    def __post_init__(self) -> None:
        if (
            any(
                v < 1
                for v in (
                    self.hidden_size,
                    self.latent_size,
                    self.batch_size,
                    self.maximum_epochs,
                    self.patience,
                )
            )
            or self.sequence_length < 2
        ):
            raise ValueError("Model dimensions, epochs and patience must be positive.")
        if not 0 < self.minimum_step_seconds <= self.maximum_step_seconds:
            raise ValueError("Invalid cadence limits.")
        if not 0 < self.calibration_quantile < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid calibration quantile or learning rate.")
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite.")
        if self.minimum_improvement < 0:
            raise ValueError("Minimum improvement cannot be negative.")


class Autoencoder(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.encoder = nn.LSTM(6, config.hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(config.hidden_size, config.latent_size)
        self.decoder = nn.LSTM(config.latent_size, config.hidden_size, batch_first=True)
        self.output = nn.Linear(config.hidden_size, 6)

    def forward(self, inputs: Tensor) -> Tensor:
        _, (hidden, _) = self.encoder(inputs)
        latent = self.bottleneck(hidden[-1])
        repeated = latent.unsqueeze(1).repeat(1, inputs.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return cast(Tensor, self.output(decoded))


class SHPLSTMAutoencoder:
    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()
        self.network = Autoencoder(self.config)
        self.scaler = StandardTelemetryScaler()
        self.threshold: float | None = None
        self.version = "untrained"

    def sequences(self, frame: pd.DataFrame) -> SequenceBatch:
        # Validate raw finite values before scaling; select only the six declared channels.
        build_sequences(
            frame,
            self.config.sequence_length,
            self.config.minimum_step_seconds,
            self.config.maximum_step_seconds,
        )
        return build_sequences(
            self.scaler.transform(frame),
            self.config.sequence_length,
            self.config.minimum_step_seconds,
            self.config.maximum_step_seconds,
        )

    def errors(
        self, values: np.ndarray[Any, np.dtype[np.float32]], device: str = "cpu"
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if len(values) == 0:
            return np.empty((0, 6), dtype=np.float32)
        self.network.to(device).eval()
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(values), self.config.batch_size):
                batch = torch.from_numpy(values[start : start + self.config.batch_size]).to(device)
                error = (self.network(batch) - batch).square().mean(dim=1)
                pieces.append(error.cpu().numpy())
        result = np.concatenate(pieces).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite reconstruction errors.")
        return result

    def score(self, frame: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
        if self.threshold is None:
            raise RuntimeError("A trained, calibrated bundle is required.")
        sequences = self.sequences(frame)
        errors = self.errors(sequences.values, device)
        result = sequences.endpoints.copy()
        result["anomaly_score"] = errors.mean(axis=1)
        result["threshold"] = self.threshold
        result["is_alert"] = result.anomaly_score > self.threshold
        result["model_version"] = self.version
        for index, channel in enumerate(MEASUREMENT_COLUMNS):
            result[f"{channel}_reconstruction_error"] = errors[:, index]
        return result

    def save(self, directory: Path, provenance: dict[str, Any]) -> None:
        if self.threshold is None or not math.isfinite(self.threshold):
            raise ValueError("Only calibrated finite-threshold models can be exported.")
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(
            {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
            directory / "weights.pt",
        )
        self.scaler.save(directory / "scaler.json")
        metadata = {
            "format_version": 1,
            "model_version": self.version,
            "config": asdict(self.config),
            "feature_columns": list(MEASUREMENT_COLUMNS),
            "threshold": self.threshold,
            "score": "Mean squared standardised reconstruction error over sequence and channels",
            "timestamp": "Last observed reading in the backward-only sequence",
            "explanation": "Per-channel mean squared errors; not causal feature attributions",
            "alert_rule": "score > threshold; no persistence or cooldown",
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "provenance": provenance,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        hashes = {
            name: compute_file_sha256(directory / name)
            for name in ("weights.pt", "scaler.json", "metadata.json")
        }
        (directory / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> SHPLSTMAutoencoder:
        directory = Path(directory)
        hashes = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        for name in ("weights.pt", "scaler.json", "metadata.json"):
            if compute_file_sha256(directory / name) != hashes[name]:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["feature_columns"] != list(MEASUREMENT_COLUMNS):
            raise ValueError("Incompatible feature schema.")
        model = cls(LSTMConfig(**metadata["config"]))
        model.network.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        model.network.eval()
        model.scaler = StandardTelemetryScaler.load(directory / "scaler.json")
        if list(model.scaler.means) != list(MEASUREMENT_COLUMNS):
            raise ValueError("Incompatible scaler schema.")
        model.threshold = float(metadata["threshold"])
        if not math.isfinite(model.threshold) or model.threshold < 0:
            raise ValueError("Invalid threshold.")
        model.version = str(metadata["model_version"])
        return model


def _state(network: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def train_lstm(
    fitting: pd.DataFrame,
    stopping: pd.DataFrame,
    calibration: pd.DataFrame,
    output: Path,
    config: LSTMConfig,
    provenance: dict[str, Any],
    *,
    device: str = "cuda",
    resume: bool = False,
) -> SHPLSTMAutoencoder:
    """Fit, early-stop and calibrate on three distinct chronological normal subsets."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "No hosted GPU available. Select a Colab GPU runtime before full training."
        )
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    if not (
        pd.to_datetime(fitting.t, utc=True).max() < pd.to_datetime(stopping.t, utc=True).min()
        and pd.to_datetime(stopping.t, utc=True).max()
        < pd.to_datetime(calibration.t, utc=True).min()
    ):
        raise ValueError("Fitting, early stopping and calibration must be disjoint and ordered.")
    if (output / "bundle").exists():
        raise FileExistsError("Completed run exists; use a new run name.")
    if output.exists() and not resume:
        raise FileExistsError("Run exists; explicitly resume or choose a new name.")
    input_hashes = {
        name: hashlib.sha256(
            pd.util.hash_pandas_object(frame[["t", *MEASUREMENT_COLUMNS]], index=False)
            .to_numpy()
            .tobytes()
        ).hexdigest()
        for name, frame in (
            ("fitting", fitting),
            ("early_stopping", stopping),
            ("calibration", calibration),
        )
    }
    signature = hashlib.sha256(
        json.dumps(
            {"config": asdict(config), "provenance": provenance, "inputs": input_hashes},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    model = SHPLSTMAutoencoder(config)
    model.scaler.fit(fitting)
    datasets = {
        name: model.sequences(frame)
        for name, frame in (
            ("fitting", fitting),
            ("early_stopping", stopping),
            ("calibration", calibration),
        )
    }
    if any(len(data.values) < 16 for data in datasets.values()):
        raise ValueError("Each normal partition needs at least sixteen eligible sequences.")
    output.mkdir(parents=True, exist_ok=resume)
    model.network.to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    best_loss, bad_epochs, first_epoch = float("inf"), 0, 0
    best_state = _state(model.network)
    elapsed_before = 0.0
    checkpoint_path = output / "checkpoint.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume configuration or input provenance changed.")
        model.network.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_state, best_loss = checkpoint["best_state"], checkpoint["best_loss"]
        bad_epochs, first_epoch = checkpoint["bad_epochs"], checkpoint["epoch"] + 1
        history, elapsed_before = checkpoint["history"], checkpoint["elapsed_seconds"]
    start = perf_counter()
    for epoch in range(first_epoch, config.maximum_epochs):
        if bad_epochs >= config.patience:
            break
        generator = torch.Generator().manual_seed(config.random_seed + epoch)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(datasets["fitting"].values)),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )
        model.network.train()
        total = 0.0
        for (inputs,) in loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            loss = (model.network(inputs) - inputs).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(inputs)
        validation_loss = float(model.errors(datasets["early_stopping"].values, device).mean())
        training_loss = total / len(datasets["fitting"].values)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "early_stopping_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss, best_state, bad_epochs = validation_loss, _state(model.network), 0
        else:
            bad_epochs += 1
        checkpoint = {
            "signature": signature,
            "epoch": epoch,
            "state": _state(model.network),
            "optimizer": optimizer.state_dict(),
            "best_state": best_state,
            "best_loss": best_loss,
            "bad_epochs": bad_epochs,
            "history": history,
            "elapsed_seconds": elapsed_before + perf_counter() - start,
        }
        temporary = output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        print(json.dumps(history[-1]), flush=True)
    model.network.load_state_dict(best_state)
    errors = model.errors(datasets["calibration"].values, device)
    model.threshold = float(np.quantile(errors.mean(axis=1), config.calibration_quantile))
    model.version = output.name
    report = {
        "status": "trained; fault-based validation not yet performed",
        "device": device,
        "history": history,
        "best_early_stopping_loss": best_loss,
        "training_seconds": elapsed_before + perf_counter() - start,
        "sequence_accounting": {k: v.accounting for k, v in datasets.items()},
        "provenance": provenance,
        "threshold": model.threshold,
        "partition_ranges": {
            name: {
                "start": pd.to_datetime(frame.t, utc=True).min().isoformat(),
                "end": pd.to_datetime(frame.t, utc=True).max().isoformat(),
            }
            for name, frame in (
                ("fitting", fitting),
                ("early_stopping", stopping),
                ("calibration", calibration),
            )
        },
    }
    model.save(output / "bundle", report)
    # Export verification is bounded and compares GPU predictions with local CPU semantics.
    check = datasets["calibration"].values[:32]
    expected = model.errors(check, device)
    loaded = SHPLSTMAutoencoder.load(output / "bundle")
    np.testing.assert_allclose(expected, loaded.errors(check), rtol=1e-4, atol=1e-6)
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return loaded
