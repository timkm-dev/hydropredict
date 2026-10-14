from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

from hydropredict.models.lstm import LSTMConfig, SHPLSTMAutoencoder, train_lstm


def normal_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(240, 6)), columns=[f"V{i}" for i in range(1, 7)])
    frame.insert(0, "t", pd.date_range("2026-01-01", periods=240, freq="5min", tz="UTC"))
    return frame.iloc[:120].copy(), frame.iloc[120:180].copy(), frame.iloc[180:].copy()


def configuration() -> LSTMConfig:
    return LSTMConfig(hidden_size=8, latent_size=4, batch_size=32, maximum_epochs=3, patience=3)


def test_training_export_cpu_inference_and_calibration(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fitting, stopping, calibration = normal_data()
    model = train_lstm(
        fitting,
        stopping,
        calibration,
        tmp_path / "run",
        configuration(),
        {"purpose": "synthetic engineering test"},
        device="cpu",
    )
    assert model.scaler.means["V1"] == pytest.approx(fitting.V1.mean())
    scores = model.score(calibration)
    assert model.threshold == pytest.approx(np.quantile(scores.anomaly_score, 0.99))
    loaded = SHPLSTMAutoencoder.load(tmp_path / "run/bundle")
    pd.testing.assert_frame_equal(scores, loaded.score(calibration.assign(fault_state=2)))
    error_columns = [f"V{i}_reconstruction_error" for i in range(1, 7)]
    np.testing.assert_allclose(scores.anomaly_score, scores[error_columns].mean(axis=1))
    prefix = model.score(calibration.iloc[:30])
    np.testing.assert_allclose(
        prefix.anomaly_score, scores.anomaly_score.iloc[: len(prefix)], rtol=1e-5, atol=1e-6
    )
    (tmp_path / "run/bundle/scaler.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SHPLSTMAutoencoder.load(tmp_path / "run/bundle")


def test_resume_matches_uninterrupted_and_rejects_changed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    fitting, stopping, calibration = normal_data()
    expected = train_lstm(
        fitting, stopping, calibration, tmp_path / "complete", configuration(), {}, device="cpu"
    )
    original_replace = Path.replace

    def interrupt_after_checkpoint(path: Path, target: Path) -> Path:
        result = original_replace(path, target)
        if path.name == "checkpoint.tmp":
            raise RuntimeError("Simulated interruption")
        return result

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="Simulated"):
            train_lstm(
                fitting,
                stopping,
                calibration,
                tmp_path / "interrupted",
                configuration(),
                {},
                device="cpu",
            )
    changed = fitting.copy()
    changed.loc[changed.index[0], "V1"] += 1
    with pytest.raises(ValueError, match="provenance"):
        train_lstm(
            changed,
            stopping,
            calibration,
            tmp_path / "interrupted",
            configuration(),
            {},
            device="cpu",
            resume=True,
        )
    actual = train_lstm(
        fitting,
        stopping,
        calibration,
        tmp_path / "interrupted",
        configuration(),
        {},
        device="cpu",
        resume=True,
    )
    np.testing.assert_allclose(
        expected.score(calibration).anomaly_score,
        actual.score(calibration).anomaly_score,
        rtol=1e-5,
        atol=1e-6,
    )


def test_training_boundaries_and_gpu_requirement(tmp_path: Path) -> None:
    fitting, stopping, calibration = normal_data()
    with pytest.raises(ValueError, match="disjoint"):
        train_lstm(
            fitting, fitting, calibration, tmp_path / "invalid", configuration(), {}, device="cpu"
        )
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="GPU"):
            train_lstm(fitting, stopping, calibration, tmp_path / "no_gpu", configuration(), {})
    with pytest.raises(RuntimeError, match="trained"):
        SHPLSTMAutoencoder().score(calibration)
