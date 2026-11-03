from __future__ import annotations

import ast
import json
from pathlib import Path


def test_notebook_is_thin_compilable_and_has_no_fabricated_outputs() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/train_lstm_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            ast.parse(text)
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            sources.append(text)
    code = "\n".join(sources)
    assert "scripts/train_lstm.py" in code
    assert "scripts/evaluate_lstm.py" in code
    assert "checkpoint.pt" in code
    assert "drive.mount" in code
    assert "cuda.is_available" in code
    assert "loss.backward" not in code


def test_care_v2_notebook_is_thin_verified_and_resumable() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/train_care_autoencoder_v2_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            ast.parse(text)
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            sources.append(text)
    code = "\n".join(sources)
    assert "scripts.train_care_autoencoder_v2" in code
    assert 'manifest["zenodo_md5"]' in code
    assert "correlation" in code and "RuntimeError" in code
    assert "drive.mount" in code and "care-autoencoder-v2" in code
    assert "loss.backward" not in code
