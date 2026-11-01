"""CWRU source isolation, sample accounting and exact saved inference."""

import base64
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("xgboost")

from hydropredict.assessment_store import AssessmentStore
from hydropredict.bearing_workflow import BearingWorkflow
from hydropredict.models.bearing import FEATURES, extract, read_signal


def mat(**arrays):
    stream = io.BytesIO()
    savemat(stream, arrays)
    return stream.getvalue()


def test_ambiguous_official_file_uses_own_recording_id():
    own = np.sin(np.arange(20000) / 11).reshape(-1, 1)
    other = np.cos(np.arange(20000) / 4).reshape(-1, 1)
    content = mat(X175_DE_time=own, X217_DE_time=other)
    np.testing.assert_array_equal(read_signal(content, "175.mat"), own.ravel())
    np.testing.assert_array_equal(read_signal(content, "217.mat"), other.ravel())
    with pytest.raises(ValueError, match="Ambiguous"):
        read_signal(content, "renamed.mat")
    with pytest.raises(ValueError, match="Ambiguous"):
        read_signal(content, "999.mat")


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((20000, 1)),
        np.full((20000, 1), np.nan),
        np.ones((100, 2)),
        np.ones((20000, 1), dtype=complex),
    ],
)
def test_invalid_signal_rejected(array):
    with pytest.raises(ValueError):
        read_signal(mat(X100_DE_time=array))


def test_feature_accounting_and_batch_independence():
    values = np.random.default_rng(42).normal(size=40000)
    features, accounting = extract(values)
    assert features.shape == (2, len(FEATURES))
    assert accounting["unused_resampled_samples"] == 1808
    assert accounting["original_samples"] == 40000
    assert np.isfinite(features).all()
    replay, _ = extract(values.copy())
    np.testing.assert_array_equal(features, replay)


def test_upload_provenance_persistence_and_owner_isolation(tmp_path):
    workflow = BearingWorkflow.__new__(BearingWorkflow)
    workflow.root = tmp_path
    workflow.store = AssessmentStore(tmp_path / "results.sqlite3")
    workflow.examples = {}
    content = mat(X100_DE_time=np.sin(np.arange(20000) / 11).reshape(-1, 1))

    def assess(data, name):
        assert name == "100.mat"
        assert np.array_equal(read_signal(data, name), read_signal(content, name))
        return {"predicted_condition": "Healthy", "scores": []}

    workflow.service = SimpleNamespace(assess=assess)
    payload = {"name": "100.mat", "sample_rate": 48000, "data": base64.b64encode(content).decode()}
    run = workflow.assess("alice", payload)
    digest = hashlib.sha256(content).hexdigest()
    assert run["result"]["source_sha256"] == digest
    assert (tmp_path / "data/raw/cwru-imports" / f"{digest}.mat").read_bytes() == content
    assert (
        AssessmentStore(tmp_path / "results.sqlite3").get("alice", run["id"])["name"] == "100.mat"
    )
    with pytest.raises(ValueError, match="not found"):
        workflow.store.get("bob", run["id"])
    for invalid in [
        {"sample_rate": 12000},
        {"data": "not base64!"},
        {"example": "../../secret.mat"},
    ]:
        with pytest.raises(ValueError):
            workflow.assess("alice", payload | invalid)
    assert len(workflow.store.list("alice")) == 1
