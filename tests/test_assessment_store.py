import pytest

from hydropredict.assessment_store import AssessmentStore


def test_restart_ownership_and_append_only_review(tmp_path):
    path = tmp_path / "assessments.sqlite3"
    store = AssessmentStore(path)
    result = {"scores": [{"t": "2019-01-01T00:00:00Z", "is_alert": True}]}
    run_id = store.save("alice", "recording.csv", [], result)
    store.review(
        "alice",
        {
            "run_id": run_id,
            "episode": result["scores"][0]["t"],
            "action": "acknowledged",
            "comment": "Inspect",
        },
    )
    restored = AssessmentStore(path)
    assert restored.get("alice", run_id)["reviews"][0]["comment"] == "Inspect"
    assert restored.list("bob") == []
    with pytest.raises(ValueError, match="not found"):
        restored.get("bob", run_id)
    with pytest.raises(ValueError, match="not found"):
        restored.review("bob", {"run_id": run_id})
    restored.review(
        "alice", {"run_id": run_id, "episode": result["scores"][0]["t"], "action": "open"}
    )
    assert len(restored.get("alice", run_id)["reviews"]) == 2
    assert restored.get("alice", run_id)["result"] == result


def test_invalid_reviews_are_not_saved(tmp_path):
    store = AssessmentStore(tmp_path / "assessments.sqlite3")
    run_id = store.save("a", "data.csv", [], {"scores": [{"t": "t", "is_alert": True}]})
    for updates in [
        {"episode": "absent"},
        {"action": "healthy"},
        {"action": "dismissed", "comment": " "},
        {"comment": "x" * 2001},
    ]:
        payload = {
            "run_id": run_id,
            "episode": "t",
            "action": "acknowledged",
            "comment": "",
        } | updates
        with pytest.raises(ValueError):
            store.review("a", payload)
    assert store.get("a", run_id)["reviews"] == []
