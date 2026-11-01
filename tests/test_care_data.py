import pytest

from hydropredict.data import care


def make_care(path):
    farm = path / "Wind Farm B"
    farm.mkdir(parents=True)
    (farm / "event_info.csv").write_text("event_id;event_label\n")
    return path


def test_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(care.ENV_VAR, str(tmp_path / "elsewhere"))
    root = make_care(tmp_path / "care")
    assert care.care_root(root) == root


def test_environment_variable_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv(care.ENV_VAR, str(make_care(tmp_path / "care")))
    assert care.care_root() == tmp_path / "care"


def test_missing_and_invalid_locations_explain_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.delenv(care.ENV_VAR, raising=False)
    monkeypatch.setattr(care, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="acquire_care.py"):
        care.care_root()
    with pytest.raises(ValueError, match="not a CARE to Compare folder"):
        care.care_root(tmp_path)
