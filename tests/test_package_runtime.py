import pytest

from scripts import package_runtime as runtime


def make_runtime(root):
    for source in runtime.SOURCES:
        path = root / source if source.endswith(".json") else root / source / "file.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (root / "data/raw/cwru").mkdir(parents=True)
    (root / "data/cwru_manifest.csv").write_text("recording,load_hp\n97.mat,0\n100.mat,3\n")
    for name in ("97.mat", "100.mat"):
        (root / "data/raw/cwru" / name).write_bytes(name.encode())


def test_runtime_package_holds_only_served_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    make_runtime(repo)
    sources = runtime.runtime_sources(repo)
    assert "data/raw/cwru/100.mat" in sources
    assert "data/raw/cwru/97.mat" not in sources

    monkeypatch.setattr(runtime, "ROOT", repo)
    archive = tmp_path / "runtime.zip"
    runtime.main(["--output", str(archive)])
    restored = tmp_path / "fresh"
    monkeypatch.setattr(runtime, "ROOT", restored)
    runtime.main(["--restore", str(archive)])
    assert (restored / "frontend/evidence-data.json").read_text() == "frontend/evidence-data.json"
    assert (restored / "data/raw/cwru/100.mat").exists()
    assert not (restored / "data/raw/cwru/97.mat").exists()
    assert not (restored / runtime.MANIFEST).exists()


def test_missing_runtime_files_stop_packaging(tmp_path):
    make_runtime(tmp_path)
    (tmp_path / "data/raw/cwru/100.mat").unlink()
    with pytest.raises(SystemExit, match="1 runtime files are missing"):
        runtime.runtime_sources(tmp_path)
