import json
import zipfile

import pytest

from scripts import backup_care_models as backup


def make_models(root):
    for source, name in [(backup.SOURCES[0], "A_0.pt"), (backup.SOURCES[1], "scores/C_1.npz")]:
        path = root / source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode() * 50)


def test_backup_round_trip_and_tamper_detection(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    make_models(repo)
    archive = tmp_path / "backup.zip"
    manifest = backup.create(repo, backup.SOURCES, archive)
    assert sorted(manifest["files"]) == [
        ".local/care-autoencoder-v1-C-colab/scores/C_1.npz",
        ".local/care-autoencoder-v1-scores/A_0.pt",
    ]
    assert backup.verify(archive) == []

    restored = tmp_path / "fresh"
    monkeypatch.setattr(backup, "ROOT", restored)
    backup.main(["--restore", str(archive)])
    assert (restored / ".local/care-autoencoder-v1-scores/A_0.pt").read_bytes() == (
        repo / ".local/care-autoencoder-v1-scores/A_0.pt"
    ).read_bytes()
    assert not (restored / backup.MANIFEST).exists()

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            target.writestr(name, b"changed" if name.endswith(".pt") else data)
    assert backup.verify(tampered) == [".local/care-autoencoder-v1-scores/A_0.pt"]
    with pytest.raises(SystemExit, match="fail verification"):
        backup.main(["--verify", str(tampered)])
    assert json.loads(zipfile.ZipFile(archive).read(backup.MANIFEST))["files"]


def test_backup_refuses_empty_sources(tmp_path):
    with pytest.raises(SystemExit, match="No model files"):
        backup.create(tmp_path, backup.SOURCES, tmp_path / "out.zip")
