import hashlib
import io
import zipfile

import pytest

from scripts import acquire_care


def care_zip(extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("CARE_To_Compare/Wind Farm A/event_info.csv", "event_id\n")
        bundle.writestr("CARE_To_Compare/Wind Farm A/datasets/0.csv", "id;value\n" * 100)
        if extra:
            bundle.writestr(extra, "x")
    return buffer.getvalue()


def publish(tmp_path, monkeypatch, data):
    source = tmp_path / "zenodo.zip"
    source.write_bytes(data)
    monkeypatch.setattr(acquire_care, "ZENODO_URL", source.as_uri())
    monkeypatch.setattr(acquire_care, "ZENODO_BYTES", len(data))
    monkeypatch.setattr(acquire_care, "ZENODO_MD5", hashlib.md5(data).hexdigest())
    monkeypatch.setattr(acquire_care, "ZENODO_SHA256", hashlib.sha256(data).hexdigest())


def test_download_verify_extract_and_rerun(tmp_path, monkeypatch, capsys):
    publish(tmp_path, monkeypatch, care_zip())
    dest = tmp_path / "data"
    acquire_care.main(["--dest", str(dest)])
    assert (dest / "CARE_To_Compare/Wind Farm A/event_info.csv").is_file()
    out = capsys.readouterr().out
    assert "Download:" in out and "Verify:" in out and "Extract:" in out
    acquire_care.main(["--dest", str(dest)])
    assert "Already extracted" in capsys.readouterr().out


def test_partial_file_is_replaced_when_range_is_not_honoured(tmp_path, monkeypatch):
    data = care_zip()
    publish(tmp_path, monkeypatch, data)
    partial = tmp_path / "partial.zip"
    partial.write_bytes(b"garbage")
    acquire_care.download(acquire_care.ZENODO_URL, partial, len(data))
    assert partial.read_bytes() == data


def test_checksum_mismatch_stops_before_extraction(tmp_path, monkeypatch):
    publish(tmp_path, monkeypatch, care_zip())
    monkeypatch.setattr(acquire_care, "ZENODO_SHA256", "0" * 64)
    dest = tmp_path / "data"
    with pytest.raises(SystemExit, match="checksums"):
        acquire_care.main(["--dest", str(dest)])
    assert not (dest / "CARE_To_Compare").exists()


def test_extraction_refuses_paths_outside_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(care_zip(extra="../escape.txt"))
    with pytest.raises(ValueError, match="Unsafe path"):
        acquire_care.extract(archive, tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()
