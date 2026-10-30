from pathlib import Path
import pytest
from tlpr2s.model_download import MODEL_FILES, MODEL_REPO, download_models

def test_repo_and_artifact_contract():
    assert MODEL_REPO == 'BuddhaSubedi/TLP-R2S'
    assert len(MODEL_FILES) == 5
    assert len(set(MODEL_FILES)) == 5

def test_downloader_with_mock(monkeypatch, tmp_path):
    for f in MODEL_FILES: (tmp_path / f).write_bytes(b'test')
    import huggingface_hub
    called = {}
    def fake_snapshot_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake_snapshot_download)
    assert download_models(revision='123abc',local_files_only=True) == tmp_path
    assert called['revision'] == '123abc'
    assert called['local_files_only'] is True
    assert set(called['allow_patterns']) == set(MODEL_FILES)

def test_missing_asset_detected(monkeypatch,tmp_path):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda **kw: str(tmp_path))
    with pytest.raises(FileNotFoundError, match='Missing original'):
        download_models()
