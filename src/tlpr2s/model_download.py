"""Download official model artifacts from a versioned Hugging Face repository.

Downloads are opt-in at the first retrieval initialization, never at import time.
Set TLPR2S_MODEL_DIR to an existing offline folder to avoid network requests.
"""
from pathlib import Path
import os

MODEL_REPO = 'BuddhaSubedi/TLP-R2S'
MODEL_FILES = (
    'xgb_tl_classifier_final.json',
    'xgb_tl_rain_rate_final.json',
    'xgb_tl_snow_rate_final_1.json',
    'cdf_match_for_rain_ssmis.npz',
    'cdf_match_for_snow_ssmis.npz',
)

def download_models(*, revision=None, cache_dir=None, local_files_only=False):
    """Return cached path to all five ORIGINAL pretrained artifacts.

    Parameters
    ----------
    revision: str, optional
        Pin a Hugging Face commit SHA or release tag for reproducibility.
        Defaults to TLPR2S_MODEL_REVISION or main (mutable).
    cache_dir: str, optional
        Hugging Face cache root; otherwise use standard HF cache.
    local_files_only: bool
        Require existing cached files and never access the network.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError('Install model downloads with: pip install "tlpr2s[hub]"') from exc
    revision = revision or os.environ.get('TLPR2S_MODEL_REVISION', 'main')
    path = Path(snapshot_download(
        repo_id=MODEL_REPO,
        repo_type='model',
        revision=revision,
        allow_patterns=list(MODEL_FILES),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ))
    missing = [name for name in MODEL_FILES if not (path/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing original TLP-R2S artifacts in {path}: {missing}')
    return path
