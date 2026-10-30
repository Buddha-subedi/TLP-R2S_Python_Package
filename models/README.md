# Model assets are hosted externally

The five original XGBoost/CDF artifacts (about 6.43 GB combined) are hosted at:
https://huggingface.co/BuddhaSubedi/TLP-R2S

Do not commit them to GitHub or package them in a PyPI wheel.

Run `tlpr2s download-models` or `from tlpr2s import download_models; path = download_models()`.
Use `revision='FULL_COMMIT_SHA'` to pin a scientifically reproducible release.
Offline users can pass `model_dir='/path/to/files'` (and optionally `cdf_dir`) or set `TLPR2S_MODEL_DIR`.

Ensure release rights, provenance, artifact integrity and model card are complete before making the model repository public.
