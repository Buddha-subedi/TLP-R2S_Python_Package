# Hugging Face model integration

Repository: https://huggingface.co/BuddhaSubedi/TLP-R2S

Five original model/CDF files are at repository root; the combined repository size shown by the owner is ~6.43 GB. `src/tlpr2s/model_download.py` downloads only those exact five names, using `huggingface_hub.snapshot_download`, with its default local cache.

1. Verify repository visibility and permissions. Private repositories require `hf auth login` or `HF_TOKEN`.
2. Pin the model commit SHA (recommended), not `main`, for reproducible science. `TLPR2S_MODEL_REVISION` or `revision=` selects it.
3. Download once on a machine with adequate disk and internet: `tlpr2s download-models --revision SHA`.
4. For HPC compute nodes with no network, set `TLPR2S_MODEL_DIR` to a shared directory holding the five files, or prepopulate the HF cache and use `local_files_only=True`.
5. Check model licenses, SHA256 digests, model training provenance, XGBoost runtime compatibility, and test numerical parity with the original orbit pipeline.

The default `predict(inputs)` convenience function may download models if no local model directory is configured. `import tlpr2s` never downloads. Download size and time can be substantial. The library does not silently substitute dummy models.
