# Hugging Face model integration

Repository: https://huggingface.co/BuddhaSubedi/TLP-R2S

Five original model/CDF files are at repository root; the combined repository size is ~6.43 GB. `src/tlpr2s/model_download.py` downloads only those exact five names, using `huggingface_hub.snapshot_download`, with its default local cache.

The default `predict(inputs)` convenience function may download models if no local model directory is configured. `import tlpr2s` never downloads. Download size and time can be substantial. The library does not silently substitute dummy models.
