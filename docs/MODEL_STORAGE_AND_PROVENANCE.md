# Hugging Face model storage and provenance

Model repository: https://huggingface.co/BuddhaSubedi/TLP-R2S

Expected exact filenames in repository root:

- `xgb_tl_classifier_final.json`
- `xgb_tl_rain_rate_final.json`
- `xgb_tl_snow_rate_final_1.json`
- `cdf_match_for_rain_ssmis.npz`
- `cdf_match_for_snow_ssmis.npz`

These files are **not** in this GitHub repository or the PyPI wheel. They are downloaded on first model initialization, not on import. `tlpr2s download-models` explicitly fetches them. The model repository must be public, or the user must authenticate.

## Reproducible model release

A branch name such as `main` can change. For a paper or production release, use the full commit SHA visible under Hugging Face `Files and versions` > `History` and pin it:

```bash
tlpr2s download-models --revision FULL_COMMIT_SHA
```

In Python:

```python
from tlpr2s import PrecipitationRetriever
retriever = PrecipitationRetriever(revision="FULL_COMMIT_SHA")
```

The code checks all five expected filenames. It does **not** currently verify SHA256 digests. For publication, record the immutable commit SHA and each file's SHA256 digest in the release notes and validate them during the real-model regression test. Keep the model card updated with training dataset provenance, supported satellite platform, required ERA5 inputs, intended use, known limitations, license/permission status, and an appropriate citation.
