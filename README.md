# tlpr2s — SSMIS F16 transfer-learning rain and snow retrieval

**Python 3.10+ · 2D S4 swath inputs · original XGBoost models and CDF calibration · Hugging Face hosting · orbit visualization · annual grid · full legacy DAAC-oriented NetCDF**

The scientific model artifacts are hosted separately at [BuddhaSubedi/TLP-R2S](https://huggingface.co/BuddhaSubedi/TLP-R2S) (approximately 6.43 GB). **They are not included in the Python wheel.** The first model initialization downloads and caches the five original files; subsequent runs reuse the cache. No model download occurs when simply importing `tlpr2s`. The model repo must be public, or the user must authenticate with Hugging Face.

> **Scientific status:** This repository is an editable integration release. Real-orbit parity against the original pipeline and NASA DAAC acceptance have not been independently verified. The package is not NASA-endorsed. Do not publish it as validated before testing the actual trained artifacts.

## Installation

```bash
python -m pip install -e '.[all,dev]'
python -m pytest -q
```

For a public release after publishing on PyPI, users would run `pip install 'tlpr2s[all]'`. The package name's availability and release are not guaranteed.

## Download models (once)

```bash
tlpr2s download-models
# Better reproducibility: pin an immutable model repository commit SHA
# tlpr2s download-models --revision FULL_COMMIT_SHA
```

The downloader retrieves ONLY these files:

```
xgb_tl_classifier_final.json
xgb_tl_rain_rate_final.json
xgb_tl_snow_rate_final_1.json
cdf_match_for_rain_ssmis.npz
cdf_match_for_snow_ssmis.npz
```

**Alternative offline/HPC approach:** download on a login node, or copy the five original files to `/scratch/models/tlpr2s/`, then set `TLPR2S_MODEL_DIR=/scratch/models/tlpr2s` or pass `model_dir=...`. To use a cached model offline, call `PrecipitationRetriever(local_files_only=True)`. To prohibit downloads altogether, call `PrecipitationRetriever(download=False, model_dir=...)`. To select a particular model revision, pass `revision='FULL_COMMIT_SHA'` or set `TLPR2S_MODEL_REVISION`.

## Python API — S4-aligned 2D arrays, not full orbit files

```python
from tlpr2s import FEATURES, predict, plot_orbit, plot_annual, save_nc

# inputs: dict with 30 predictor keys; each is 2D (npix,nscan),
# except Month and Day may be scalar. The 11 lower-frequency SSMIS
# TB channels and 17 ERA5 fields MUST already be mapped to the S4 grid.
# See docs/INPUT_FORMAT.md for the exact feature names.
result = predict(inputs, revision='FULL_COMMIT_SHA')

# Result arrays have same (npix,nscan) shape.
phase = result.phase                 # 0=clear, 1=rain, 2=snow
rain = result.rain                   # mm/hr
snow = result.snow                   # mm/hr
precip = result.surface_precip

# Latitude and longitude must be 2D arrays on the same S4 grid.
plot_orbit(latitude, longitude, precip, vmax=15, save='orbit.png')
```

You may also use `PrecipitationRetriever(revision='FULL_COMMIT_SHA')` once and call `.predict(inputs)` on many orbits to avoid reloading multi-GB models for each orbit. The `predict` convenience function initializes the model each call, which is inefficient for batch processing.

### Annual grid and visualization

```python
from tlpr2s import AnnualAccumulator, plot_annual
acc = AnnualAccumulator(resolution=0.25)
for latitude, longitude, result in retrieved_orbits:
    acc.add(latitude, longitude, result.rain, result.snow)
annual = acc.result()
annual.save('annual_stats.npz')
plot_annual(annual, field='mean_precip', save='annual_mean.png')
```

Grid is **721 x 1440**: latitudes −90 to +90 inclusive, longitudes −180 to +179.75, with +180 and −180 assigned to the same column. These are swath-sample means and counts, **not unbiased annual accumulation**. See `docs/SCIENTIFIC_LIMITATIONS.md`.

### Save the full legacy DAAC-oriented NetCDF

```python
from tlpr2s import save_nc
save_nc('TLP-R2S_orbit.nc', result=result,
        latitude=latitude, longitude=longitude,
        input_arrays=original_s4_fields,
        ssmis_full_filename=original_ssmis_path.name,
        ssmis_source_path=original_ssmis_path,
        model_version='TLP-R2S_V01')
```

The writer preserves the original product layout, SSMIS TB and ERA5 fields, and actual source scan-time metadata. **A complete source SSMIS granule is required** to recover authentic times; a filename alone is insufficient. The NetCDF schema is DAAC-oriented, not formally certified. See `docs/DAAC_NETCDF.md`.

## Command-line tools

```bash
tlpr2s download-models
tlpr2s retrieve prepared_s4_arrays.npz --output orbit.npz
tlpr2s plot-orbit orbit.npz --output orbit.png
tlpr2s annual orbit1.npz orbit2.npz --output annual.npz
tlpr2s plot-annual annual.npz --output annual.png
```

## Publish on GitHub / PyPI

See `docs/PUBLISH_GITHUB_PYPI.md`, `docs/RELEASE_GUIDE.md`, `docs/INPUT_FORMAT.md`, and `examples/`. Keep model artifacts **outside** the GitHub repository and wheel. Review licenses, model provenance, original algorithm numerical parity, and the DAAC's product requirements before publication. The example `CITATION.cff` and `LICENSE-TO-SELECT.txt` need owner review.

## GitHub upload checklist

Start with [docs/GITHUB_UPLOAD_STEPS.md](docs/GITHUB_UPLOAD_STEPS.md) and [docs/MODEL_STORAGE_AND_PROVENANCE.md](docs/MODEL_STORAGE_AND_PROVENANCE.md). The model artifacts are hosted on Hugging Face and must **not** be committed to GitHub.
