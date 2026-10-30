# tlpr2s — SSMIS F16 2D-array precipitation retrieval

**Scientific status:** Source-based package starter; real-model numerical validation pending.

## Install

```bash
python -m pip install -e ".[plot,dev]"
python -m pytest -q
```

## Original pretrained XGBoost models

The uploaded pipeline did **not** contain the trained model JSONs or calibration NPZs.
This package is designed to use **your original trained XGBoost models**, not newly
trained or placeholder models. Copy all five files into `src/tlpr2s/model_data/`
**before building the public wheel**, or pass `model_dir` and `cdf_dir`.
See `src/tlpr2s/model_data/README.md`. If bundled, verify redistribution rights.

## Retrieve from 2D arrays (no MATLAB file needed)

```python
from tlpr2s import FEATURES, PrecipitationRetriever
# arrays: dict of 30 named 2D arrays; Month and Day may be scalars.
# arrays = {'19.35v': ..., ..., 'Month': 1, 'Day': 15}
retriever = PrecipitationRetriever()  # bundled original models
# Or: PrecipitationRetriever(model_dir='models', cdf_dir='cdf')
result = retriever.predict(arrays)
# result.phase, result.rain, result.snow, result.surface_precip
```

For a stacked array use `shape=(30,H,W)` in the exact `FEATURES` order.
These are *30 predictor maps*, not a single 2D map.

## Orbit visualization

```python
from tlpr2s.plotting import plot_orbit
plot_orbit(latitude_2d, longitude_2d, result.surface_precip,
           save='orbit.png')
```

## Annual statistics: 721 × 1440

```python
from tlpr2s import AnnualAccumulator
from tlpr2s.plotting import plot_annual
acc = AnnualAccumulator(resolution=0.25)
acc.add(latitude_2d, longitude_2d, result.rain, result.snow)
stats = acc.result()
assert stats.mean_precip.shape == (721, 1440)
plot_annual(stats, save='annual.png')
```

Latitude nodes: `-90,-89.75,...,90` (721). Longitude nodes:
`-180,-179.75,...,179.75` (1440). **-180 and +180 represent the same
longitude** and both map to column zero. Longitude wraps periodically.
Pixels map to the nearest grid node. Annual maps show observed swath-pixel
means and counts; they are **not unbiased annual precipitation totals**.

## Command line

```bash
tlpr2s retrieve input_features.npz --output orbit.npz
tlpr2s plot-orbit orbit.npz --output orbit.png
tlpr2s annual orbit1.npz orbit2.npz --output annual.npz
tlpr2s plot-annual annual.npz --output annual.png
```

## Public release

1. Add original trained model/calibration artifacts (or document external download).
2. Run tests and compare against the legacy pipeline on known SSMIS orbits.
3. Confirm code/model licensing and add author metadata.
4. `python -m build`, then `python -m twine check dist/*`.
5. Publish on TestPyPI before PyPI; confirm `tlpr2s` name availability.

# tlpr2s

**Research-ready packaging scaffold** for the existing SSMIS F16 TLP-R2S transfer-learning precipitation retrieval. The numerical inference algorithm (30 features, three XGBoost models, rain/snow CDF calibration) is adapted from the uploaded research pipeline. It accepts **2D NumPy arrays**, not complete satellite files. Orbit plotting and annual swath statistics are included.

> **Not a validated public science release yet.** The supplied archive did not contain trained models, CDF calibration arrays, representative orbit inputs, or reference outputs. Install and run unit tests, then perform an orbit-by-orbit numerical regression comparison before publishing results. The package is **not** a raw-SSMIS L1C reader or ERA5 collocation engine.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[plot,dev]'
python -m pytest -q
python examples/demo_visualization.py
```

### Model files you must provide

```
models/
  xgb_tl_classifier_final.json
  xgb_tl_rain_rate_final.json
  xgb_tl_snow_rate_final_1.json   # or xgb_tl_snow_rate_final.json
cdf/
  cdf_match_for_rain_ssmis.npz
  cdf_match_for_snow_ssmis.npz
```

These files **are not included**. Never commit private models, raw datasets, or secrets without permission. Store them in `models/` and `cdf/` locally (both ignored by Git).

### 2D array inference

```python
from tlpr2s import FEATURES, PrecipitationRetriever
# feature_arrays: dict with all 30 feature names; each value (H,W),
# except Month and Day may be scalars. These are REAL physical input fields.
retriever = PrecipitationRetriever(model_dir='models', cdf_dir='cdf')
result = retriever.predict(feature_arrays)
# result.phase: 0 clear, 1 rain, 2 snow
# result.rain, result.snow, result.surface_precip: 2D (H,W)
```

Or stack in the **exact order** of `tlpr2s.FEATURES` as `(30,H,W)`. An arbitrary 2D array alone is insufficient: the pretrained model needs 11 brightness temperatures, 17 ERA5 fields, Month and Day. It is essential that fields use the same units, footprint alignment, spatial ordering and preprocessing as the training data.

### Save and plot a swath

```python
from tlpr2s import save_swath
from tlpr2s.plotting import plot_orbit
save_swath('orbit.npz', latitude_2d, longitude_2d, result.phase, result.rain, result.snow)
plot_orbit(latitude_2d, longitude_2d, result.surface_precip, save='orbit.png')
```

### Annual statistics from many swaths

```python
from pathlib import Path
from tlpr2s import aggregate_swaths
from tlpr2s.plotting import plot_annual
stats = aggregate_swaths(sorted(Path('orbits').glob('*.npz')), resolution=0.25)
stats.save('annual_statistics.npz')
plot_annual(stats, 'mean_precip', save='annual_mean.png')
plot_annual(stats, 'observation_count', save='annual_coverage.png')
```

**Scientific interpretation:** The annual product is a pixel-weighted mean of *observed swath samples*, not annual accumulated rainfall, not a true area/time-weighted annual mean, and not a seamless global estimate. Regions with no valid observations are NaN. It uses a 721x1440 node grid (0.25°), with a periodic longitude seam, different from the legacy pipeline's 721x1441 grid. Overlapping swaths and temporal sampling can bias the statistics. `swath_coverage_count` counts swaths with valid samples in each cell. See `docs/SCIENTIFIC_LIMITATIONS.md`.

### CLI

```bash
tlpr2s retrieve features_and_geolocation.npz --models models --cdf cdf --output orbit.npz
tlpr2s plot-orbit orbit.npz --output orbit.png
tlpr2s annual orbits/*.npz --output annual.npz
tlpr2s plot-annual annual.npz --field mean_precip --output annual.png
```

`features_and_geolocation.npz` contains 30 named 2D feature arrays plus `latitude` and `longitude`. `Month` and `Day` can be scalar. See `examples/make_input_template.py`.

## Documentation

- `docs/INPUT_FORMAT.md` — exact model input contract and units warning
- `docs/SCIENTIFIC_LIMITATIONS.md` — annual aggregation, spatial geometry, and science caveats
- `docs/RELEASE_GUIDE.md` — end-to-end steps to GitHub, TestPyPI and PyPI
- `docs/MIGRATION_CHECKLIST.md` — parity tests against the uploaded research pipeline
- `examples/` — runnable synthetic plotting demo and array templates
- `legacy_reference/` — original scientific source files, **reference only**, not installed

## Licensing and citation

**No LICENSE has been assigned automatically.** Confirm software and model redistribution rights with all relevant contributors/institution before choosing an open-source license. Add `LICENSE`, `CITATION.cff`, authors, project URL and scientific publication DOI before a public release. Package name availability on PyPI has not been checked.
