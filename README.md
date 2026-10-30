# tlpr2s — Transfer Learning for Passive Microwave Rain and Snow Retrieval

**SSMIS F16 · 2D S4-swath arrays · XGBoost · CDF-calibrated precipitation · orbit and annual visualization · full original DAAC-oriented NetCDF**

> **Release status:** Editable public-repository starter. The five trained artifacts were not included in the supplied source archive; real-model inference and archive acceptance are **not yet independently validated**. Do not advertise this as NASA-approved or DAAC-accepted.

## 1. Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[plot,io,dev]"
python -m pytest -q
```

Python >=3.10. This project preserves the legacy trained feature order and XGBoost/CDF algorithm. To publish: see `docs/PUBLISH_GITHUB_PYPI.md`.

## 2. Where to put your XGBoost models and CDF files

**Place these EXACT original files in `src/tlpr2s/model_data/` BEFORE building the distributable wheel:**

```text
src/tlpr2s/model_data/
    xgb_tl_classifier_final.json
    xgb_tl_rain_rate_final.json
    xgb_tl_snow_rate_final_1.json
    cdf_match_for_rain_ssmis.npz
    cdf_match_for_snow_ssmis.npz
```

`pyproject.toml` includes `model_data/*.json` and `model_data/*.npz` as package data. Thus the files are included in a built wheel when present. Verify with `python -m zipfile -l dist/*.whl` (or inspect the wheel with Python's zipfile module). **Do not publish without confirming redistribution rights.**

If you do not want to bundle them, use `PrecipitationRetriever(model_dir='/models', cdf_dir='/models')`. The five artifacts must match the training version and feature order; no model retraining is performed by the package.

## 3. Predict from 2D arrays (NO MATLAB file required)

```python
from tlpr2s import FEATURES, PrecipitationRetriever
# input_arrays: dict[str, np.ndarray], each feature shape (npix,nscan)
# Month and Day can also be scalar. All 30 trained predictors required.
features = {key: input_arrays[key] for key in FEATURES}
model = PrecipitationRetriever()
result = model.predict(features)
phase = result.phase            # 0 clear, 1 rain, 2 snow
rain = result.rain              # mm/hour, after legacy CDF matching
snow = result.snow
surface_precip = result.surface_precip
```

Alternatively supply a NumPy array with shape `(30,npix,nscan)` **in the exact `FEATURES` order**. The SSMIS S1/S2/S3 brightness temperatures must already be collocated onto the S4 swath; this package does not automatically remap raw channels or fetch ERA5.

### Required feature order

```python
from tlpr2s import FEATURES
print(FEATURES)
```

The 30 features are 11 TB channels, 17 ERA5 predictors, Month and Day. Their exact names and order are defined in `src/tlpr2s/features.py` and must not be rearranged. Source MAT aliases are in `FIELD_MAP` and `TB_FIELDS`.

## 4. Plot one S4 orbit

```python
from tlpr2s import plot_orbit
fig, ax = plot_orbit(latitude_2d, longitude_2d, result.surface_precip,
                     title='SSMIS F16 TLP-R2S', vmax=15, save='orbit.png')
```

All coordinates and data must have the same S4 `(npix,nscan)` shape. Cartopy is optional but recommended.

## 5. Aggregate swaths and plot annual statistics

```python
from tlpr2s import AnnualAccumulator, plot_annual
acc = AnnualAccumulator(resolution=0.25)
for lat, lon, result in swaths:   # one entry per retrieved orbit
    acc.add(lat, lon, result.rain, result.snow)
stats = acc.result()
stats.save('annual_stats.npz')
plot_annual(stats, field='mean_precip', save='annual_mean.png')
plot_annual('annual_stats.npz', field='observation_count', save='annual_counts.png')
```

Or save each orbit using `save_swath(...)`, then run `examples/annual_from_orbits.py`.

**Grid:** latitude `-90..90` inclusive = **721**, longitude `-180..179.75` periodic = **1440**, at 0.25°. `+180` and `-180` are the same longitude cell. Current means are **observed-swath sample means**, not a time-complete annual accumulation; swath sampling biases and missing hours are not corrected. `swath_coverage_count` counts how many `add` calls hit a cell; `observation_count` counts mapped pixels.

## 6. Save the FULL original DAAC-oriented NetCDF

```python
from tlpr2s import save_nc
save_nc('TLP-R2S_orbit.nc', result=result,
        latitude=latitude_2d, longitude=longitude_2d,
        input_arrays=original_s4_fields,
        ssmis_full_filename=original_ssmis_path.name,
        ssmis_source_path=original_ssmis_path,
        model_version='TLP-R2S_V01')
```

**Important:** `original_s4_fields` must include `TbS1`, `TbS2`, `TbS3`, `TbS4` (3D channel stacks, S4 aligned) **or** all individual 2D channels, plus the complete original `ERA5_*` fields. It is not necessarily the same dict as the 30 normalized feature names. The writer preserves the original `products.py` schema: 2D `npix,nscan`, `Latitude`, `Longitude`, `Time`, SSMIS channels, ERA5 inputs, and `TLP-R2S/Phase` and `TLP-R2S/SurfacePrecip` with original metadata conventions. The **full original SSMIS filename and accessible source granule** are required to recover genuine scan times. Merely providing a filename cannot reconstruct time metadata. See `docs/DAAC_NETCDF.md`.

This is **DAAC-oriented**, not a claim of formal NASA DAAC acceptance. Coordinate, product naming, metadata, and data validation must be approved by the intended archive.

## 7. Example scripts

- `examples/complete_workflow.py`: 2D-array prediction, orbit plot, annual map, and NetCDF template.
- `examples/annual_from_orbits.py`: batch aggregate saved orbit NPZ files.
- `examples/retrieve_real_arrays.py`: standalone inference example.
- `examples/save_full_daac_netcdf.py`: full metadata writer example.
- `examples/demo_visualization.py`: synthetic plotting demonstration only.

## 8. Repository and publication

Read `docs/PUBLISH_GITHUB_PYPI.md`, `docs/INPUT_FORMAT.md`, `docs/DAAC_NETCDF.md`, and `docs/SCIENTIFIC_LIMITATIONS.md`. `legacy_reference/` contains your original scientific pipeline for comparison. Before release, add the appropriate license and authors, verify numerical equivalence on real orbits, and confirm artifact redistribution permissions.
