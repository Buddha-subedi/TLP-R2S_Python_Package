# Full original SSMIS F16 DAAC-oriented NetCDF product

The public Python function is `from tlpr2s import DAAC_compatible_nc`.
Python identifiers cannot contain spaces; `from tlpr2s import DAAC_compatible nc` is invalid syntax.

The code reuses the original `products.py` writer, not a simplified replacement.
Its original groups, attributes, variable names, dimensions, compression and
S4 scan times are retained. Original output includes root `Latitude`,
`Longitude`, `Time`, `mapping`, group `TLP-R2S` (`Phase`, `SurfacePrecip`),
`Input/SSMIS` mapped brightness temperatures, `Input/ERA5`, and
`Reference/ERA5` (`msr`, `mtpr` when supplied).

## Example

```python
from tlpr2s import DAAC_compatible_nc
DAAC_compatible_nc(
    'TLP-R2S_V01_1C.F16.SSMIS.XCAL2021-V.20230101-S000000-E013000.000001.V07A.nc',
    result=result, latitude=lat_s4, longitude=lon_s4,
    input_arrays=all_original_s4_fields,
    ssmis_full_filename='1C.F16.SSMIS.XCAL2021-V.20230101-S000000-E013000.000001.V07A.nc',
    ssmis_source_path='/data/1C.F16.SSMIS.XCAL2021-V.20230101-S000000-E013000.000001.V07A.nc',
)
```

**Replace example granule names with the actual source.** The full name is
required and must match the source file basename exactly. The source file is
also required because the writer reads `/S4/ScanTime` from the actual granule;
filenames alone do not provide authoritative per-scan timestamps.

The full original writer expects `TbS1`, `TbS2`, `TbS3`, `TbS4` arrays already
mapped to the S4 grid (channel axis last) or separate 2D channel arrays with
original names. It also expects all original `ERA5_*` input fields; `ERA5_msr`
and `ERA5_mtpr` are included when available. All arrays share the same S4
spatial dimensions `(npix,nscan)`. The writer does not perform collocation.

Install with `python -m pip install -e '.[io]'`.

This is a DAAC-**oriented** schema copied from your research pipeline, **not**
a claim that NASA has formally accepted or certified the files. Verify the
archive's current requirements and validate representative granules.
