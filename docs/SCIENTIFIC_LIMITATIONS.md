# Scientific limitations and annual statistics

- Current model is specifically the uploaded SSMIS F16 TLP-R2S model. No generic cross-sensor support is claimed.
- The package does not generate 30 predictors from raw satellite granules. Collocation and auxiliary ERA5 inputs are the caller's responsibility.
- `AnnualAccumulator` bins each valid swath pixel into a regular 0.25-degree cell-center grid and computes mean rain/snow rates using pixel counts.
- Annual mean **does not** imply temporal or spatial representativeness. Orbit overlap, uneven temporal coverage, viewing geometry and sample density can bias means. There is no interpolation across gaps.
- This is not annual rainfall accumulation in mm/year. To compute a physically interpretable accumulation requires time integration, sampling assumptions and validation.
- Grid is 721x1440 at 0.25°. The 1440 grid in the longitude direction is due to the fact that -180° and 180° is same.
- The plotting function uses `pcolormesh` on native 2D swath coordinates. Dateline-crossing and polar geometries may require polygon splitting; inspect maps near ±180 longitude. Cartopy is optional.
- `swath_coverage_count` counts number of supplied swath files with at least one valid sample in each cell; `observation_count` counts individual pixels.
- No uncertainty quantification or error bars are provided.

