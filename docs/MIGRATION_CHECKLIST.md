# Source-to-package parity checklist

Original code retained under `legacy_reference/` (not part of installed package).

1. Confirm the original 30 `FEATURES` ordering matches `tlpr2s.FEATURES` byte-for-byte.
2. Obtain the three original XGBoost model JSONs and two CDF NPZs; record SHA256 checksums.
3. Obtain one real prepared MAT file with `X[0,0]`, and the corresponding SSMIS source granule. Do not publicly redistribute data without authorization.
4. Extract the 11 TB 2D channels using original `TB_FIELDS`, the 17 ERA5 arrays using `FIELD_MAP`, plus Month and Day. Verify all shapes match `LatS4` and flattening order is C.
5. Run the original `Predictor.predict(mat_path)` and `tlpr2s.PrecipitationRetriever.predict(features)` with identical models and calibration.
6. Assert exact phase agreement; compare rain and snow with documented numerical tolerance, checking maximum absolute and relative differences.
7. Test all-clear, all-rain, all-snow, NaN TBs, missing channels, Month/Day scalars, swath boundaries, and memory usage.
8. Validate CDF output against frozen numerical reference for rain and snow.
9. Test swath NPZ save/load and plotted geolocation. Compare maps near dateline and polar regions.
10. Decide whether to retain legacy 721x1441 grid as an optional mode. Current 721x1440 nearest-node aggregation is **not numerically equivalent** to legacy annual statistics.
11. Add an end-to-end CI test using a redistributable tiny fixture with known expected outputs.
12. Document precise calibration units, training data and model limitations, metadata, authors, and citation.
