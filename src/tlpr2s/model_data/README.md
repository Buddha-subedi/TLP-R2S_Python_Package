# Original pretrained artifacts (not supplied in the uploaded archive)

Copy the following files here if you wish to bundle your trained models in the wheel:

- `xgb_tl_classifier_final.json`
- `xgb_tl_rain_rate_final.json`
- `xgb_tl_snow_rate_final_1.json` (or `xgb_tl_snow_rate_final.json`)
- `cdf_match_for_rain_ssmis.npz`
- `cdf_match_for_snow_ssmis.npz`

**Do not train new substitute models or invent calibration files.** Only redistribute files if you have permission. Alternatively, leave this folder empty and pass `model_dir` and `cdf_dir` to `PrecipitationRetriever`.
