# Input contract

`PrecipitationRetriever.predict(data)` accepts either:

1. A `dict[str, np.ndarray]` containing the 30 feature names from `tlpr2s.FEATURES`; all are 2D `(height,width)` except Month/Day may be scalar.
2. An array `(30,height,width)` with channels in exactly `FEATURES` order. Optionally pass `feature_names=FEATURES` for explicit order checking.

## Exact feature order

```python
from tlpr2s import FEATURES
print(FEATURES)
```

Brightness temperatures (11): `19.35v`, `19.35h`, `22.235v`, `37v`, `37h`, `91.665v`, `91.665h`, `150h`, `183-1`, `183-3`, `183-7`.
ERA5/environmental fields (17): `tciw`, `tclw`, `tcwv`, `t2m`, `cape`, `u10`, `v10`, `skt`, `asn`, `rsn`, `cin`, `sd`, `tcslw`, `tcw`, `swvl1`, `lsm`, `siconc`.
Temporal features (2): `Month`, `Day`.

**Training units and interpolation are not established by the supplied archive alone.** Preserve the source pipeline's exact units and preprocessing; do not assume unit conversions. Brightness temperatures less than 0 are set to NaN, matching the original inference code. The 11 TB channels must already be collocated to the S4 target footprint. Latitude and longitude are 2D and used for visualization, not model inference.

The saved swath NPZ keys are `latitude`, `longitude`, `phase`, `rain`, `snow` (matching 2D shapes). Use `np.load(...,allow_pickle=False)` to read.
