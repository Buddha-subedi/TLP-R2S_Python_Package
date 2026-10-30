#!/usr/bin/env python3
"""
Convert paired TLP-R2S prediction and X_ERA5 MATLAB files to NetCDF-4.

Revised product organization
----------------------------
1. SSMIS S4 is the single product grid.
2. The entire NetCDF file contains only two dimensions: npix and nscan.
3. Every two-dimensional variable is stored as (npix, nscan).
4. S1-S4 brightness temperatures are assumed to have already been mapped to S4.
5. Brightness-temperature variable names do not use inserted "p" characters.
6. TLP-R2S contains Phase and SurfacePrecip only; separate rain and snow
   output variables are not written.
7. Reference/ERA5 contains only MSR and MTPR.

The script assumes supporting metadata dictionaries or fields not shown in the
original excerpt are already available where applicable, but it also provides
safe defaults so this file can run independently.
"""

from __future__ import annotations

import glob
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import scipy.io



# =============================================================================
# USER SETTINGS
# =============================================================================

PREDICTION_DIR = Path(r"E:\DAAC\SSMIS\2023")
X_ERA5_DIR = Path(r"E:\DAAC\SSMIS\2023")
SSMIS_DIR = Path(r"E:\DAAC\SSMIS\SSMIS_pps")
OUTPUT_DIR = Path(r"E:\DAAC\SSMIS\output")

PREDICTION_PATTERN = "GPROF_Pred_X_ERA5_*.mat"
OUTPUT_PREFIX = "TLP-R2S_V01_"

FILL_VALUE = np.float32(-9999.900390625)
PHASE_FILL_VALUE = np.int8(-128)
COMPRESSION_LEVEL = 4


# Each MATLAB TbS* field is assumed to have already been mapped to the S4 grid.
# The NetCDF variable names deliberately avoid decimal-point substitutions such
# as 19p35V. They use compact, readable channel identifiers instead.
SSMIS_CHANNELS = {
    "S1": [
        ("19.35V", "19.35V"),
        ("19.35H", "19.35H"),
        ("22.235V", "22.235V"),
    ],
    "S2": [
        ("37.0V", "37.0V"),
        ("37.0H", "37.0H"),
    ],

    "S3": [
        ("150H", "150H"),
        ("183.31±1H", "183.31±1H"),
        ("183.31±3H", "183.31±3H"),
        ("183.31±6.6H", "183.31±6.6H"),
    ],
    "S4": [
        ("91.655V", "91.655V"),
        ("91.655H", "91.655H"),
    ],
}


# Environmental predictors stored under Input/ERA5. MSR and MTPR are excluded
# here because they are written only under Reference/ERA5.
ERA5_MAT_FIELDS = {
    "tciw": "ERA5_IWP",
    "tclw": "ERA5_LWP",
    "tcwv": "ERA5_WVP",
    "t2m": "ERA5_t2m",
    "cape": "ERA5_CAPE",
    "u10": "ERA5_u10",
    "v10": "ERA5_v10",
    "cin": "ERA5_cin",
    "lsm": "ERA5_lsm",
    "siconc": "ERA5_siconc",
    "skt": "ERA5_skt",
    "asn": "ERA5_asn",
    "rsn": "ERA5_rsn",
    "sd": "ERA5_sd",
    "tcslw": "ERA5_tcslw",
    "tcw": "ERA5_tcw",
    "swvl1": "ERA5_swvl1",
}

ERA5_UNITS = {
    "tciw": "kg m-2",
    "tclw": "kg m-2",
    "tcwv": "kg m-2",
    "t2m": "K",
    "cape": "J kg-1",
    "u10": "m s-1",
    "v10": "m s-1",
    "cin": "J kg-1",
    "lsm": "1",
    "siconc": "1",
    "skt": "K",
    "asn": "1",
    "rsn": "kg m-3",
    "sd": "m",
    "tcslw": "kg m-2",
    "tcw": "kg m-2",
    "swvl1": "m3 m-3",
}



# Only these two variables are written under Reference/ERA5.
ERA5_REFERENCE_MAT_FIELDS = {
    "msr": "ERA5_msr",
    "mtpr": "ERA5_mtpr",
}

ERA5_REFERENCE_UNITS = {
    "msr": "kg m-2 s-1",
    "mtpr": "kg m-2 s-1",
}



# Metadata dictionaries may be populated from the original metadata reference.
# Use descriptive fallbacks where metadata is not supplied.
era5_long_name = {k:k for k in ERA5_MAT_FIELDS}
era5_input_comments = {k:f"ERA5 {k} collocated to SSMIS S4" for k in ERA5_MAT_FIELDS}
era5_standard_name = {}
era5_r_long_name = {"msr":"ERA5 mean snowfall rate", "mtpr":"ERA5 mean total precipitation rate"}
era5_r_standard_name = {"msr":"snowfall_flux", "mtpr":"precipitation_flux"}
era5_r_comment = {k:f"ERA5 {k} collocated to SSMIS S4" for k in ERA5_REFERENCE_MAT_FIELDS}

def validate_preloaded_metadata() -> None:
    """Confirm that the metadata dictionaries from the previous code cell exist."""
    required_names = (
        "era5_input_comments",
        "era5_standard_name",
        "era5_long_name",
        "era5_r_comment",
        "era5_r_long_name",
        "era5_r_standard_name",
    )
    missing = [name for name in required_names if name not in globals()]
    if missing:
        raise NameError(
            "Run the previous ERA5 metadata code before this conversion code. "
            f"Missing variable(s): {', '.join(missing)}"
        )


# =============================================================================
# MATLAB HELPERS
# =============================================================================


def load_mat(path: Path) -> dict[str, Any]:
    """Load a MATLAB file and report unsupported MATLAB v7.3/HDF5 files."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{path.name} is probably MATLAB v7.3/HDF5. "
            "Save it as a pre-v7.3 MAT file or add an h5py loader."
        ) from exc


def unwrap(value: Any) -> Any:
    """Remove scalar object-array wrappers produced by scipy.io.loadmat."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def get_field(container: Any, names: Iterable[str], required: bool = True) -> Any:
    """Return the first matching key or MATLAB structure field."""
    candidates = list(names)

    if isinstance(container, np.void) and container.dtype.names:
        fields = {k.lower(): k for k in container.dtype.names}
        for name in candidates:
            if name.lower() in fields:
                return unwrap(container[fields[name.lower()]])

    if isinstance(container, dict):
        keys = {
            str(key).lower(): key
            for key in container
            if not str(key).startswith("__")
        }
        for name in candidates:
            if name.lower() in keys:
                return unwrap(container[keys[name.lower()]])

    field_names = getattr(container, "_fieldnames", None)
    if field_names:
        fields = {str(name).lower(): name for name in field_names}
        for name in candidates:
            if name.lower() in fields:
                return unwrap(getattr(container, fields[name.lower()]))

    if required:
        raise KeyError(f"None of these fields were found: {candidates}")
    return None


def prediction_container(mat: dict[str, Any]) -> Any:
    """Use nested X prediction structure when present."""
    nested = get_field(mat, ["X"], required=False)
    return nested if nested is not None else mat


def as_2d(array: Any, name: str) -> np.ndarray:
    arr = np.asarray(unwrap(array))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeezing; got {arr.shape}")
    return arr


def orient_2d(array: Any, npix: int, nscan: int, name: str) -> np.ndarray:
    """Orient a field as (npix, nscan) on the S4 product grid."""
    arr = as_2d(array, name)
    if arr.shape == (npix, nscan):
        return arr
    if arr.shape == (nscan, npix):
        return arr.T
    raise ValueError(
        f"{name} has shape {arr.shape}; expected {(npix, nscan)} "
        f"or {(nscan, npix)} on the S4 grid."
    )


def orient_from_nscan(array: Any, nscan: int, name: str) -> np.ndarray:
    """Infer npix from a 2-D field and orient it as (npix, nscan)."""
    arr = as_2d(array, name)
    if arr.shape[1] == nscan:
        return arr
    if arr.shape[0] == nscan:
        return arr.T
    raise ValueError(
        f"{name} has shape {arr.shape}; neither axis matches nscan={nscan}."
    )


def orient_tb(
    array: Any,
    npix: int,
    nscan: int,
    nchannels: int,
    name: str,
) -> np.ndarray:
    """Orient mapped brightness temperatures as (npix, nscan, channel)."""
    arr = np.asarray(unwrap(array))
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3-D after squeezing; got {arr.shape}")

    channel_axes = [axis for axis, size in enumerate(arr.shape) if size == nchannels]
    if not channel_axes:
        raise ValueError(
            f"Could not identify the {nchannels}-channel axis in {name}: {arr.shape}"
        )

    # Prefer the last matching axis, consistent with common MATLAB storage.
    arr = np.moveaxis(arr, channel_axes[-1], -1)

    if arr.shape[:2] == (npix, nscan):
        return arr
    if arr.shape[:2] == (nscan, npix):
        return np.transpose(arr, (1, 0, 2))

    raise ValueError(
        f"{name} has spatial shape {arr.shape[:2]}; expected {(npix, nscan)} "
        f"or {(nscan, npix)} because all channels must already be mapped to S4."
    )


def safe_data(array: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert dtype and replace non-finite values with the NetCDF fill value."""
    arr = np.asarray(array, dtype=dtype)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isfinite(arr), arr, FILL_VALUE).astype(dtype, copy=False)
    return arr


def read_surface_precip(pred_data: Any, npix: int, nscan: int) -> np.ndarray:
    """
    Read total TLP-R2S precipitation.

    A directly supplied total-precipitation field is preferred. For compatibility
    with existing prediction files, rain and snow may be read internally and
    summed, but separate RainRate and SnowRate variables are never written.
    """
    direct = get_field(
        pred_data,
        [
            "SurfacePrecip",
            "surface_precip",
            "surfacePrecip",
            "precip",
            "Precip",
            "precipitation",
            "total_precip",
            "TotalPrecip",
            "rate",
            "Rate",
        ],
        required=False,
    )
    if direct is not None:
        return orient_2d(direct, npix, nscan, "SurfacePrecip")

    rain_raw = get_field(pred_data, ["rain", "Rain", "rain_30"], required=False)
    snow_raw = get_field(pred_data, ["snow", "Snow", "snow_30"], required=False)
    if rain_raw is None or snow_raw is None:
        raise KeyError(
            "No total precipitation field was found, and rain/snow fields were "
            "not both available for constructing SurfacePrecip."
        )

    rain = orient_2d(rain_raw, npix, nscan, "rain")
    snow = orient_2d(snow_raw, npix, nscan, "snow")
    return rain + snow


# =============================================================================
# FILE MATCHING AND TIME
# =============================================================================

PAIR_REGEX = re.compile(
    r"^GPROF_Pred_X_ERA5_(?P<date>\d{8})_(?P<orbit>\d+)\.mat$",
    re.IGNORECASE,
)


def parse_prediction_name(path: Path) -> tuple[str, str]:
    match = PAIR_REGEX.match(path.name)
    if not match:
        raise ValueError(
            "Prediction filename must follow "
            "GPROF_Pred_X_ERA5_YYYYMMDD_ORBIT.mat"
        )
    return match.group("date"), match.group("orbit")


def find_ssmis_source(date_string: str, orbit: str) -> Path | None:
    """Find the source SSMIS file used for scan time and provenance."""
    patterns = [
        f"*{date_string}*{orbit}*",
        f"*{orbit}*{date_string}*",
        f"*{orbit}*",
    ]
    for pattern in patterns:
        matches = [
            Path(item)
            for item in glob.glob(str(SSMIS_DIR / pattern))
            if Path(item).is_file()
        ]
        if matches:
            return sorted(matches)[0]
    return None


def build_time_from_ssmis(
    ssmis_path: Path,
    nscan: int,
) -> tuple[np.ndarray, str, str]:
    """Read S4 scan times directly from the source SSMIS HDF5 file."""
    scan_time_group = "/S4/ScanTime"

    # NetCDF-4 is HDF5-backed. Read the native /S4/ScanTime group from the
    # original .nc file; do not infer scan times from its filename.
    with h5py.File(ssmis_path, "r") as ssmis:
        required_paths = {
            "Year": f"{scan_time_group}/Year",
            "Month": f"{scan_time_group}/Month",
            "Day": f"{scan_time_group}/DayOfMonth",
            "Hour": f"{scan_time_group}/Hour",
            "Minute": f"{scan_time_group}/Minute",
            "Second": f"{scan_time_group}/Second",
        }
        missing = [v for v in required_paths.values() if v not in ssmis]
        if missing:
            raise KeyError(
                f"Source NetCDF {ssmis_path.name} is missing S4 scan-time "
                f"datasets: {missing}. Inspect its actual group layout; "
                "do not fabricate timestamps."
            )
        year = np.ravel(np.asarray(ssmis[required_paths["Year"]][:]))
        month = np.ravel(np.asarray(ssmis[required_paths["Month"]][:]))
        day = np.ravel(np.asarray(ssmis[required_paths["Day"]][:]))
        hour = np.ravel(np.asarray(ssmis[required_paths["Hour"]][:]))
        minute = np.ravel(np.asarray(ssmis[required_paths["Minute"]][:]))
        second = np.ravel(np.asarray(ssmis[required_paths["Second"]][:]))

    for name, values in {
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "Minute": minute,
        "Second": second,
    }.items():
        if values.size != nscan:
            raise ValueError(
                f"SSMIS {name} has length {values.size}; expected nscan={nscan}."
            )

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    output = np.full(nscan, np.float64(FILL_VALUE), dtype=np.float64)
    valid_datetimes: list[datetime] = []

    for index, values in enumerate(zip(year, month, day, hour, minute, second)):
        try:
            y, m, d, h, mi, s = (int(value) for value in values)
            dt = datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)
            output[index] = (dt - epoch).total_seconds() / 60.0
            valid_datetimes.append(dt)
        except (TypeError, ValueError, OverflowError):
            continue

    start = min(valid_datetimes).strftime("%Y-%m-%dT%H:%M:%SZ") if valid_datetimes else ""
    end = max(valid_datetimes).strftime("%Y-%m-%dT%H:%M:%SZ") if valid_datetimes else ""
    return output, start, end


# =============================================================================
# NETCDF HELPERS
# =============================================================================


def chunk_2d(npix: int, nscan: int) -> tuple[int, int]:
    return min(180, npix), min(256, nscan)


def create_float_variable(
    group: Any,
    name: str,
    dimensions: tuple[str, ...],
    chunks: tuple[int, ...],
):
    return group.createVariable(
        name,
        "f4",
        dimensions,
        fill_value=FILL_VALUE,
        chunksizes=chunks,
        zlib=True,
        complevel=COMPRESSION_LEVEL,
        shuffle=True,
    )


def add_coordinates(variable: Any) -> None:
    variable.coordinates = "Latitude Longitude Time"
    variable.grid_mapping = "mapping"


def build_geospatial_bounds(
    latitude_s4: np.ndarray,
    longitude_s4: np.ndarray,
    points_per_edge: int = 20,
) -> str:
    """
    Build a simplified polygon for the full SSMIS orbital swath using the two
    outer S4 cross-track edges. The output follows the requested
    latitude-longitude coordinate order and uses fewer than about 50 points.
    """
    lat_edge_1 = latitude_s4[0, :]
    lon_edge_1 = longitude_s4[0, :]

    lat_edge_2 = latitude_s4[-1, :]
    lon_edge_2 = longitude_s4[-1, :]

    valid_1 = (
        np.isfinite(lat_edge_1)
        & np.isfinite(lon_edge_1)
        & (lat_edge_1 >= -90.0)
        & (lat_edge_1 <= 90.0)
        & (lon_edge_1 >= -180.0)
        & (lon_edge_1 <= 180.0)
    )

    valid_2 = (
        np.isfinite(lat_edge_2)
        & np.isfinite(lon_edge_2)
        & (lat_edge_2 >= -90.0)
        & (lat_edge_2 <= 90.0)
        & (lon_edge_2 >= -180.0)
        & (lon_edge_2 <= 180.0)
    )

    lat_edge_1 = lat_edge_1[valid_1]
    lon_edge_1 = lon_edge_1[valid_1]
    lat_edge_2 = lat_edge_2[valid_2]
    lon_edge_2 = lon_edge_2[valid_2]

    if len(lat_edge_1) == 0 or len(lat_edge_2) == 0:
        raise ValueError("Could not construct geospatial bounds from S4 coordinates.")

    idx_1 = np.linspace(
        0,
        len(lat_edge_1) - 1,
        min(points_per_edge, len(lat_edge_1)),
        dtype=int,
    )

    idx_2 = np.linspace(
        0,
        len(lat_edge_2) - 1,
        min(points_per_edge, len(lat_edge_2)),
        dtype=int,
    )

    polygon_points = []

    for i in idx_1:
        polygon_points.append(
            (float(lat_edge_1[i]), float(lon_edge_1[i]))
        )

    for i in idx_2[::-1]:
        polygon_points.append(
            (float(lat_edge_2[i]), float(lon_edge_2[i]))
        )

    polygon_points.append(polygon_points[0])

    return "POLYGON ((" + ", ".join(
        f"{lat:.6f} {lon:.6f}"
        for lat, lon in polygon_points
    ) + "))"


# =============================================================================
# MAIN CONVERSION
# =============================================================================


def netcdf(path, X, phase, rain, snow, source, source_ssmis=None, model_version="TLP-R2S_V01") -> None:
    """Write reference-schema NetCDF from in-memory inference and real SSMIS S4 times."""
    if source_ssmis is None:
        raise FileNotFoundError("Original SSMIS granule is required for exact source timestamps")
    source_ssmis = Path(source_ssmis)
    source_name = source_ssmis.name
    from .ssmis_source import SOURCE_RE
    source_match = SOURCE_RE.fullmatch(source_name)
    if source_match is None:
        raise ValueError(f"Unexpected original SSMIS granule filename: {source_name}")
    orbit = source_match.group('orbit')
    source_version = source_name.rsplit('.', 2)[-2]
    output_path = Path(path)
    x_data = X
    pred_data = {"Phase": phase, "SurfacePrecip": np.asarray(rain) + np.asarray(snow)}
    # -------------------------------------------------------------------------
    # S4 is always the product grid. No S3 fallback is used.
    # -------------------------------------------------------------------------
    latitude_s4_raw = as_2d(get_field(x_data, ["LatS4"]), "LatS4")
    product_npix, nscan = latitude_s4_raw.shape
    latitude_s4 = latitude_s4_raw
    longitude_s4 = orient_2d(
        get_field(x_data, ["LonS4"]), product_npix, nscan, "LonS4"
    )

    if latitude_s4.shape != longitude_s4.shape:
        raise ValueError(
            f"LatS4 shape {latitude_s4.shape} does not match "
            f"LonS4 shape {longitude_s4.shape}."
        )

    print(f"S4 product grid: npix={product_npix}, nscan={nscan}")

    phase = orient_2d(
        get_field(pred_data, ["phase", "Phase", "phase_30"]),
        product_npix,
        nscan,
        "phase",
    )
    surface_precip = read_surface_precip(pred_data, product_npix, nscan)

    # Validate all mapped SSMIS channels before opening the output file.
    mapped_tb: dict[str, tuple[np.ndarray, list[tuple[str, str]]]] = {}
    for swath_name, channels in SSMIS_CHANNELS.items():
        tb = orient_tb(
            get_field(x_data, [f"Tb{swath_name}"]),
            product_npix,
            nscan,
            len(channels),
            f"Tb{swath_name}",
        )
        mapped_tb[swath_name] = (tb, channels)

    time_values, time_start, time_end = build_time_from_ssmis(
        source_ssmis,
        nscan,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("DAAC NetCDF output requires: pip install tlpr2s[io]") from exc
    with Dataset(output_path, "w", format="NETCDF4") as nc:
        # ---------------------------------------------------------------------
        # Global attributes
        # ---------------------------------------------------------------------
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nc.date_created = now
        nc.date_modified = now
        nc.time_coverage_start = time_start
        nc.time_coverage_end = time_end
        valid_lat = latitude_s4[np.isfinite(latitude_s4)]
        valid_lon = longitude_s4[np.isfinite(longitude_s4)]

        nc.geospatial_lat_min = float(np.min(valid_lat))
        nc.geospatial_lat_max = float(np.max(valid_lat))
        nc.geospatial_lon_min = float(np.min(valid_lon))
        nc.geospatial_lon_max = float(np.max(valid_lon))

        nc.geospatial_bounds = build_geospatial_bounds(
            latitude_s4,
            longitude_s4,
            points_per_edge=20,
        )
        nc.geospatial_bounds_crs = "EPSG:4326"

        nc.title = (
            "MEaSUREs Precipitation Phase and Rate from Passive Microwave Observations and Earth System Models using Transfer Learning, Version 1"
        )
        nc.summary = (
            "NASA Making Earth System Data Records for Use in Research Environments (MEaSUREs) "
            "Precipitation Phase and Rate is obtained from a novel machine learning framework "
            "called Transfer Learning of global Precipitation from Reanalysis data to Satellite "
            "observations (TLP-R2S) that leverages incremental training of an ensemble of boosted "
            "decision trees to map low-level precipitation characteristics from Earth System "
            "Models (ESMs) to high-level features captured by satellite active precipitation "
            "observations."
        )
        nc.source = "TLP-R2S model v01"
        nc.input_ssmis_file = source_name
        nc.source_mat_file = str(source)
        nc.model_version = model_version

        nc.input_granule_number = str(orbit)
        nc.orbit_number = str(orbit)
        nc.orbit_start_time = time_start
        nc.orbit_end_time = time_end
        nc.input_platform = "DMSP-F16"
        nc.input_instrument = "SSMIS"
        nc.input_calibration_version = "XCAL2021-V"
        nc.input_product = (
            "GPM SSMIS on F16 Common Calibrated Brightness Temperatures "
            "L1C 1.5 hours 12 km"
        )
        nc.input_product_version = source_version
        nc.Conventions = "CF-1.8, ACDD-1.3"
        nc.keywords = "Projects > M-O > MEaSUREs, EARTH SCIENCE > ATMOSPHERE > PRECIPITATION > SURFACE PRECIPITATION"

        nc.references = (
            "Subedi, B., Garshasbi, M., Ebtehaj, A., Milani, L., Utsumi, N., & Huffman, G. J. (2026). "
            "An Incremental Learning Framework Linking Reanalysis and Radar Data for Passive Microwave Precipitation Retrievals. "
            "IEEE Transactions on Geoscience and Remote Sensing; "
        
            "Garshasbi, M., Subedi, B., Ebtehaj, A., Milani, L., Turk, F. J., & Huffman, G. J. (2026). "
            "Incremental Learning for Passive Microwave Precipitation Retrievals Using Advanced Technology Microwave Sounder. "
            "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing; "
        
            "Randel, D. L., Kummerow, C. D., & Ringerud, S. (2020). "
            "The Goddard Profiling (GPROF) precipitation retrieval algorithm. "
            "In Satellite Precipitation Measurement (Vol. 1, pp. 141–152). "
            "Springer International Publishing; "
        
            "Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., "
            "Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., "
            "Schepers, D., Simmons, A., Soci, C., Dee, D., & Thépaut, J.-N. (2023). "
            "ERA5 hourly data on single levels from 1940 to present. "
            "Copernicus Climate Change Service (C3S) Climate Data Store. "
            "https://doi.org/10.24381/cds.adbb2d47; "
        
            "U.S. Geological Survey. (1996). "
            "GTOPO30 Global 30 Arc-Second Elevation Model; "
        
            "Hou, A. Y., Kakar, R. K., Neeck, S., Azarbarzin, A. A., "
            "Kummerow, C. D., Kojima, M., et al. (2014). "
            "The Global Precipitation Measurement Mission. "
            "Bulletin of the American Meteorological Society, 95(5), 701–722."
        )
   

        nc.product_grid = "SSMIS S4"

        # The only dimensions defined anywhere in this NetCDF file.
        nc.createDimension("nscan", nscan)
        nc.createDimension("npix", product_npix)

        chunks = chunk_2d(product_npix, nscan)

        # ---------------------------------------------------------------------
        # Root coordinates on S4
        # ---------------------------------------------------------------------
        latitude = create_float_variable(
            nc, "Latitude", ("npix", "nscan"), chunks
        )
        longitude = create_float_variable(
            nc, "Longitude", ("npix", "nscan"), chunks
        )
        latitude[:] = safe_data(latitude_s4)
        longitude[:] = safe_data(longitude_s4)

        latitude.units = "degrees_north"
        latitude.standard_name = "latitude"
        latitude.long_name = "latitude of SSMIS S4 pixel center"
        longitude.units = "degrees_east"
        longitude.standard_name = "longitude"
        longitude.long_name = "longitude of SSMIS S4 pixel center"

        time_var = nc.createVariable(
            "Time",
            "f8",
            ("nscan",),
            zlib=True,
            complevel=COMPRESSION_LEVEL,
            shuffle=True,
        )
        time_var[:] = time_values
        time_var.units = "minutes since 1970-01-01 00:00:00 UTC"

        time_var.long_name = "Scan time"
        time_var.comment = "Along-track scan time"
        time_var.standard_name = "time"

        mapping = nc.createVariable('mapping', str, ())
        mapping.grid_mapping_name = "latitude_longitude"
        mapping.longitude_of_prime_meridian = 0.0
        mapping.semi_major_axis = 6378137.0
        mapping.inverse_flattening = 298.257223563
        mapping.srid = "urn:ogc:def:crs:EPSG::4326"
        mapping.crs_wkt = "GEOGCRS[\"WGS 84\",ENSEMBLE[\"World Geodetic System 1984 ensemble\",MEMBER[\"World Geodetic System 1984 (Transit)\"],MEMBER[\"World Geodetic System 1984 (G730)\"],MEMBER[\"World Geodetic System 1984 (G873)\"],MEMBER[\"World Geodetic System 1984 (G1150)\"],MEMBER[\"World Geodetic System 1984 (G1674)\"],MEMBER[\"World Geodetic System 1984 (G1762)\"],MEMBER[\"World Geodetic System 1984 (G2139)\"],ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]],ENSEMBLEACCURACY[2.0]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]],CS[ellipsoidal,2],AXIS[\"geodetic latitude (Lat)\",north,ORDER[1],ANGLEUNIT[\"degree\",0.0174532925199433]],AXIS[\"geodetic longitude (Lon)\",east,ORDER[2],ANGLEUNIT[\"degree\",0.0174532925199433]],USAGE[SCOPE[\"Horizontal component of 3D system.\"],AREA[\"World.\"],BBOX[-90,-180,90,180]],ID[\"EPSG\",4326]]"
        # TLP-R2S: no separate rain or snow variables
        # ---------------------------------------------------------------------
        tlpr2s = nc.createGroup("TLP-R2S")
        tlpr2s.GroupDescription = "Surface precipitation phase and rate obtained from TLP-R2S"
        tlpr2s.long_name = (
            "Transfer Learning of global Precipitation from Reanalysis data "
            "to Satellite observations"
        )
        tlpr2s.comment = (
            "TLP-R2S surface precipitation phase and total precipitation rate "
            "on the SSMIS S4 grid."
        )

        phase_var = tlpr2s.createVariable(
            "Phase",
            "i1",
            ("npix", "nscan"),
            fill_value=PHASE_FILL_VALUE,
            chunksizes=chunks,
            zlib=True,
            complevel=COMPRESSION_LEVEL,
            shuffle=True,
        )
        phase_array = np.asarray(phase, dtype=np.float64)
        finite_phase = np.isfinite(phase_array)
        phase_clean = np.full(phase_array.shape, PHASE_FILL_VALUE, dtype=np.int16)
        phase_clean[finite_phase] = np.rint(phase_array[finite_phase]).astype(np.int16)
        valid_phase = finite_phase & np.isin(phase_clean, [0, 1, 2])
        phase_var[:] = np.where(
            valid_phase, phase_clean, PHASE_FILL_VALUE
        ).astype(np.int8)
        phase_var.units = "1"
        phase_var.standard_name = "predominant_precipitation_type_at_surface"
        phase_var.long_name = (
            "surface precipitation phase (0=clear, 1=rain, 2=snow)"
        )
        phase_var.comment = 'Phase detected by TLP-R2S algorithm'
        phase_var.flag_values = np.asarray([0, 1, 2], dtype=np.int8)
        phase_var.flag_meanings = "clear rain snow"
        add_coordinates(phase_var)

        surface_var = create_float_variable(
            tlpr2s, "SurfacePrecip", ("npix", "nscan"), chunks
        )
        surface_var[:] = safe_data(surface_precip)
        surface_var.units = "mm h-1"
        surface_var.standard_name = "lwe_precipitation_rate"
        surface_var.long_name = "TLP-R2S total surface precipitation rate"
        surface_var.comment = (
            "Total precipitation output. Separate rain and snow variables are "
            "intentionally not included in the product."
        )
        add_coordinates(surface_var)

        # ---------------------------------------------------------------------
        # Input data, all mapped to S4 and using only npix/nscan
        # ---------------------------------------------------------------------
        input_group = nc.createGroup("Input")
        input_group.comment = "Inputs used by the TLP-R2S retrieval."

        ssmis_group = input_group.createGroup("SSMIS")
        ssmis_group.long_name = (
            "Special Sensor Microwave Imager/Sounder"
        )
        ssmis_group.comment = (
            f"Brightness temperatures read from {source_name}. All S1-S4 "
            "channels are stored on the common S4 grid. "
        )

        for swath_name, (tb, channels) in mapped_tb.items():
            for channel_index, (variable_name, channel_label) in enumerate(channels):
                variable = create_float_variable(
                    ssmis_group,
                    variable_name,
                    ("npix", "nscan"),
                    chunks,
                )
                variable[:] = safe_data(tb[:, :, channel_index])
                variable.units = "K"
                variable.standard_name = "brightness_temperature"
                variable.long_name = (
                    f"SSMIS F16 {channel_label} GHz brightness temperature"
                )
                variable.source_swath = swath_name
                variable.comment = (
                    f"The {swath_name} channel was mapped to the SSMIS S4 grid "
                    "before writing."
                )
                add_coordinates(variable)

        era5_group = input_group.createGroup("ERA5")
        era5_group.long_name = "European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis v5 (ERA5)"
        era5_group.comment = (
            "ERA5 is the fifth generation ECMWF reanalysis for the global climate and weather for the past 8 decades. Data is available from 1940 onwards. ERA5 replaces the ERA-Interim reanalysis."
        )
        era5_group.references = (
            "Hersbach et al. (2023), ERA5 hourly data on single levels from "
            "1940 to present, DOI: 10.24381/cds.adbb2d47."
        )

        for output_name, mat_name in ERA5_MAT_FIELDS.items():
            raw = get_field(x_data, [mat_name], required=False)
            if raw is None:
                print(f"  ERA5 input field missing; skipped: {mat_name}")
                continue

            data = orient_2d(raw, product_npix, nscan, mat_name)
            variable = create_float_variable(
                era5_group,
                output_name,
                ("npix", "nscan"),
                chunks,
            )
            variable[:] = safe_data(data)
            variable.units = ERA5_UNITS[output_name]
            variable.long_name = era5_long_name.get(output_name, output_name)
            variable.comment = era5_input_comments.get(
                output_name,
                (
                    f"ERA5 variable read from MATLAB field {mat_name} and "
                    "collocated to the SSMIS S4 grid."
                ),
            )
            standard_name = era5_standard_name.get(output_name)
            if standard_name:
                variable.standard_name = standard_name
            add_coordinates(variable)

        # ---------------------------------------------------------------------
        # Reference/ERA5: exactly MSR and MTPR
        # ---------------------------------------------------------------------
        reference_group = nc.createGroup("Reference")
        reference_group.comment = (
            "Reference precipitation variables used for comparison with TLP-R2S."
        )

        era5_reference = reference_group.createGroup("ERA5")
        era5_reference.long_name = "ECMWF Reanalysis v5 precipitation reference"
        era5_reference.comment = (
            "This group contains only MSR and MTPR on the SSMIS S4 grid."
        )
        era5_reference.references = (
            "Hersbach et al. (2023), ERA5 hourly data on single levels from "
            "1940 to present, DOI: 10.24381/cds.adbb2d47."
        )

        for output_name, mat_name in ERA5_REFERENCE_MAT_FIELDS.items():
            raw = get_field(x_data, [mat_name], required=False)
            if raw is None:
                print(f"  ERA5 reference field missing; skipped: {mat_name}")
                continue

            data = orient_2d(raw, product_npix, nscan, mat_name)
            variable = create_float_variable(
                era5_reference,
                output_name,
                ("npix", "nscan"),
                chunks,
            )
            variable[:] = safe_data(data)
            variable.units = ERA5_REFERENCE_UNITS[output_name]
            variable.long_name = era5_r_long_name[output_name]
            variable.standard_name = era5_r_standard_name[output_name]
            variable.comment = era5_r_comment[output_name]
            add_coordinates(variable)

    print(f"Created: {output_path}")



def plot(path, X, rain, snow, orbit_number=None):
    """One global map with rain and snow overlaid, two original rate colorbars."""
    from precipitation_map import save_combined_map
    import re
    if orbit_number is None:
        match = re.search(r'(\d{6})$', Path(path).stem)
        orbit_number = match.group(1) if match else Path(path).stem
    save_combined_map(path, X['LonS4'], X['LatS4'], rain, snow,
                      title=f'TLP-R2S Precipitation Rate - Orbit {orbit_number}')
