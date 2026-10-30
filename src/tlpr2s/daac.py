"""Public interface for full legacy NASA DAAC-oriented SSMIS F16 NetCDF schema.

This writer requires the actual source granule to preserve genuine S4 scan times.
A full source filename is required and checked against the supplied file path.
"""
from pathlib import Path
import numpy as np
from .daac_products import netcdf, SSMIS_CHANNELS, ERA5_MAT_FIELDS, ERA5_REFERENCE_MAT_FIELDS
from .ssmis_source import SOURCE_RE


def _as_2d(value, name, shape=None):
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 2D, got {arr.shape}')
    if shape is not None and arr.shape != shape:
        raise ValueError(f'{name} shape {arr.shape} != {shape}')
    return arr


def DAAC_compatible_nc(output_path, *, result, latitude, longitude,
                       input_arrays, ssmis_full_filename, ssmis_source_path,
                       source_identifier='2D-array retrieval', model_version='TLP-R2S_V01'):
    """Save the complete original TLP-R2S DAAC-oriented NetCDF product.

    Parameters
    ----------
    output_path : str or Path
        Destination .nc file.
    result : RetrievalResult
        Inference output with phase, rain, snow arrays.
    latitude, longitude : 2D array
        SSMIS S4 coordinates, shaped (npix,nscan).
    input_arrays : dict
        2D arrays for ALL original SSMIS brightness-temperature and ERA5 fields.
        Required SSMIS keys: TbS1, TbS2, TbS3, TbS4. Each is a
        channel-stacked 3D array (npix,nscan,nchannels), OR provide individual
        2D channel maps under keys such as '19.35V', '183.31±1H'.
        ERA5 fields are named ERA5_tciw etc. Reference ERA5_msr and
        ERA5_mtpr are optional, as in the original writer.
    ssmis_full_filename : str
        Complete original SSMIS F16 L1C granule filename; validated exactly.
    ssmis_source_path : str or Path
        Actual original SSMIS NetCDF file, used to read genuine S4 scan times.
        Metadata cannot be fully populated from a filename alone.
    source_identifier : str
        Provenance identifier for input 2D arrays.
    model_version : str
        Retrieval model version.

    Returns
    -------
    pathlib.Path
        Written NetCDF path.
    """
    if not isinstance(input_arrays, dict):
        raise TypeError('input_arrays must be a dictionary of mapped S4 fields')
    name=Path(ssmis_full_filename).name
    if name != str(ssmis_full_filename):
        raise ValueError('ssmis_full_filename must be a filename, not a path')
    if SOURCE_RE.fullmatch(name) is None:
        raise ValueError(f'Full SSMIS F16 source filename does not match original pattern: {name}')
    source=Path(ssmis_source_path)
    if source.name != name:
        raise ValueError(f'Source file basename {source.name!r} differs from full filename {name!r}')
    if not source.is_file():
        raise FileNotFoundError(f'SSMIS source file required for scan-time metadata: {source}')
    lat=_as_2d(latitude,'latitude')
    lon=_as_2d(longitude,'longitude',lat.shape)
    phase=_as_2d(result.phase,'phase',lat.shape)
    rain=_as_2d(result.rain,'rain',lat.shape)
    snow=_as_2d(result.snow,'snow',lat.shape)
    x=dict(input_arrays)
    x['LatS4']=lat
    x['LonS4']=lon
    for swath,channels in SSMIS_CHANNELS.items():
        key='Tb'+swath
        if key in x:
            continue
        absent=[ch for ch,_ in channels if ch not in x]
        if absent:
            raise KeyError(f'Missing {key} or mapped 2D channel fields: {absent}')
        x[key]=np.stack([_as_2d(x[ch],ch,lat.shape) for ch,_ in channels],axis=-1)
    missing=[key for key in ERA5_MAT_FIELDS.values() if key not in x]
    if missing:
        raise KeyError(f'Full DAAC product requires original ERA5 input fields: {missing}')
    for key in list(ERA5_MAT_FIELDS.values())+list(ERA5_REFERENCE_MAT_FIELDS.values()):
        if key in x: _as_2d(x[key],key,lat.shape)
    output=Path(output_path)
    netcdf(output, x, phase, rain, snow, source_identifier,
           source_ssmis=source, model_version=model_version)
    return output
