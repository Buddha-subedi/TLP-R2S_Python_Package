"""2D swath input validation. No files are required for inference."""
import numpy as np
import pandas as pd
from .features import FEATURES

def feature_frame(data, feature_names=None):
    """Return (DataFrame, (height,width)) for dict of 2D features or (30,H,W) array.

    All 30 features must be present in the exact training order. Month/Day may
    be scalar, while other features must be 2D arrays of identical shape.
    """
    if isinstance(data, np.ndarray):
        a=np.asarray(data)
        if a.ndim != 3 or a.shape[0] != len(FEATURES):
            raise ValueError(f'Stacked input must have shape ({len(FEATURES)}, H, W)')
        if feature_names is not None and list(feature_names) != FEATURES:
            raise ValueError('feature_names must match the trained 30-feature order')
        data={name:a[i] for i,name in enumerate(FEATURES)}
    if not isinstance(data,dict):
        raise TypeError('Input must be dict[str, 2D array] or ndarray (30,H,W)')
    missing=[k for k in FEATURES if k not in data]
    if missing:raise ValueError(f'Missing model features: {missing}')
    extras=set(data)-set(FEATURES)
    if extras:raise ValueError(f'Unexpected model features: {sorted(extras)}')
    shape=None
    for k in FEATURES:
        a=np.asarray(data[k])
        if a.ndim==2:
            if shape is None:shape=a.shape
            elif shape!=a.shape:raise ValueError(f'{k} shape {a.shape} != {shape}')
        elif not (k in ('Month','Day') and a.ndim==0):
            raise ValueError(f'{k} must be a 2D array'+(' or scalar' if k in ('Month','Day') else ''))
    if shape is None or min(shape)<1:raise ValueError('No nonempty 2D features')
    columns={}
    for k in FEATURES:
        a=np.asarray(data[k],dtype=np.float32)
        if a.ndim==0:a=np.full(shape,a.item(),dtype=np.float32)
        if k in FEATURES[:11]:a=np.where(a<0,np.nan,a)
        columns[k]=a.ravel(order='C')
    return pd.DataFrame(columns)[FEATURES],shape

def save_swath(path,latitude,longitude,phase,rain,snow):
    """Save a complete retrieved swath as portable NPZ."""
    lat=np.asarray(latitude);lon=np.asarray(longitude)
    if lat.shape!=lon.shape or any(np.asarray(v).shape!=lat.shape for v in (phase,rain,snow)):
        raise ValueError('Latitude, longitude, phase, rain and snow must have identical 2D shape')
    np.savez_compressed(path,latitude=lat,longitude=lon,phase=phase,rain=rain,snow=snow)

def load_swath(path):
    with np.load(path,allow_pickle=False) as z:
        required=('latitude','longitude','phase','rain','snow')
        if any(k not in z for k in required):raise ValueError(f'NPZ must contain {required}')
        d={k:z[k] for k in required}
    if d['latitude'].ndim!=2 or any(v.shape!=d['latitude'].shape for v in d.values()):
        raise ValueError('All swath variables must be matching 2D arrays')
    return d
