import os, json, tempfile
from pathlib import Path
import numpy as np

FEATURES=['19.35v','19.35h','22.235v','37v','37h','91.665v','91.665h','150h','183-1','183-3','183-7','tciw','tclw','tcwv','t2m','cape','u10','v10','skt','asn','rsn','cin','sd','tcslw','tcw','swvl1','lsm','siconc','Month','Day']
FIELD_MAP={'tciw':'ERA5_IWP','tclw':'ERA5_LWP','tcwv':'ERA5_WVP','t2m':'ERA5_t2m','cape':'ERA5_CAPE','u10':'ERA5_u10','v10':'ERA5_v10','skt':'ERA5_skt','asn':'ERA5_asn','rsn':'ERA5_rsn','cin':'ERA5_cin','sd':'ERA5_sd','tcslw':'ERA5_tcslw','tcw':'ERA5_tcw','swvl1':'ERA5_swvl1','lsm':'ERA5_lsm','siconc':'ERA5_siconc'}
GRID_LAT=np.arange(-90,90.0001,.25)
GRID_LON=np.arange(-180,180.0001,.25)
SHAPE=(len(GRID_LAT),len(GRID_LON)) # exact legacy notebook 721x1441 grid nodes
NAMES=['snow_tl','rain_tl','phase_tl_1','phase_tl_2','phase_era5_1','phase_era5_2','era5_snow','era5_rain']

def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.partial_',suffix='.npz',dir=path.parent);os.close(fd)
    try:
        np.savez_compressed(tmp,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def valid_npz(path, keys):
    try:
        with np.load(path) as d:return all(k in d for k in keys)
    except (OSError,ValueError,EOFError):return False
