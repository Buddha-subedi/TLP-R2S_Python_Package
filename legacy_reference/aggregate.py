"""Sparse orbit contribution, exactly matching notebook KDTree nearest grid-node averaging.
"""
import numpy as np
from scipy.spatial import cKDTree
from common import GRID_LAT,GRID_LON,SHAPE,NAMES,atomic_npz

# Nearest regular grid point = nearest KDTree grid point for this axis-aligned grid;
# ties are rare and should be tested against legacy KDTree on representative files.
def contribution(X,phase,rain,snow,path):
 lat=np.asarray(X['LatS4']).ravel();lon=np.asarray(X['LonS4']).ravel()
 valid=np.isfinite(lat)&np.isfinite(lon)&(lat>=-90)&(lat<=90)&(lon>=-180)&(lon<=180)
 if not np.any(valid):raise ValueError('No valid coordinates')
 # Legacy skip rule; skipped orbits do not contribute to coverage.
 if np.nansum(rain)==0 and np.nansum(snow)==0:
  atomic_npz(path,indices=np.array([],dtype=np.int32),coverage=np.array([],dtype=np.int32),values=np.empty((len(NAMES),0),dtype=np.float64),skipped=np.array([1]))
  return
 rows=np.clip(np.rint((lat[valid]+90)*4).astype(np.int32),0,len(GRID_LAT)-1)
 cols=np.clip(np.rint((lon[valid]+180)*4).astype(np.int32),0,len(GRID_LON)-1)
 ids=rows*len(GRID_LON)+cols
 unique,inverse=np.unique(ids,return_inverse=True)
 msr=np.asarray(X['ERA5_msr'],dtype=np.float32)*3600
 rain_era=(np.asarray(X['ERA5_mtpr'],dtype=np.float32)-np.asarray(X['ERA5_msr'],dtype=np.float32))*3600
 era_snow=np.clip(msr,0,None);era_rain=np.clip(rain_era,0,None)
 flags=np.select([(era_snow<.01)&(era_rain<.01),(era_rain>.01)&(era_snow<.01),(era_snow>.01)&(era_rain<.01)],[0,1,2],default=3)
 arrays=[snow,rain,phase==1,phase==2,flags==1,flags==2,era_snow,era_rain]
 vals=[]
 for arr in arrays:
  v=np.asarray(arr).ravel()[valid].astype(np.float64)
  ok=np.isfinite(v);s=np.bincount(inverse[ok],weights=v[ok],minlength=len(unique))
  n=np.bincount(inverse[ok],minlength=len(unique));vals.append(np.divide(s,n,out=np.full(len(unique),np.nan),where=n>0))
 atomic_npz(path,indices=unique.astype(np.int32),coverage=np.ones(len(unique),dtype=np.int32),values=np.array(vals),skipped=np.array([0]))
