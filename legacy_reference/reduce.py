"""Reduce sparse orbit contributions in bounded memory, retaining notebook semantics.
Annual values are SUM of orbit-cell MEANS; orbit_coverage counts observed orbits.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from common import SHAPE,GRID_LAT,GRID_LON,NAMES,atomic_npz

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
 with open(a.manifest) as f:rows=[r for r in csv.DictReader(f) if int(r['year'])==a.year]
 sums=np.zeros((len(NAMES),SHAPE[0]*SHAPE[1]),dtype=np.float64)
 cover=np.zeros(SHAPE[0]*SHAPE[1],dtype=np.int32)
 missing=[];skipped=0
 for r in rows:
  path=Path(a.out)/str(a.year)/'contributions'/(Path(r['path']).stem+'.npz')
  if not path.exists():missing.append(str(path));continue
  try:
   with np.load(path) as d:
    idx=d['indices'];vals=d['values'];cov=d['coverage'];skip=int(d['skipped'][0]);assert vals.shape==(len(NAMES),len(idx))
   skipped+=skip
   for j in range(len(NAMES)):
    ok=np.isfinite(vals[j]);np.add.at(sums[j],idx[ok],vals[j,ok])
   np.add.at(cover,idx,cov)
  except Exception:missing.append(str(path))
 if missing and not a.allow_incomplete:raise SystemExit(f'REFUSING INCOMPLETE YEAR: {len(missing)} missing/bad contributions. First: {missing[0]}')
 dest=Path(a.out)/str(a.year)/'annual'/f'accumulated_precip_data_ssmis_{a.year}.npz'
 arrays={('acc_'+name if name not in ('era5_snow','era5_rain') else 'accumulated_'+name):sums[i].reshape(SHAPE) for i,name in enumerate(NAMES)}
 arrays.update(orbit_coverage=cover.reshape(SHAPE),lat=GRID_LAT,lon=GRID_LON,missing_count=np.array(len(missing)),skipped_zero_precip=np.array(skipped))
 if dest.exists():raise SystemExit(f'Annual file already exists; refusing overwrite: {dest}')
 atomic_npz(dest,**arrays)
 print('Saved',dest,'input orbits',len(rows),'skipped zero',skipped,'missing',len(missing))
if __name__=='__main__':main()
