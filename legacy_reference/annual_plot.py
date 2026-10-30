import argparse
from pathlib import Path
import numpy as np
from common import GRID_LAT, GRID_LON
from precipitation_map import save_combined_map

p = argparse.ArgumentParser()
p.add_argument('--file', required=True)
a = p.parse_args()
with np.load(a.file) as d:
    coverage = d['orbit_coverage']
    rain = d['acc_rain_tl']
    snow = d['acc_snow_tl']
    rain_mean = np.divide(rain, coverage, out=np.full_like(rain, np.nan, dtype=float), where=coverage>0)
    snow_mean = np.divide(snow, coverage, out=np.full_like(snow, np.nan, dtype=float), where=coverage>0)
lon, lat = np.meshgrid(GRID_LON, GRID_LAT)
out = Path(a.file).with_suffix('.png')
save_combined_map(out, lon, lat, rain_mean, snow_mean,
    title=f'TLP-R2S Mean Precipitation Rate - {Path(a.file).stem}')
print(out)
