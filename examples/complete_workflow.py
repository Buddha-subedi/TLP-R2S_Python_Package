"""Template: replace the array loader with your own preprocessed S4 arrays."""
import numpy as np
from pathlib import Path
from tlpr2s import PrecipitationRetriever, plot_orbit, plot_annual, save_nc, save_swath, AnnualAccumulator

# Prepare a .npz with ALL 30 feature keys, 2D latitude/longitude, plus
# the ORIGINAL DAAC fields required by save_nc. This is NOT raw L1C ingestion.
with np.load('my_s4_arrays.npz',allow_pickle=False) as z:
    inputs={k:z[k] for k in z.files}
lat=inputs.pop('latitude');lon=inputs.pop('longitude')
from tlpr2s import FEATURES
features={k:inputs[k] for k in FEATURES}
retriever=PrecipitationRetriever()  # expects 5 bundled original artifacts
result=retriever.predict(features)
plot_orbit(lat,lon,result.surface_precip,save='orbit.png')
save_swath('orbit_retrieved.npz',lat,lon,result.phase,result.rain,result.snow)
acc=AnnualAccumulator(resolution=0.25)
acc.add(lat,lon,result.rain,result.snow)
annual=acc.result();annual.save('annual_stats.npz')
plot_annual('annual_stats.npz',save='annual_mean.png')
# This final step needs the ORIGINAL SSMIS granule, not merely its name:
# save_nc('orbit.nc',result=result,latitude=lat,longitude=lon,
#         input_arrays=inputs,ssmis_full_filename='1C.F16.SSMIS.XCAL2021-....HDF5',
#         ssmis_source_path='/path/to/original/1C.F16.SSMIS.XCAL2021-....HDF5')
