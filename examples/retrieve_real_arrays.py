"""Edit the NPZ filename and model paths to use real data."""
import numpy as np
from tlpr2s import FEATURES,PrecipitationRetriever,save_swath
input_file='real_30_features_and_geolocation.npz'
with np.load(input_file,allow_pickle=False) as z:
    data={k:z[k] for k in FEATURES}
    latitude=z['latitude'];longitude=z['longitude']
retriever=PrecipitationRetriever('models','cdf')
result=retriever.predict(data)
save_swath('retrieved_orbit.npz',latitude,longitude,result.phase,result.rain,result.snow)
print('Saved retrieved_orbit.npz')
