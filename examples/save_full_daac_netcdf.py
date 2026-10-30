"""Template: use real model output and original F16 granule, not synthetic metadata."""
import numpy as np
from tlpr2s import PrecipitationRetriever, DAAC_compatible_nc

# Prepare your actual S4-aligned arrays. In this example the NPZ contains:
# 30 inference predictors, latitude, longitude, TbS1..TbS4, and ERA5_* fields.
with np.load('prepared_s4_arrays.npz') as z:
    arrays={k:z[k] for k in z.files}

# You MUST provide the full actual filename and source granule on disk.
full_name=input('Full original SSMIS F16 granule filename: ').strip()
source_path=input('Path to that original SSMIS .nc granule: ').strip()

model=PrecipitationRetriever()
result=model.predict(arrays)
DAAC_compatible_nc('TLP-R2S_V01_'+full_name,
    result=result, latitude=arrays['latitude'], longitude=arrays['longitude'],
    input_arrays=arrays, ssmis_full_filename=full_name,
    ssmis_source_path=source_path)
