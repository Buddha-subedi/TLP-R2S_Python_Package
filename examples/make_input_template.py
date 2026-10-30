"""Create an input-format template with NaNs, NOT valid scientific predictors."""
import numpy as np
from tlpr2s import FEATURES
H,W=20,30
arrays={k:np.full((H,W),np.nan,dtype=np.float32) for k in FEATURES}
arrays['Month']=np.array(1,dtype=np.float32)
arrays['Day']=np.array(15,dtype=np.float32)
arrays['latitude']=np.broadcast_to(np.linspace(-5,5,H)[:,None],(H,W))
arrays['longitude']=np.broadcast_to(np.linspace(70,90,W)[None,:],(H,W))
np.savez_compressed('replace_with_real_inputs.npz',**arrays)
print('Template only. Replace all NaNs with collocated real SSMIS and ERA5 data before inference.')
