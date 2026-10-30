import numpy as np
import pytest
from tlpr2s import FEATURES,feature_frame,save_swath,load_swath

def test_dict_and_stack():
    data={k:np.full((3,4),i,dtype=np.float32) for i,k in enumerate(FEATURES)}
    df,shape=feature_frame(data)
    assert shape==(3,4) and list(df.columns)==FEATURES and len(df)==12
    stack=np.stack([data[k] for k in FEATURES]);df2,_=feature_frame(stack,feature_names=FEATURES)
    assert np.allclose(df.values,df2.values)

def test_missing_and_shape():
    d={k:np.ones((2,3)) for k in FEATURES}
    d.pop(FEATURES[0])
    with pytest.raises(ValueError,match='Missing'):feature_frame(d)
    d[FEATURES[0]]=np.ones((4,5))
    with pytest.raises(ValueError,match='shape'):feature_frame(d)

def test_swath_roundtrip(tmp_path):
    a=np.ones((2,3));p=tmp_path/'s.npz'
    save_swath(p,a,a,np.zeros((2,3),dtype=np.int8),a,a)
    assert load_swath(p)['rain'].shape==(2,3)
