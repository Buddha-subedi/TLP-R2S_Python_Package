import numpy as np
from tlpr2s.retrieval import make_mapping

def test_mapping_is_finite():
    x=np.linspace(.1,10,200);fn=make_mapping(x,x*1.1)
    assert np.isfinite(fn(np.array([1.,3.,5.]))).all()
