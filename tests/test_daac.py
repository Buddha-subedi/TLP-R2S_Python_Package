import numpy as np
import pytest
from types import SimpleNamespace
from tlpr2s import DAAC_compatible_nc


def test_full_filename_is_required(tmp_path):
    zeros=np.zeros((3,4),dtype=float)
    result=SimpleNamespace(phase=zeros,rain=zeros,snow=zeros)
    with pytest.raises(ValueError,match='filename'):
        DAAC_compatible_nc(tmp_path/'out.nc',result=result,latitude=zeros,
            longitude=zeros,input_arrays={},ssmis_full_filename='short.nc',
            ssmis_source_path=tmp_path/'short.nc')
