import numpy as np
from tlpr2s import AnnualAccumulator,AnnualStatistics

def test_pixel_mean_and_coverage(tmp_path):
    a=AnnualAccumulator(resolution=1.0)
    lat=np.array([[0.2,0.3]]);lon=np.array([[0.1,0.2]])
    assert a.add(lat,lon,np.array([[2.,4.]]),np.array([[1.,1.]]))==2
    assert a.add(lat[:,:1],lon[:,:1],np.array([[6.]]),np.array([[0.]]))==1
    s=a.result();i,j=90,180
    assert s.observation_count[i,j]==3
    assert s.swath_coverage_count[i,j]==2
    assert np.isclose(s.mean_rain[i,j],4)
    assert np.isclose(s.mean_precip[i,j],(3+5+6)/3)
    assert np.isnan(s.mean_precip[0,0])
    path=tmp_path/'annual.npz';s.save(path)
    assert AnnualStatistics.load(path).shape==(181,360)

def test_dateline():
    a=AnnualAccumulator(1)
    a.add(np.array([[0.,0.]]),np.array([[-180.,180.]]),np.array([[1.,3.]]),np.array([[0.,0.]]))
    s=a.result();assert s.observation_count[90,0]==2


def test_default_global_grid_and_seam():
    a=AnnualAccumulator()
    lat=np.array([[-90.,90.,0.,0.]])
    lon=np.array([[-180.,180.,-180.,180.]])
    a.add(lat,lon,np.ones((1,4)),np.zeros((1,4)))
    r=a.result()
    assert r.shape==(721,1440)
    assert r.latitude[0]==-90 and r.latitude[-1]==90
    assert r.longitude[0]==-180 and r.longitude[-1]==179.75
    assert r.observation_count[360,0]==2
    assert r.observation_count[0,0]==1
    assert r.observation_count[-1,0]==1
