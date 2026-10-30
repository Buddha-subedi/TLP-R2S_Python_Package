import numpy as np
from tlpr2s import FEATURES, plot_orbit, plot_annual, save_nc, predict, AnnualAccumulator

def test_exports():
    assert callable(plot_orbit) and callable(plot_annual) and callable(save_nc) and callable(predict)

def test_periodic_grid():
    a=AnnualAccumulator(); a.add(np.array([[0.,0.]]),np.array([[-180.,180.]]),np.array([[2.,4.]]),np.zeros((1,2)))
    s=a.result();assert s.mean_rain.shape==(721,1440)
    assert s.observation_count[360,0]==2
    assert s.mean_rain[360,0]==3

def test_features():
    assert len(FEATURES)==30
