"""Streaming swath-to-grid statistics. A sampling mean, not an annual accumulation."""
from dataclasses import dataclass
import numpy as np
from .arrays import load_swath

@dataclass
class AnnualStatistics:
    latitude: np.ndarray
    longitude: np.ndarray
    mean_rain: np.ndarray
    mean_snow: np.ndarray
    mean_precip: np.ndarray
    observation_count: np.ndarray
    swath_coverage_count: np.ndarray
    @property
    def shape(self):return self.mean_precip.shape
    def save(self,path):
        np.savez_compressed(path,**vars(self))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:return cls(**{k:z[k] for k in cls.__dataclass_fields__})

class AnnualAccumulator:
    """Periodic longitude grid and inclusive polar latitude nodes.

    Default grid: latitude [-90,90] inclusive (721 points), longitude
    [-180,180) (1440 points). -180 and +180 map to the SAME column.
    Observations are assigned to nearest grid node. Each pixel counts once;
    annual means are sampling means, not time-weighted accumulations.
    """
    def __init__(self,resolution=0.25):
        nlat=180/resolution;nlon=360/resolution
        if int(nlat)!=nlat or int(nlon)!=nlon:
            raise ValueError('resolution must divide 180 and 360')
        self.resolution=float(resolution)
        self.nlat=int(nlat)+1
        self.nlon=int(nlon)
        n=self.nlat*self.nlon
        self.count=np.zeros(n,dtype=np.int64)
        self.coverage=np.zeros(n,dtype=np.int64)
        self.rain_sum=np.zeros(n,dtype=np.float64)
        self.snow_sum=np.zeros(n,dtype=np.float64)
    def add(self,latitude,longitude,rain,snow):
        lat=np.asarray(latitude);lon=np.asarray(longitude)
        r=np.asarray(rain);s=np.asarray(snow)
        if lat.ndim!=2 or any(v.shape!=lat.shape for v in (lon,r,s)):
            raise ValueError('All swath arrays must have identical 2D shape')
        good=(np.isfinite(lat)&np.isfinite(lon)&np.isfinite(r)&np.isfinite(s)
              &(lat>=-90)&(lat<=90)&(r>=0)&(s>=0))
        if not good.any():return 0
        # Round to nearest latitude node, including both poles.
        la=np.clip(np.floor((lat[good]+90)/self.resolution+0.5).astype(np.int64),0,self.nlat-1)
        # Periodic longitude: -180 and +180 map to column 0.
        lo=(np.floor(((lon[good]+180)%360)/self.resolution+0.5).astype(np.int64))%self.nlon
        idx=la*self.nlon+lo
        self.count+=np.bincount(idx,minlength=self.count.size)
        self.rain_sum+=np.bincount(idx,weights=r[good],minlength=self.count.size)
        self.snow_sum+=np.bincount(idx,weights=s[good],minlength=self.count.size)
        self.coverage[np.unique(idx)]+=1
        return int(good.sum())
    def add_file(self,path):
        d=load_swath(path)
        return self.add(d['latitude'],d['longitude'],d['rain'],d['snow'])
    def result(self):
        shape=(self.nlat,self.nlon)
        def mean(total):
            return np.divide(total,self.count,out=np.full_like(total,np.nan),where=self.count>0).reshape(shape)
        lat=-90+np.arange(self.nlat)*self.resolution
        lon=-180+np.arange(self.nlon)*self.resolution
        return AnnualStatistics(lat,lon,mean(self.rain_sum),mean(self.snow_sum),mean(self.rain_sum+self.snow_sum),self.count.reshape(shape).copy(),self.coverage.reshape(shape).copy())

def aggregate_swaths(paths,resolution=0.25):
    acc=AnnualAccumulator(resolution)
    paths=list(paths)
    if not paths:raise ValueError('No swath files provided')
    for p in paths:acc.add_file(p)
    return acc.result()
