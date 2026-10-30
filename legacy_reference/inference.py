"""Source-faithful feature order, classification, regression and CDF calibration.
The original notebook recomputed CDF mappings per orbit; this caches the mapping once per process.
"""
import numpy as np
import pandas as pd
import scipy.io
import xgboost as xgb
from scipy import stats
from scipy.interpolate import interp1d
from common import FEATURES,FIELD_MAP

def make_mapping(pred,actual):
    pred=pd.Series(pred).clip(lower=np.min(actual)).to_numpy()
    a=stats.cumfreq(pred,numbins=min(len(pred),4000))
    b=stats.cumfreq(actual,numbins=min(len(actual),4000))
    xa=a.lowerlimit+np.arange(len(a.cumcount))*a.binsize
    xb=b.lowerlimit+np.arange(len(b.cumcount))*b.binsize
    fa=a.cumcount/a.cumcount[-1];fb=b.cumcount/b.cumcount[-1]
    xa,ix=np.unique(xa,return_index=True);fa=fa[ix]
    fb,ix=np.unique(fb,return_index=True);xb=xb[ix]
    if len(xa)<2 or len(fb)<2:raise ValueError('Insufficient CDF calibration support')
    f1=interp1d(xa,fa,fill_value='extrapolate',assume_sorted=True)
    f2=interp1d(fb,xb,fill_value='extrapolate',assume_sorted=True)
    def apply(v):
        x=f2(f1(v));x=np.where(np.isnan(x),0,x)
        return np.where(x<0,.001,x)
    return apply

class Predictor:
 def __init__(self,model_dir,cdf_dir,threads=1):
    from pathlib import Path
    m=Path(model_dir);d=Path(cdf_dir)
    self.classifier=xgb.XGBClassifier(n_jobs=threads)
    self.classifier.load_model(str(m/'xgb_tl_classifier_final.json'))
    self.rain=xgb.Booster(params={'nthread':threads});self.rain.load_model(str(m/'xgb_tl_rain_rate_final.json'))
    self.snow=xgb.Booster(params={'nthread':threads})
    snow_candidates=[m/'xgb_tl_snow_rate_final_1.json',m/'xgb_tl_snow_rate_final.json']
    snow_path=next((p for p in snow_candidates if p.is_file()),None)
    if snow_path is None:raise FileNotFoundError(f'Snow model missing; checked: {snow_candidates}')
    print(f'Using snow model: {snow_path}',flush=True)
    self.snow.load_model(str(snow_path))
    self.maps={}
    for name,model,limit in [('rain',self.rain,40),('snow',self.snow,20)]:
        with np.load(d/f'cdf_match_for_{name}_ssmis.npz') as z:
            df=pd.DataFrame({k:z[k] for k in z.files})
        target=f'cdf_{name}';df=df[df[target].notna()&(df[target]>0)&(df[target]<limit)]
        y=model.predict(xgb.DMatrix(df[FEATURES]))
        self.maps[name]=(make_mapping(y,df[target].values),float(y.min()),float(y.max()))
 def predict(self,path):
    mat=scipy.io.loadmat(path);X=mat['X'][0,0]
    shape=X['LatS4'].shape
    tb={'19.35v':('TbS1',0),'19.35h':('TbS1',1),'22.235v':('TbS1',2),'37v':('TbS2',0),'37h':('TbS2',1),'150h':('TbS3',0),'183-1':('TbS3',1),'183-3':('TbS3',2),'183-7':('TbS3',3),'91.665v':('TbS4',0),'91.665h':('TbS4',1)}
    columns={}
    for key,(field,channel) in tb.items():
        v=np.asarray(X[field][:,:,channel],dtype=np.float32)
        columns[key]=np.where(v<0,np.nan,v).ravel()
    for key,field in FIELD_MAP.items():columns[key]=np.asarray(X[field]).ravel()
    for key in ('Month','Day'):
        v=np.asarray(X[key]);
        if v.size==1:columns[key]=np.full(shape[0]*shape[1],v.item())
        elif v.size==shape[1]:columns[key]=np.tile(v.ravel(),shape[0])
        elif v.size==shape[0]*shape[1]:columns[key]=v.ravel()
        else:raise ValueError(f'Unexpected {key} shape: {v.shape}')
    df=pd.DataFrame(columns)[FEATURES]
    phase=np.asarray(self.classifier.predict(df)).astype(np.int8)
    rain=np.zeros(len(df),dtype=np.float32);snow=np.zeros(len(df),dtype=np.float32)
    for label,name,model,out in [(1,'rain',self.rain,rain),(2,'snow',self.snow,snow)]:
        ids=np.flatnonzero(phase==label)
        if not len(ids):continue
        v=model.predict(xgb.DMatrix(df.iloc[ids]));fn,lo,hi=self.maps[name]
        out[ids]=fn(np.clip(v,lo,hi))
    return mat,phase.reshape(shape),rain.reshape(shape),snow.reshape(shape)
