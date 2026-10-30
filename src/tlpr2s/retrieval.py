"""Faithful adaptation of uploaded inference.py: array API, same XGBoost and CDF method."""
from pathlib import Path
import os
from .model_download import download_models
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.interpolate import interp1d
from .arrays import feature_frame
from .features import FEATURES

@dataclass
class RetrievalResult:
    phase: np.ndarray
    rain: np.ndarray
    snow: np.ndarray
    @property
    def surface_precip(self):return self.rain+self.snow

def make_mapping(pred, actual):
    """Legacy CDF algorithm. Do not change without orbit-level parity tests."""
    pred=np.asarray(pred,dtype=float);actual=np.asarray(actual,dtype=float)
    if len(pred)<2 or len(actual)<2:raise ValueError('Need at least two CDF calibration samples')
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

class PrecipitationRetriever:
    """Load the original 3 XGBoost models and 2 CDF files; predict on 2D arrays.

    Phase: 0=clear, 1=rain, 2=snow. Units of rain/snow follow trained
    calibration targets (expected mm/hour); validate against your model metadata.
    """
    def __init__(self, model_dir=None, cdf_dir=None, threads=1, *, download=True, revision=None, cache_dir=None, local_files_only=False):
        """Use bundled models by default; override with local directories.

        The five trained artifacts must be supplied by the model owner.
        See models/README.md. This package does not train replacement models.
        """
        default=Path(__file__).resolve().parent/'model_data'
        env_dir=os.environ.get('TLPR2S_MODEL_DIR')
        if model_dir is not None:
            m=Path(model_dir)
        elif env_dir:
            m=Path(env_dir)
        elif all((default/name).is_file() for name in ('xgb_tl_classifier_final.json', 'xgb_tl_rain_rate_final.json', 'xgb_tl_snow_rate_final_1.json')) and all((default/name).is_file() for name in ('cdf_match_for_rain_ssmis.npz', 'cdf_match_for_snow_ssmis.npz')):
            m=default
        elif download:
            m=download_models(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
        else:
            m=default
        d=Path(cdf_dir) if cdf_dir is not None else m
        required=['xgb_tl_classifier_final.json','xgb_tl_rain_rate_final.json',
                  'cdf_match_for_rain_ssmis.npz','cdf_match_for_snow_ssmis.npz']
        missing=[str(m/x) if x.endswith('.json') else str(d/x) for x in required
                 if not (m/x if x.endswith('.json') else d/x).is_file()]
        if not any((m/n).is_file() for n in ('xgb_tl_snow_rate_final_1.json','xgb_tl_snow_rate_final.json')):
            missing.append(str(m/'xgb_tl_snow_rate_final_1.json'))
        if missing:
            raise FileNotFoundError('Missing pretrained model/calibration files: '+', '.join(missing)+
              '. Set model_dir, TLPR2S_MODEL_DIR, or enable download=True.')
        self.classifier=xgb.XGBClassifier(n_jobs=threads)
        self.classifier.load_model(str(m/'xgb_tl_classifier_final.json'))
        self.rain=xgb.Booster(params={'nthread':threads})
        self.rain.load_model(str(m/'xgb_tl_rain_rate_final.json'))
        self.snow=xgb.Booster(params={'nthread':threads})
        candidates=[m/'xgb_tl_snow_rate_final_1.json',m/'xgb_tl_snow_rate_final.json']
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:raise FileNotFoundError(f'Snow model missing: {candidates}')
        self.snow.load_model(str(path))
        self.maps={}
        for name,model,limit in [('rain',self.rain,40),('snow',self.snow,20)]:
            with np.load(d/f'cdf_match_for_{name}_ssmis.npz',allow_pickle=False) as z:
                df=pd.DataFrame({k:z[k] for k in z.files})
            target=f'cdf_{name}'
            missing=[k for k in FEATURES+[target] if k not in df]
            if missing:raise ValueError(f'{name} CDF file missing columns: {missing}')
            df=df[df[target].notna()&(df[target]>0)&(df[target]<limit)]
            if df.empty:raise ValueError(f'No valid {name} calibration samples')
            y=model.predict(xgb.DMatrix(df[FEATURES]))
            self.maps[name]=(make_mapping(y,df[target].values),float(y.min()),float(y.max()))

    def predict(self, data, feature_names=None):
        frame,shape=feature_frame(data,feature_names=feature_names)
        phase=np.asarray(self.classifier.predict(frame)).astype(np.int8)
        rain=np.zeros(len(frame),dtype=np.float32)
        snow=np.zeros(len(frame),dtype=np.float32)
        for label,name,model,out in [(1,'rain',self.rain,rain),(2,'snow',self.snow,snow)]:
            ids=np.flatnonzero(phase==label)
            if len(ids)==0:continue
            values=model.predict(xgb.DMatrix(frame.iloc[ids]))
            fn,lo,hi=self.maps[name]
            out[ids]=fn(np.clip(values,lo,hi))
        return RetrievalResult(phase.reshape(shape),rain.reshape(shape),snow.reshape(shape))
