"""Convenience API; full writer delegated to original DAAC-oriented schema."""
from .retrieval import PrecipitationRetriever
from .daac import DAAC_compatible_nc

def predict(features, *, model_dir=None, cdf_dir=None, threads=1, feature_names=None, download=True, revision=None, cache_dir=None, local_files_only=False):
    """Predict from a dictionary of 30 S4 2D features or stacked (30,npix,nscan) array."""
    return PrecipitationRetriever(model_dir=model_dir,cdf_dir=cdf_dir,threads=threads,download=download,revision=revision,cache_dir=cache_dir,local_files_only=local_files_only).predict(features,feature_names=feature_names)

def save_nc(output_path, *, result, latitude, longitude, input_arrays, ssmis_full_filename, ssmis_source_path, source_identifier='2D-array retrieval', model_version='TLP-R2S_V01'):
    """Write complete original DAAC-oriented S4 schema using authentic SSMIS scan times."""
    return DAAC_compatible_nc(output_path,result=result,latitude=latitude,longitude=longitude,input_arrays=input_arrays,ssmis_full_filename=ssmis_full_filename,ssmis_source_path=ssmis_source_path,source_identifier=source_identifier,model_version=model_version)
