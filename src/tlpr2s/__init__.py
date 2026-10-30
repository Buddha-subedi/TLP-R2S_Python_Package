"""TLP-R2S: SSMIS F16 S4-swath precipitation retrieval."""
from .model_download import download_models
from .features import FEATURES
from .arrays import feature_frame, save_swath, load_swath
from .retrieval import PrecipitationRetriever, RetrievalResult
from .annual import AnnualAccumulator, AnnualStatistics, aggregate_swaths
from .plotting import plot_orbit, plot_annual
from .daac import DAAC_compatible_nc
from .public import predict, save_nc
__version__ = "0.2.0"
__all__ = ["download_models", "FEATURES", "feature_frame", "save_swath", "load_swath", "PrecipitationRetriever", "RetrievalResult", "predict", "plot_orbit", "plot_annual", "save_nc", "DAAC_compatible_nc", "AnnualAccumulator", "AnnualStatistics", "aggregate_swaths"]
