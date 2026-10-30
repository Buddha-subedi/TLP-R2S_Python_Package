"""TLP-R2S: SSMIS F16 S4-swath precipitation retrieval."""
from .features import FEATURES
from .arrays import feature_frame, save_swath, load_swath
from .retrieval import PrecipitationRetriever, RetrievalResult
from .annual import AnnualAccumulator, AnnualStatistics, aggregate_swaths
from .plotting import plot_orbit, plot_annual
from .daac import DAAC_compatible_nc
from .public import predict, save_nc
__version__ = "0.1.0"
__all__ = ["FEATURES", "feature_frame", "save_swath", "load_swath", "PrecipitationRetriever", "RetrievalResult", "predict", "plot_orbit", "plot_annual", "save_nc", "DAAC_compatible_nc", "AnnualAccumulator", "AnnualStatistics", "aggregate_swaths"]
