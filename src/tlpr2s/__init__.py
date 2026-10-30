"""tlpr2s: SSMIS F16 precipitation retrieval from user-provided 2D arrays."""
from .features import FEATURES
from .arrays import feature_frame,save_swath,load_swath
from .retrieval import PrecipitationRetriever,RetrievalResult
from .annual import AnnualAccumulator,AnnualStatistics,aggregate_swaths
__version__='0.1.0'
__all__=['FEATURES','feature_frame','save_swath','load_swath','PrecipitationRetriever','RetrievalResult','AnnualAccumulator','AnnualStatistics','aggregate_swaths']
