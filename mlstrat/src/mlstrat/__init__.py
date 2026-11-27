"""mlstrat: strategy-based EDA, cleaning and training helpers for tabular data."""

from importlib.metadata import PackageNotFoundError, version

from mlstrat.eda import (
    BivariateStrategy,
    CategoricalCategoricalBivariate,
    CategoricalUnivariate,
    DataOverview,
    MultivariateAnalyzer,
    NumericCategoricalBivariate,
    NumericNumericBivariate,
    NumericUnivariate,
    UnivariateStrategy,
    bivariate_strategy,
    univariate_strategy,
)
from mlstrat.modeling import ModelTrainer, build_preprocessor, make_model_pipeline
from mlstrat.preprocessing import MissingValueImputer, OutlierClipper
from mlstrat.scaffold import create_project

try:
    __version__ = version("mlstrat")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "BivariateStrategy",
    "CategoricalCategoricalBivariate",
    "CategoricalUnivariate",
    "DataOverview",
    "MissingValueImputer",
    "ModelTrainer",
    "MultivariateAnalyzer",
    "NumericCategoricalBivariate",
    "NumericNumericBivariate",
    "NumericUnivariate",
    "OutlierClipper",
    "UnivariateStrategy",
    "__version__",
    "bivariate_strategy",
    "build_preprocessor",
    "create_project",
    "make_model_pipeline",
    "univariate_strategy",
]
