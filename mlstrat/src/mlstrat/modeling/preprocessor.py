"""Ready-made scikit-learn preprocessing for mixed numeric/categorical tables."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scale: bool = True,
) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode the others.

    Columns are selected by dtype when the transformer is fitted, so the same
    object works on any DataFrame.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )


def make_model_pipeline(estimator: BaseEstimator, **preprocessor_kwargs) -> Pipeline:
    """``Pipeline([("preprocessor", build_preprocessor(...)), ("model", estimator)])``."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(**preprocessor_kwargs)),
            ("model", estimator),
        ]
    )
