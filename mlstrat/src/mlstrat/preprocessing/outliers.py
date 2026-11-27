"""IQR-based outlier clipping with a chainable API."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class OutlierClipper:
    """Clip numeric values to ``[Q1 - k*IQR, Q3 + k*IQR]``. The input DataFrame is copied.

    Example:
        >>> df_clean = OutlierClipper(df).clip("Fare", factor=3).clip_all(exclude=["Survived"]).get_df()
    """

    def __init__(self, df: pd.DataFrame, factor: float = 1.5):
        self.df = df.copy()
        self.factor = factor
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def clip(self, column: str, factor: Optional[float] = None) -> "OutlierClipper":
        """Clip one numeric column. Non-numeric columns are skipped with a warning."""
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        col = self.df[column]
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            logger.warning("Column '%s' is not numeric; skipped", column)
            return self
        k = self.factor if factor is None else factor
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        n_out = int(((col < lower) | (col > upper)).sum())
        self.df[column] = col.clip(lower=lower, upper=upper)
        self.bounds[column] = (float(lower), float(upper))
        logger.info("Clipped %d outliers in '%s' to [%.4g, %.4g]", n_out, column, lower, upper)
        return self

    def clip_all(self, exclude: Iterable[str] = ()) -> "OutlierClipper":
        """Clip every numeric column not already clipped, except those in ``exclude``."""
        exclude = set(exclude)
        for column in self.df.select_dtypes(include="number").columns:
            if column not in exclude and column not in self.bounds:
                self.clip(column)
        return self

    def get_df(self) -> pd.DataFrame:
        """The clipped DataFrame."""
        return self.df
