"""Missing value imputation with a chainable API."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_STRATEGIES = ("mean", "median", "mode", "min", "max")


class MissingValueImputer:
    """Fill missing values column by column. The input DataFrame is copied, never modified.

    Numeric columns accept ``mean``, ``median``, ``mode``, ``min``, ``max`` or a
    constant; other columns are filled with their mode (or a constant).

    Example:
        >>> df_clean = (
        ...     MissingValueImputer(df)
        ...     .impute("Age", "median")
        ...     .impute("Embarked")
        ...     .impute_all()
        ...     .get_df()
        ... )
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.fill_values: Dict[str, object] = {}

    def _value(self, feature: str, strategy) -> object:
        col = self.df[feature]
        numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
        if strategy is None:
            strategy = "mean" if numeric else "mode"
        if not isinstance(strategy, str):  # constant fill value
            return strategy
        if strategy not in NUMERIC_STRATEGIES and not numeric:  # text constant, e.g. "Unknown"
            return strategy
        if strategy == "mode":
            mode = col.mode()
            return mode.iloc[0] if not mode.empty else None
        if not numeric:
            raise ValueError(f"Column '{feature}' is not numeric: use strategy 'mode' or a constant")
        if strategy not in NUMERIC_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(NUMERIC_STRATEGIES)}")
        return getattr(col, strategy)()

    def impute(self, feature: str, strategy=None) -> "MissingValueImputer":
        """Fill one column.

        Args:
            feature: column name.
            strategy: ``mean`` / ``median`` / ``mode`` / ``min`` / ``max``, or a
                constant value. Defaults to ``mean`` for numeric columns and
                ``mode`` otherwise.
        """
        if feature not in self.df.columns:
            raise KeyError(f"Column '{feature}' not found in DataFrame")
        n_missing = int(self.df[feature].isna().sum())
        if n_missing == 0:
            return self
        value = self._value(feature, strategy)
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            logger.warning("Column '%s' is entirely empty; left unchanged", feature)
            return self
        self.df[feature] = self.df[feature].fillna(value)
        self.fill_values[feature] = value
        logger.info("Filled %d missing values in '%s' with %r", n_missing, feature, value)
        return self

    def impute_all(self, strategies: Optional[Dict[str, object]] = None) -> "MissingValueImputer":
        """Fill every column that still has missing values.

        Args:
            strategies: optional per-column strategy, e.g. ``{"Age": "median"}``.
                Other columns use the default (``mean`` or ``mode``).
        """
        strategies = strategies or {}
        for feature in self.df.columns[self.df.isna().any()]:
            self.impute(feature, strategies.get(feature))
        return self

    def get_df(self) -> pd.DataFrame:
        """The imputed DataFrame."""
        return self.df
