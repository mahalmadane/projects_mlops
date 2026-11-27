"""Single-column analysis strategies (numeric vs categorical)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _check_column(df: pd.DataFrame, name: str) -> None:
    if name not in df.columns:
        raise KeyError(f"Column '{name}' not found in DataFrame")


class UnivariateStrategy(ABC):
    """Common interface for analysing one column of a DataFrame.

    Methods never modify the DataFrame passed in: ``impute`` and
    ``clip_outliers`` return a new Series.
    """

    @abstractmethod
    def impute(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Return the column with missing values filled."""

    @abstractmethod
    def clip_outliers(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Return the column with outliers handled."""

    @abstractmethod
    def plot(self, df: pd.DataFrame, name: str):
        """Plot the distribution of the column."""

    @abstractmethod
    def describe(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Summary statistics of the column."""


class NumericUnivariate(UnivariateStrategy):
    """Strategy for numeric columns.

    Args:
        fill: statistic used by :meth:`impute` (``"mean"``, ``"median"`` or ``"mode"``).
        iqr_factor: multiplier of the IQR used by :meth:`clip_outliers`.
    """

    def __init__(self, fill: str = "mean", iqr_factor: float = 1.5):
        if fill not in {"mean", "median", "mode"}:
            raise ValueError("fill must be one of: mean, median, mode")
        self.fill = fill
        self.iqr_factor = iqr_factor

    def impute(self, df: pd.DataFrame, name: str) -> pd.Series:
        _check_column(df, name)
        col = df[name]
        if self.fill == "mode":
            mode = col.mode()
            value = mode.iloc[0] if not mode.empty else None
        else:
            value = getattr(col, self.fill)()
        return col.fillna(value) if value is not None and not pd.isna(value) else col.copy()

    def bounds(self, df: pd.DataFrame, name: str) -> tuple:
        """Lower and upper IQR bounds ``(Q1 - k*IQR, Q3 + k*IQR)``."""
        _check_column(df, name)
        q1, q3 = df[name].quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr

    def clip_outliers(self, df: pd.DataFrame, name: str) -> pd.Series:
        lower, upper = self.bounds(df, name)
        return df[name].clip(lower=lower, upper=upper)

    def plot(self, df: pd.DataFrame, name: str):
        """Histogram with KDE next to a box plot. Returns the figure."""
        _check_column(df, name)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(data=df, x=name, ax=axes[0], kde=True)
        sns.boxplot(data=df, x=name, ax=axes[1])
        fig.suptitle(f"Distribution of {name}")
        return fig

    def describe(self, df: pd.DataFrame, name: str) -> pd.Series:
        _check_column(df, name)
        return df[name].describe()


class CategoricalUnivariate(UnivariateStrategy):
    """Strategy for categorical / text columns. Missing values are filled with the mode."""

    def impute(self, df: pd.DataFrame, name: str) -> pd.Series:
        _check_column(df, name)
        mode = df[name].mode()
        if mode.empty:
            return df[name].copy()
        return df[name].fillna(mode.iloc[0])

    def clip_outliers(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Outlier clipping does not apply to categories: returns the column unchanged."""
        _check_column(df, name)
        return df[name].copy()

    def plot(self, df: pd.DataFrame, name: str, ax: Optional[Axes] = None) -> Axes:
        """Count plot ordered by frequency. Returns the Axes."""
        _check_column(df, name)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=name, order=df[name].value_counts().index, ax=ax)
        ax.set_title(f"Counts of {name}")
        return ax

    def describe(self, df: pd.DataFrame, name: str) -> pd.Series:
        _check_column(df, name)
        return df[name].value_counts(dropna=False)


def univariate_strategy(series: pd.Series, **kwargs) -> UnivariateStrategy:
    """Pick the right strategy from the column's dtype.

    Booleans are treated as categories. Extra keyword arguments go to
    :class:`NumericUnivariate`.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return NumericUnivariate(**kwargs)
    return CategoricalUnivariate()
