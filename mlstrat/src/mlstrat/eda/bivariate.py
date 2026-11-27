"""Two-column analysis strategies: numeric/numeric, numeric/categorical, categorical/categorical."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy import stats


def _axes(ax: Optional[Axes]) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    return ax


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


class BivariateStrategy(ABC):
    """Common interface for the relationship between two columns ``x`` and ``y``."""

    @abstractmethod
    def plot(self, df: pd.DataFrame, x: str, y: str, ax: Optional[Axes] = None) -> Axes:
        """Default plot of the relationship."""

    @abstractmethod
    def test(self, df: pd.DataFrame, x: str, y: str) -> dict:
        """Default statistical test of the relationship."""


class NumericNumericBivariate(BivariateStrategy):
    """Two numeric columns."""

    def plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
        ax.set_title(f"{y} vs {x}")
        return ax

    def test(self, df, x, y, method: str = "pearson") -> dict:
        """Correlation coefficient and p-value (``pearson`` or ``spearman``)."""
        data = df[[x, y]].dropna()
        if method == "pearson":
            r, p = stats.pearsonr(data[x], data[y])
        elif method == "spearman":
            r, p = stats.spearmanr(data[x], data[y])
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
        return {"method": method, "correlation": float(r), "p_value": float(p), "n": len(data)}

    def regression_plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.regplot(data=df, x=x, y=y, line_kws={"color": "red"}, ax=ax)
        ax.set_title(f"Regression of {y} on {x}")
        return ax

    def heatmap(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.heatmap(df[[x, y]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation between {x} and {y}")
        return ax


class NumericCategoricalBivariate(BivariateStrategy):
    """A numeric column ``x`` against a categorical column ``y``."""

    def plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.boxplot(data=df, x=y, y=x, ax=ax)
        ax.set_title(f"{x} by {y}")
        return ax

    def test(self, df, x, y) -> dict:
        """One-way ANOVA of ``x`` across the groups of ``y``."""
        groups = [g[x].dropna().to_numpy() for _, g in df.groupby(y, observed=True)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            raise ValueError(f"ANOVA needs at least two non-empty groups in '{y}'")
        f_stat, p = stats.f_oneway(*groups)
        return {"method": "anova", "f_statistic": float(f_stat), "p_value": float(p), "n_groups": len(groups)}

    def violin_plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.violinplot(data=df, x=y, y=x, ax=ax)
        ax.set_title(f"{x} by {y}")
        return ax

    def mean_bar_plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.barplot(data=df, x=y, y=x, ax=ax)
        ax.set_title(f"Mean {x} by {y}")
        return ax


class CategoricalCategoricalBivariate(BivariateStrategy):
    """Two categorical columns."""

    def plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        sns.countplot(data=df, x=x, hue=y, ax=ax)
        ax.set_title(f"{x} by {y}")
        return ax

    def test(self, df, x, y) -> dict:
        """Chi-square test of independence."""
        table = pd.crosstab(df[x], df[y])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        return {"method": "chi2", "chi2": float(chi2), "p_value": float(p), "dof": int(dof)}

    def stacked_bar_plot(self, df, x, y, ax=None):
        ax = _axes(ax)
        pd.crosstab(df[x], df[y]).plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{x} by {y}")
        return ax


def bivariate_strategy(df: pd.DataFrame, x: str, y: str) -> BivariateStrategy:
    """Pick the strategy from the dtypes of ``x`` and ``y``.

    When exactly one column is numeric, pass it as ``x``.
    """
    x_num, y_num = _is_numeric(df[x]), _is_numeric(df[y])
    if x_num and y_num:
        return NumericNumericBivariate()
    if x_num and not y_num:
        return NumericCategoricalBivariate()
    if not x_num and not y_num:
        return CategoricalCategoricalBivariate()
    raise ValueError(
        f"'{x}' is categorical and '{y}' is numeric: swap them, "
        "NumericCategoricalBivariate expects the numeric column first"
    )
