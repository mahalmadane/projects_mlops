"""Analysis across many columns at once."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


class MultivariateAnalyzer:
    """Pair plots and correlation matrices."""

    def pairplot(self, df: pd.DataFrame, columns: Optional[Sequence[str]] = None, hue: Optional[str] = None):
        """Seaborn pair plot of the numeric columns. Returns the ``PairGrid``."""
        if columns is not None:
            cols = list(columns) + ([hue] if hue and hue not in columns else [])
            df = df[cols]
        return sns.pairplot(df, hue=hue)

    def correlation(self, df: pd.DataFrame, columns: Optional[Sequence[str]] = None, method: str = "pearson") -> pd.DataFrame:
        """Correlation matrix of the numeric columns."""
        if columns is not None:
            df = df[list(columns)]
        return df.select_dtypes(include="number").corr(method=method)

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        method: str = "pearson",
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Annotated heatmap of :meth:`correlation`."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.correlation(df, columns, method), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{method.capitalize()} correlation")
        return ax
