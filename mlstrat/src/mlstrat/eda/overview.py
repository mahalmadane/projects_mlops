"""Whole-DataFrame overview: missing values, types, summary statistics."""

from __future__ import annotations

import pandas as pd


class DataOverview:
    """Quick, non-plotting overview of a DataFrame.

    Every method returns a pandas object instead of printing, so results can be
    displayed in a notebook, logged, or tested.
    """

    def missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count and percentage of missing values per column, most missing first."""
        count = df.isna().sum()
        pct = (count / len(df) * 100).round(2) if len(df) else count * 0.0
        out = pd.DataFrame({"missing": count, "missing_pct": pct})
        return out.sort_values("missing", ascending=False)

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics for every column (numeric and categorical)."""
        return df.describe(include="all").T

    def dtypes(self, df: pd.DataFrame) -> pd.Series:
        """Data type of each column."""
        return df.dtypes

    def report(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per column: dtype, missing values, unique values."""
        return pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
                "n_unique": df.nunique(dropna=True),
            }
        )
