import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class StrategieMultivarie(ABC):

    @abstractmethod
    def plot_relationship(self, df:pd.DataFrame):
        pass

class MultivariateAnalyzer(StrategieMultivarie):

    def plot_relationship(self, df):
        plt.figure(figsize=(8,6))
        sns.pairplot(df)
        plt.show()
    
    def heatmap_correlation(self, df, columns: list = None):
        plt.figure(figsize=(10,8))
        if columns is not None:
            df = df[columns]
        df_numeric = df.select_dtypes(include='number')
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
        plt.show()

