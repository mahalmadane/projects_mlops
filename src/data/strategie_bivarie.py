import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class StrategieBivarie(ABC):

    @abstractmethod
    def plot_relationship(self, df:pd.DataFrame, name1:str, name2:str):
        pass


class StrategieNUmNUMBivarie(StrategieBivarie):

    def plot_relationship(self, df, name1, name2):
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x=name1, y=name2)
        plt.title(f'Scatter plot between {name1} and {name2}')
        plt.show()

    def correlation_analysis(self, df, name1, name2):
        correlation = df[[name1, name2]].corr().iloc[0,1]
        print(f'Correlation between {name1} and {name2}: {correlation}')

    def regression_plot(self, df, name1, name2):
        plt.figure(figsize=(8,6))
        sns.regplot(data=df, x=name1, y=name2, line_kws={"color":"red"})
        plt.title(f'Regression plot between {name1} and {name2}')
        plt.show()
    
    def heatmap_correlation(self, df, name1, name2):
        corr_matrix = df[[name1, name2]].corr()
        plt.figure(figsize=(6,4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Heatmap of correlation between {name1} and {name2}')
        plt.show()
    
class StrategieNumCatBivarie(StrategieBivarie):
    def plot_relationship(self, df, name1, name2):
        plt.figure(figsize=(8,6))
        sns.boxplot(x=name2, y=name1, data=df)
        plt.title(f'Box plot of {name1} by {name2}')
        plt.show()
    
    def anova_analysis(self, df, name1, name2):
        import scipy.stats as stats
        groups = [group[name1].values for name, group in df.groupby(name2)]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f'ANOVA results between {name1} and {name2}: F-statistic={f_stat}, p-value={p_value}')
    
    def violin_plot(self, df, name1, name2):
        plt.figure(figsize=(8,6))
        sns.violinplot(x=name2, y=name1, data=df)
        plt.title(f'Violin plot of {name1} by {name2}')
        plt.show()
    
    def bar_plot_means(self, df, name1, name2):
        means = df.groupby(name2)[name1].mean().reset_index()
        plt.figure(figsize=(8,6))
        sns.barplot(x=name2, y=name1, data=means)
        plt.title(f'Bar plot of mean {name1} by {name2}')
        plt.show()
class StrategieCatCatBivarie(StrategieBivarie):
    def plot_relationship(self, df, name1, name2):
        plt.figure(figsize=(8,6))
        sns.countplot(x=name1, hue=name2, data=df)
        plt.title(f'Count plot of {name1} by {name2}')
        plt.show()
    
    def chi_square_analysis(self, df, name1, name2):
        import scipy.stats as stats
        contingency_table = pd.crosstab(df[name1], df[name2])
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        print(f'Chi-square results between {name1} and {name2}: Chi2={chi2}, p-value={p}, dof={dof}')
    
    def stacked_bar_plot(self, df, name1, name2):
        contingency_table = pd.crosstab(df[name1], df[name2])
        contingency_table.plot(kind='bar', stacked=True, figsize=(8,6))
        plt.title(f'Stacked bar plot of {name1} by {name2}')
        plt.show()
    