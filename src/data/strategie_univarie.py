import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class StrategieUnivarie(ABC):

    @abstractmethod
    def missing_value(self, df:pd.DataFrame,name:str):
        pass

    @abstractmethod
    def outliers_management(self, df:pd.DataFrame,name:str):
        pass

    @abstractmethod
    def plot_distribution(self, df:pd.DataFrame,name:str):
        pass

    @abstractmethod
    def description(self, df:pd.DataFrame,name:str):
        pass

class StrategieNumberValue(StrategieUnivarie):
    
    def missing_value(self, df, name):
        if df[name].isnull().sum()>0:
            print(f"{name} has {df[name].isnull().sum()} missing values")
            df.loc[(df[name].isnull() | df[name].isna()),name]=df[name].mean()
            print(df[name].isnull().sum())
        else:
            print(f"{name} has no missing values")

    def outliers_management(self, df, name):
        Q1=df[name].quantile(0.25)
        Q3=df[name].quantile(0.75)
        IQR=Q3-Q1
        lower_bound=Q1-1.5*IQR
        upper_bound=Q3+1.5*IQR
        df.loc[(df[name]<lower_bound),name]=lower_bound
        df.loc[(df[name]>upper_bound),name]=upper_bound

    def plot_distribution(self, df, name):
        fig, axes=plt.subplots(1,2, figsize=(12,5))
        sns.histplot(data=df, x=name, ax=axes[0], kde=True)
        sns.boxplot(data=df, x=name, ax=axes[1])
        plt.show()
    
    def description(self, df, name):
        print(df[name].describe())

class StrategieCategoricalValue(StrategieUnivarie):

    def missing_value(self, df, name):
        if df[name].isnull().sum()>0:
            print(f"{name} has {df[name].isnull().sum()} missing values")
            df.loc[(df[name].isnull() | df[name].isna()),name]=df[name].mode()[0]
            print(df[name].isnull().sum())
        else:
            print(f"{name} has no missing values")

    def outliers_management(self, df, name):
        print("Outliers management is not applicable for categorical variables.")

    def plot_distribution(self, df, name):
        plt.figure(figsize=(8,5))
        sns.countplot(data=df, x=name, order=df[name].value_counts().index)
        plt.show()

    def description(self, df, name):
        print(df[name].describe())
        print(df[name].value_counts())