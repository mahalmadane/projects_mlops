import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class StrategieAnalyses(ABC):
    
    @abstractmethod
    def missing(self, df:pd.DataFrame):
        pass

    @abstractmethod
    def descibtion(self, df:pd.DataFrame):
        pass 

    @abstractmethod
    def type(self, df:pd.DataFrame):
        pass

class StrategieMultipleValues(StrategieAnalyses):

    def missing(self, df:pd.DataFrame):
        print(df.isnull().sum()) 

    def descibtion(self, df:pd.DataFrame):
        print(df.describe())

    def type(self, df:pd.DataFrame):
        print(df.dtypes)