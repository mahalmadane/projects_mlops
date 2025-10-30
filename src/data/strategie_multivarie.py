import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class StrategieMultivarie(ABC):

    @abstractmethod
    def plot_relationship(self, df:pd.DataFrame, names:list):
        pass