from zenml import step
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

@step
def training_Lg(X_train: pd.DataFrame, y_train: pd.Series):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

@step
def training_Kn(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors: int = 5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
    