"""Thin training/evaluation wrapper around any scikit-learn classifier."""

from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlstrat.modeling.preprocessor import make_model_pipeline


class ModelTrainer:
    """Train, evaluate and predict with a classifier.

    Args:
        model: an estimator instance (``LogisticRegression(C=0.1)``) or class
            (``LogisticRegression``), in which case ``**params`` are passed to it.
        preprocess: wrap the model in :func:`make_model_pipeline` so raw mixed
            DataFrames can be passed directly.
        **params: hyperparameters, used only when ``model`` is a class.

    Example:
        >>> trainer = ModelTrainer(LogisticRegression, C=0.1, max_iter=1000)
        >>> trainer.fit(X_train, y_train).evaluate(X_test, y_test)
        {'accuracy': 0.81, 'f1_score': 0.80, 'precision': 0.81, 'recall': 0.81}
    """

    def __init__(self, model, preprocess: bool = True, **params):
        if isinstance(model, type):
            model = model(**params)
        elif params:
            model.set_params(**params)
        if not isinstance(model, BaseEstimator):
            raise TypeError("model must be a scikit-learn estimator or estimator class")
        self.model = make_model_pipeline(model) if preprocess else model

    def fit(self, X, y) -> "ModelTrainer":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, average: str = "weighted") -> Dict[str, float]:
        """Accuracy, F1, precision and recall on ``(X, y)``."""
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_score": float(f1_score(y, y_pred, average=average, zero_division=0)),
            "precision": float(precision_score(y, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y, y_pred, average=average, zero_division=0)),
        }
