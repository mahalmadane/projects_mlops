import os
import sys
import pandas as pd
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from zenml import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.pipeline.steps.load import load_data
from src.pipeline.steps.splitter import split
from src.pipeline.steps.training import training_LogisticRegression, training_KNeighborsClassifier
from src.pipeline.steps.evaluation import evaluation
from src.pipeline.steps.missing_value import missing_value
from src.pipeline.steps.outlier_detection import outlier_detection

logging.basicConfig(level=logging.INFO)

@pipeline(name="my_clean_training_pipeline")
def my_pipeline():
    # Chargement des données
    df = load_data()

    # Missing values management
    df_imputed = missing_value(df)

    # Outlier management
    df_outliers = outlier_detection(df_imputed)

    # Split
    x_train, x_test, y_train, y_test = split(
        df_outliers,
        target_column="Survived",
        exclude_columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    list_params_lg = [
        {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'},
    ]

    reports_lg = []  # liste pour stocker les rapports
    for i, params_lg in enumerate(list_params_lg):
        run_name = f"LogisticRegression_run_{i+1}"
        lgmodel, run_id_lg = training_LogisticRegression(
            x_train, y_train, params=params_lg, run_name=run_name
        )
        report_lg = evaluation(x_test, y_test, lgmodel, run_id_lg)
        reports_lg.append(report_lg)

    # -----------------------------
    # KNeighborsClassifier
    # -----------------------------
    list_params_knn = [
        {'n_neighbors': 3, 'weights': 'uniform', 'algorithm': 'auto'},
    ]

    reports_knn = []  # liste pour stocker les rapports
    for i, params_knn in enumerate(list_params_knn):
        run_name = f"KNeighborsClassifier_run_{i+1}"
        knnmodel, run_id_knn = training_KNeighborsClassifier(
            x_train, y_train, params=params_knn, run_name=run_name
        )
        report_knn = evaluation(x_test, y_test, knnmodel, run_id_knn)
        reports_knn.append(report_knn)

    # Retour optionnel des listes si besoin
    return reports_lg, reports_knn


if __name__ == "__main__":
    pipeline_run = my_pipeline()
    logging.info("success pipeline run")

