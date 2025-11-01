import os
import sys
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from zenml import pipeline
from src.pipeline.steps.load import split, load_data
from src.pipeline.steps.training import training_Lg, training_Kn
from src.pipeline.steps.evaluation import evaluation
from src.pipeline.steps.missing_value import missing_value
import pandas as pd



@pipeline(name="my_clean_training_pipeline")
def my_pipeline():

    # Chargement des données
    df = load_data()

    # missing values management
    df = missing_value(df)

    # Outlier management

    # # Split
    # x_train, x_test, y_train, y_test = split(df)  # ZenML décompose automatiquement le NamedTuple

    # # Training
    # lgmodel = training_Lg(x_train, y_train)
    # knmodel = training_Kn(x_train, y_train)

    # # Evaluation
    # accuracy_bestScore = evaluation(lgmodel, knmodel, x_test, y_test)
    # return accuracy_bestScore



if __name__ == "__main__":
    pipeline_run = my_pipeline() 