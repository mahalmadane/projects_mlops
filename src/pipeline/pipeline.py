import os
import sys
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from zenml import pipeline
from src.pipeline.steps.load import split, load_data
from src.pipeline.steps.training import training_Lg, training_Kn
from src.pipeline.steps.evaluation import evaluation
from src.pipeline.steps.preprocessing import preprocess
import pandas as pd



@pipeline
def my_pipeline():
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)  # ZenML décompose automatiquement le NamedTuple
    lgmodel = training_Lg(x_train, y_train)
    knmodel = training_Kn(x_train, y_train)
    accuracy_bestScore = evaluation(lgmodel, knmodel, x_test, y_test)
    return accuracy_bestScore



if __name__ == "__main__":
    pipeline_run = my_pipeline() 