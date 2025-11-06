from zenml import step
import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import mlflow
from typing import Tuple

logging.basicConfig(level=logging.INFO)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ML Ops Experiment")


@step(name="LogisticRegression step")
def training_LogisticRegression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    run_name: str,
)-> Tuple[object,str]:
    with mlflow.start_run(run_name=run_name) as run:

        run_id=run.info.run_id
        mlflow.log_params(params)

        # pipelines categories
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        logging.info(cat_pipeline)

        # pipelines numeriques
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        logging.info(num_pipeline)
        num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns

        # column transformer
        processor = make_column_transformer(
            (cat_pipeline, cat_cols),
            (num_pipeline, num_cols)
        )

        # pipelines final
        pipe = Pipeline([
            ('preprocessor', processor),
            ('classifier', LogisticRegression(**params))
        ])

        # training
        pipe.fit(X_train, y_train)
        mlflow.sklearn.log_model(sk_model=pipe, name="model LogisticRegression", 
                                 input_example=X_train.head(1),
                                 registered_model_name="LogisticRegression")
        logging.info("Model LogisticRegression trained successfully")
        return pipe,run_id

@step(name="KNeighborsClassifier step")
def training_KNeighborsClassifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    run_name: str,
)-> Tuple[object,str]:
    with mlflow.start_run(run_name=run_name) as run:

        run_id=run.info.run_id
        mlflow.log_params(params)

        # pipelines categories
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        logging.info(cat_pipeline)

        # pipelines numeriques
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        logging.info(num_pipeline)
        num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns

        # column transformer
        processor = make_column_transformer(
            (cat_pipeline, cat_cols),
            (num_pipeline, num_cols)
        )

        # pipelines final
        pipe = Pipeline([
            ('preprocessor', processor),
            ('classifier', KNeighborsClassifier(**params))
        ])

        # training
        pipe.fit(X_train, y_train)
        mlflow.sklearn.log_model( sk_model=pipe,name="model",input_example=X_train.head(1),
                                 registered_model_name="KNeighborsClassifier")
        logging.info("Model KNeighborsClassifier trained successfully")
        return pipe,run_id