from zenml import step
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

@step
def load_data() -> pd.DataFrame:
    project_root = os.path.dirname(os.getcwd())  # parent du notebook
    candidate = os.path.join(project_root, 'data', 'processed', 'data_processed.csv')
    
    if not os.path.exists(candidate):
        found = None
        for root, _, files in os.walk(project_root):
            if 'data_processed.csv' in files:
                found = os.path.join(root, 'data_processed.csv')
                break
        if found:
            candidate = found
        else:
            raise FileNotFoundError(f"data_processed.csv not found in {project_root}")
    
    df = pd.read_csv(candidate)
    return df


@step
def split(
    df: pd.DataFrame,
    target_column: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into X_train, X_test, y_train, y_test and return them
    as multiple outputs so ZenML treats them as separate step artifacts.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]),
        df[target_column],
        test_size=test_size,
        random_state=random_state,
    )

    # Return as a tuple so the step produces multiple outputs
    return x_train, x_test, y_train, y_test
