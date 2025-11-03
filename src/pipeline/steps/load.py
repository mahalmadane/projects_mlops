from zenml import step
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

@step(name="load step")
def load_data() -> pd.DataFrame:
    project_root = os.path.dirname(os.getcwd())  # parent du notebook
    candidate = os.path.join(project_root, 'data', 'raw', 'titanic.csv')
    
    if not os.path.exists(candidate):
        found = None
        for root, _, files in os.walk(project_root):
            if 'titanic.csv' in files:
                found = os.path.join(root, 'titanic.csv')
                break
        if found:
            candidate = found
        else:
            raise FileNotFoundError(f"titanic.csv not found in {project_root}")
    
    df = pd.read_csv(candidate)
    return df
