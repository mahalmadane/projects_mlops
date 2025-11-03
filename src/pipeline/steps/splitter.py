from zenml import step
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


@step(name="split step")
def split(
    df: pd.DataFrame,
    target_column: str ,
    exclude_columns: list = [],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into X_train, X_test, y_train, y_test and return them
    as multiple outputs so ZenML treats them as separate step artifacts.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=exclude_columns),
        df[target_column],
        test_size=test_size,
        random_state=random_state,
    )

    # Return as a tuple so the step produces multiple outputs
    return x_train, x_test, y_train, y_test
