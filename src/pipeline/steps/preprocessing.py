from zenml import step
import pandas as pd
from typing import Tuple


@step
def preprocess(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple preprocessing step that one-hot-encodes categorical
    columns and ensures train/test have the same columns.

    This is a lightweight approach suitable for quick experiments. For
    production use, prefer an sklearn Pipeline / ColumnTransformer with
    persisted encoders.
    """
    # Remember original indices so we can split back after concat
    n_train = len(X_train)

    combined = pd.concat([X_train, X_test], axis=0)

    # One-hot encode categorical/object columns
    combined_encoded = pd.get_dummies(combined, drop_first=False)

    X_train_enc = combined_encoded.iloc[:n_train].reset_index(drop=True)
    X_test_enc = combined_encoded.iloc[n_train:].reset_index(drop=True)

    return X_train_enc, X_test_enc
