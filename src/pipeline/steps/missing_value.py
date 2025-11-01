from zenml import step
import pandas as pd
from preprocessing_strategie.missing_value import MissingValue


@step
def missing_value(df: pd.DataFrame) -> pd.DataFrame:
    mv = MissingValue(df)
    mv.impute("Age", "mean").impute("Embarked", "mode").impute("Cabin", "mode").impute_rest_features()
    df = mv.get_df()
    return df
