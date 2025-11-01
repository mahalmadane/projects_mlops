import pandas as pd 
from zenml import step
from preprocessing_strategie.outlier_detection import OutlierDetection

@step
def outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    od = OutlierDetection(df)
    od.outliers_traitement("Age").outlier_rest_traitement()
    return od.get_df()