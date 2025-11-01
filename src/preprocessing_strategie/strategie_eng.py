import pandas as pd

class StrategieSelectionFeature:

    def select_features(self, df: pd.DataFrame, path: str, features: list=None):
        if features is not None:
            df = df[features]
        df.to_csv(path)