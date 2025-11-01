import pandas as pd
import logging

# Config logging global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OutlierDetection:
    """
    Classe pour détecter et traiter les outliers d'un DataFrame.
    Les valeurs en dehors des bornes [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 
    sont remplacées par les bornes correspondantes.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # éviter de modifier le DataFrame original

    def outliers_traitement(self, column: str):
        """
        Traite les outliers d'une colonne numérique.
        """
        if column not in self.df.columns:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame")
        
        if pd.api.types.is_numeric_dtype(self.df[column]):
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            logging.info(f"Outliers de la colonne '{column}' traités")
        return self  # permet le chaining

    def outlier_rest_traitement(self):
        """
        Traite les outliers pour toutes les colonnes numériques du DataFrame.
        """
        for column in self.df.select_dtypes(include='number').columns:
            self.outliers_traitement(column)
        return self  # permet le chaining

    def get_df(self):
        """Retourne le DataFrame après traitement des outliers"""
        return self.df
