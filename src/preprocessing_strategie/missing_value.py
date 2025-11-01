import pandas as pd
import logging

# Config logging global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MissingValue:
    """
    Classe pour gérer les valeurs manquantes d'un DataFrame.
    Peut imputer une colonne spécifique ou toutes les colonnes automatiquement.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # pour éviter de modifier le DataFrame original
    
    def impute(self, feature: str, strategy: str = 'mean'):
        """
        Impute les valeurs manquantes d'une colonne selon la stratégie choisie.
        Colonnes catégorielles -> mode
        Colonnes numériques -> mean/median/mode/min/max
        """
        if feature not in self.df.columns:
            raise ValueError(f"La colonne '{feature}' n'existe pas dans le DataFrame")
        
        # Colonnes catégorielles / object / category
        if self.df[feature].dtype in ['object', 'category']:
            value = self.df[feature].mode()[0]
            self.df[feature].fillna(value, inplace=True)
            logging.info(f"Valeurs manquantes de '{feature}' remplacées par le mode: {value}")
        else:  # Colonnes numériques
            strategies = {
                'mean': self.df[feature].mean(),
                'median': self.df[feature].median(),
                'mode': self.df[feature].mode()[0],
                'min': self.df[feature].min(),
                'max': self.df[feature].max()
            }
            if strategy not in strategies:
                raise ValueError("Strategy non supportée. Choisir: mean, median, mode, min, max.")
            
            value = strategies[strategy]
            self.df[feature].fillna(value, inplace=True)
            logging.info(f"Valeurs manquantes de '{feature}' remplacées par ({strategy}) : {value}")
        
        return self  # permet le chaining
    
    def impute_rest_features(self, strategies: dict = None):
        """
        Impute toutes les colonnes avec des valeurs manquantes.
        On peut passer un dictionnaire de stratégies par colonne si besoin.
        Ex: {'Age': 'mean', 'Fare': 'median'}
        """
        for feature in self.df.columns:
            if self.df[feature].isnull().sum() > 0:
                # Utiliser la stratégie spécifique si définie
                strat = strategies.get(feature, 'mode') if strategies else None
                
                if self.df[feature].dtype in ['object', 'category']:
                    self.impute(feature, strategy='mode')
                else:
                    self.impute(feature, strategy=strat or 'mean')
        
        return self  # permet le chaining

    def get_df(self) -> pd.DataFrame:
        """Retourne le DataFrame imputé"""
        return self.df
