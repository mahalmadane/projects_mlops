from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from typing import Optional
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri("http://localhost:5001")

client=MlflowClient()

# Nom du modèle MLflow et version
models_names = ["KNeighborsClassifier", "LogisticRegression"]
best_model_alias = "champion"

for model in models_names:
    try: 
        model_info=client.get_model_version_by_alias(name=model, alias=best_model_alias)
        model_name=model_info.name
        break
    except MlflowException as e:
        print(f"Alias '{best_model_alias}' not found for model '{model}'. Exception: {str(e)}")


# Chargement du modèle depuis MLflow
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@{best_model_alias}")

# Création de l'app FastAPI
app = FastAPI(title="Titanic Prediction API")

# Définir le schéma des données d'entrée avec Pydantic
class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: float
    Parch: int
    Fare: float
    Embarked: str

# Endpoint racine
@app.get("/")
async def root():
    return {"message": "Titanic Prediction API is running"}

# Endpoint de prédiction
@app.post("/predict")
async def predict(input_data: TitanicInput):
    # Convertir en DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Faire la prédiction
    prediction = model.predict(df)
    
    # Retourner la prédiction
    return {"prediction": int(prediction[0]), "model": model_name}
