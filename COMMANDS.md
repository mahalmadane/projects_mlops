# Commandes d'exécution

## 0. Tout lancer en une commande

`run_all.py` (à la racine) démarre MLflow, l'API FastAPI et l'interface Streamlit dans le
bon ordre, en attendant que chaque service réponde avant de lancer le suivant :

```bash
python3 run_all.py
```

Options : `--no-mlflow`, `--no-api`, `--no-streamlit` pour désactiver un service, `--zenml`
pour démarrer aussi le dashboard ZenML local avant le reste. `Ctrl+C` arrête proprement tous
les services lancés par le script.

Prérequis : les venvs `env/` et `env_streamlit/` doivent déjà exister (voir sections
suivantes) et un modèle doit avoir l'alias `champion` dans le registre MLflow (sinon l'API
ne démarre pas). Les commandes détaillées ci-dessous restent utiles pour lancer les services
un par un (débogage, développement avec `--reload`, etc.).

## 1. Activer l'environnement virtuel

```bash
source env/bin/activate
```

## 2. Variable d'environnement macOS (ZenML)

Requis sur Mac pour lancer le serveur ZenML local (fork-safety). À ajouter une fois pour
toutes dans `~/.zshrc` :

```bash
echo 'export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES' >> ~/.zshrc
```

## 3. Lancer le serveur ZenML local (dashboard)

```bash
zenml login --local
```

Dashboard disponible sur http://127.0.0.1:8237

Pour l'arrêter :

```bash
zenml logout --local
```

## 4. Lancer le serveur MLflow (tracking + registry)

```bash
mlflow server --host 0.0.0.0 --port 5001
```

Le pipeline d'entraînement (`src/pipeline/steps/training.py`) est codé pour utiliser
`http://localhost:5001` — le serveur MLflow doit tourner sur ce port avant de lancer
le pipeline.

## 5. Lancer le pipeline d'entraînement

```bash
python src/pipeline/pipeline.py
```

Étapes exécutées : chargement des données → imputation des valeurs manquantes →
traitement des outliers → split train/test → entraînement (LogisticRegression +
KNeighborsClassifier) → évaluation. Chaque run est loggé dans MLflow sous
l'expérience `"ML Ops Experiment"`.

## 6. Lancer l'API de prédiction

```bash
uvicorn api.app:app --reload
```

Nécessite que le serveur MLflow (étape 4) tourne déjà sur `http://localhost:5001` — `api/app.py`
s'y connecte au démarrage (`mlflow.set_tracking_uri`) pour résoudre les artefacts du modèle.
Nécessite aussi qu'un modèle ait l'alias `champion` dans le registre MLflow, sinon l'API
échoue au démarrage.

## 7. Tester l'API

Avec l'API lancée (étape 6) dans un autre terminal :

```bash
python api/test_api.py
```

## 8. Réinstaller le package en mode éditable (si `ModuleNotFoundError: preprocessing_strategie`)

```bash
pip install -e . --no-deps
```

## 9. Lancer l'interface Streamlit

Streamlit vit dans son **propre venv** (`env_streamlit/`), séparé de `env/` : les dernières
versions de Streamlit exigent une version de `starlette` incompatible avec la version de
FastAPI pinnée dans `requirements.txt`. Les deux processus communiquent uniquement en HTTP,
donc pas besoin de partager l'environnement.

Créer le venv (une seule fois) :

```bash
python3 -m venv env_streamlit
env_streamlit/bin/pip install streamlit requests
```

Lancer l'interface (l'API — étape 6 — doit déjà tourner) :

```bash
env_streamlit/bin/streamlit run streamlit_app/app.py
```

L'URL de l'API est modifiable dans la barre latérale de l'interface (par défaut
`http://127.0.0.1:8000`).
