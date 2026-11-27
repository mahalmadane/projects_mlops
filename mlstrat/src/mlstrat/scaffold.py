"""Generate an empty ML project structure (folders + starter files)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

Tree = Dict[str, Union[str, None, "Tree"]]  # str = file content, None = empty folder, dict = folder


REQUIREMENTS_TEMPLATE = """\
# Dépendances du projet.
#
# Rien n'est installé automatiquement : `pip install -e .` installe seulement le code de src/.
# Installe les dépendances toi-même :
#
#     pip install -r requirements.txt
#
# Commente (#) les lignes dont tu n'as pas besoin, et décommente les optionnelles.
# Pour figer les versions qui marchent :  pip freeze > requirements.txt

# --- Outils d'analyse et d'entraînement (EDA, nettoyage, modèles) ---
mlstrat
numpy
pandas
scikit-learn
matplotlib
seaborn

# --- Lecture des fichiers configs/*.yaml ---
pyyaml

# --- Notebooks ---
jupyter

# --- MLOps : pipelines et suivi des expériences ---
zenml
mlflow

# --- Tests ---
pytest

# --- Optionnel : servir le modèle (API) et interface web ---
# fastapi
# uvicorn
# streamlit
"""

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
# Volontairement vide : les dépendances sont dans requirements.txt
# (pip install -r requirements.txt). Ainsi `pip install -e .` ne fait que
# rendre le code de src/ importable, sans rien télécharger.
dependencies = []

[tool.setuptools.packages.find]
include = ["src*"]
"""

README_TEMPLATE = """\
# {name}

Description du projet : objectif, données utilisées, métrique visée.

## Installation

Prérequis : Python 3.10 ou plus récent.

1. Créer et activer un environnement virtuel (une seule fois) :

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows : .venv\\Scripts\\activate
   ```

2. Installer les dépendances listées dans `requirements.txt` :

   ```bash
   pip install -r requirements.txt
   ```

   Rien n'est installé automatiquement : ouvre `requirements.txt` et commente les
   outils dont tu n'as pas besoin (par exemple `zenml` ou `mlflow`) avant de lancer
   la commande.

3. Rendre le code de `src/` importable depuis les notebooks et les scripts :

   ```bash
   pip install -e .
   ```

   Cette commande n'installe aucune dépendance, seulement le projet lui-même.

## Utilisation

1. Mettre les données brutes dans `data/raw/` (elles ne sont pas versionnées par git).
2. Explorer avec `notebooks/01_exploration.ipynb` :

   ```bash
   jupyter notebook
   ```

3. Écrire le chargement et le nettoyage dans `src/data/`, l'entraînement dans `src/models/`.
4. Lancer le pipeline complet :

   ```bash
   python scripts/run_pipeline.py
   ```

5. Lancer les tests :

   ```bash
   pytest
   ```

## Structure

```
{name}/
├── data/
│   ├── raw/            données d'origine, jamais modifiées
│   ├── interim/        données en cours de transformation
│   └── processed/      données prêtes pour l'entraînement
├── notebooks/          exploration, prétraitement, essais de modèles
├── src/
│   ├── data/           load_data.py (chargement), preprocess.py (nettoyage)
│   ├── features/       build_features.py (création de variables)
│   ├── models/         train_model.py, evaluate.py, predict.py
│   ├── utils/          fonctions utilitaires
│   └── pipelines/      training_pipeline.py et ses étapes (steps/)
├── configs/            config.yaml, model_params.yaml, pipeline_config.yaml
├── reports/            figures/ et metrics.json
├── scripts/            run_pipeline.py, train_local.py, deploy_model.py
├── tests/              tests pytest
├── requirements.txt    dépendances à installer
└── pyproject.toml      rend src/ installable (pip install -e .)
```

Projet créé avec [mlstrat](https://pypi.org/project/mlstrat/).
"""


CONFIG_TEMPLATE = """\
# =============================================================================
# config.yaml : configuration générale du projet
# =============================================================================
# Tout est en commentaire : décommente (enlève le "# ") et adapte ce dont tu as
# besoin. Tant que tout reste commenté, le fichier est valide mais vide.
#
# --- Exemple ---------------------------------------------------------------
#
# data:
#   raw_path: data/raw/my_data.csv          # fichier d'origine
#   processed_path: data/processed/data.csv # sortie du nettoyage
#   target: target                          # colonne à prédire
#   drop_columns: [id, name]                # colonnes inutiles pour le modèle
#
# split:
#   test_size: 0.2                          # 20 % des lignes pour le test
#   random_state: 42                        # même découpage à chaque run
#   stratify: true                          # garde la proportion des classes
#
# preprocessing:
#   missing_values:                         # stratégie par colonne
#     age: median                           # mean | median | mode | min | max
#     city: Unknown                         # ou une valeur constante
#                                           # les autres colonnes : mean (numérique) ou mode
#   outliers:
#     factor: 1.5                           # bornes Q1 - 1.5*IQR / Q3 + 1.5*IQR
#     exclude: [target]                     # colonnes à ne pas toucher
#
# mlflow:
#   tracking_uri: http://localhost:5000     # serveur lancé avec `mlflow server`
#   experiment_name: mon_experience
#
# --- Utilisation en Python -------------------------------------------------
# (nécessite pyyaml : pip install pyyaml, déjà dans requirements.txt)
#
# import yaml
# import pandas as pd
# from mlstrat import MissingValueImputer, OutlierClipper
#
# with open("configs/config.yaml") as f:
#     cfg = yaml.safe_load(f) or {}
#
# df = pd.read_csv(cfg["data"]["raw_path"])
# df = df.drop(columns=cfg["data"]["drop_columns"])
#
# prep = cfg["preprocessing"]
# df = MissingValueImputer(df).impute_all(prep["missing_values"]).get_df()
# df = (
#     OutlierClipper(df, factor=prep["outliers"]["factor"])
#     .clip_all(exclude=prep["outliers"]["exclude"])
#     .get_df()
# )
# df.to_csv(cfg["data"]["processed_path"], index=False)
#
# # Découpage train / test
# from sklearn.model_selection import train_test_split
# target = cfg["data"]["target"]
# X, y = df.drop(columns=target), df[target]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=cfg["split"]["test_size"],
#     random_state=cfg["split"]["random_state"],
#     stratify=y if cfg["split"]["stratify"] else None,
# )
#
# # MLflow
# import mlflow
# mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
# mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
"""

MODEL_PARAMS_TEMPLATE = """\
# =============================================================================
# model_params.yaml : modèles à entraîner et leurs hyperparamètres
# =============================================================================
# Tout est en commentaire : décommente et adapte. Chaque entrée de "models"
# donne la classe scikit-learn ("class") et ses paramètres ("params").
# Une liste de jeux de paramètres = un entraînement par jeu (pour comparer).
#
# --- Exemple ---------------------------------------------------------------
#
# models:
#   logistic_regression:
#     class: sklearn.linear_model.LogisticRegression
#     params:
#       - {C: 0.1, solver: liblinear}       # C petit = modèle plus simple
#       - {C: 1.0, max_iter: 1000}
#
#   knn:
#     class: sklearn.neighbors.KNeighborsClassifier
#     params:
#       - {n_neighbors: 3, weights: uniform}
#       - {n_neighbors: 7, weights: distance}
#
#   random_forest:
#     class: sklearn.ensemble.RandomForestClassifier
#     params:
#       - {n_estimators: 200, max_depth: 8, random_state: 42}
#
# metric: f1_score        # métrique pour choisir le meilleur modèle
#                         # accuracy | f1_score | precision | recall
#
# --- Utilisation en Python -------------------------------------------------
#
# import importlib
# import yaml
# from mlstrat import ModelTrainer
#
# with open("configs/model_params.yaml") as f:
#     cfg = yaml.safe_load(f) or {}
#
# def load_class(path):                     # "sklearn.svm.SVC" -> classe SVC
#     module, name = path.rsplit(".", 1)
#     return getattr(importlib.import_module(module), name)
#
# results = []
# for model_name, spec in cfg["models"].items():
#     model_class = load_class(spec["class"])
#     for params in spec["params"]:
#         trainer = ModelTrainer(model_class, **params).fit(X_train, y_train)
#         scores = trainer.evaluate(X_test, y_test)
#         results.append((model_name, params, scores, trainer))
#
# best = max(results, key=lambda r: r[2][cfg["metric"]])
# print("Meilleur modèle :", best[0], best[1], best[2])
"""

PIPELINE_CONFIG_TEMPLATE = """\
# =============================================================================
# pipeline_config.yaml : configuration d'un run de pipeline ZenML
# =============================================================================
# Tout est en commentaire : décommente et adapte. Ce fichier suit le format
# de configuration de ZenML : n'y ajoute pas de clés inventées (ZenML les
# refuserait). Les réglages MLflow sont dans config.yaml.
#
# --- Exemple ---------------------------------------------------------------
#
# run_name: training_{date}_{time}   # nom du run dans le dashboard ZenML
# enable_cache: false                # true = réutilise les étapes déjà calculées
#
# parameters:                        # arguments de la fonction @pipeline
#   data_path: data/raw/my_data.csv
#   target: target
#
# steps:                             # réglages étape par étape
#   load_step:
#     enable_cache: true             # le chargement peut rester en cache
#   train_step:
#     parameters:                    # arguments de la fonction @step
#       n_neighbors: 5
#   eval_step:
#     enable_cache: false
#
# --- Utilisation en Python -------------------------------------------------
# (dans src/pipelines/training_pipeline.py)
#
# from zenml import pipeline, step
#
# @step
# def load_step(data_path: str) -> pd.DataFrame:
#     return pd.read_csv(data_path)
#
# @step
# def train_step(df: pd.DataFrame, target: str, n_neighbors: int = 3): ...
#
# @pipeline
# def training_pipeline(data_path: str, target: str):
#     df = load_step(data_path)
#     train_step(df, target)
#
# if __name__ == "__main__":
#     training_pipeline.with_options(
#         config_path="configs/pipeline_config.yaml"
#     )()
#
# Les valeurs de "parameters" et "steps.<nom>.parameters" sont passées
# automatiquement aux fonctions : pas besoin de les écrire dans l'appel.
"""


def _notebook(title: str) -> str:
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import pandas as pd\n", "from mlstrat import DataOverview"],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1) + "\n"


def project_tree(name: str) -> Tree:
    """The files and folders created by :func:`create_project`."""
    return {
        "README.md": README_TEMPLATE.replace("{name}", name),
        "pyproject.toml": PYPROJECT_TEMPLATE.replace("{name}", name),
        "requirements.txt": REQUIREMENTS_TEMPLATE,
        ".gitignore": "__pycache__/\n*.py[cod]\n*.log\n.venv/\nenv/\n.ipynb_checkpoints/\n"
        "data/**/*\n!data/**/.gitkeep\n",
        "data": {"raw": None, "interim": None, "processed": None},
        "notebooks": {
            "01_exploration.ipynb": _notebook("Exploration"),
            "02_preprocessing.ipynb": _notebook("Preprocessing"),
            "03_model_training.ipynb": _notebook("Model training"),
        },
        "src": {
            "__init__.py": "",
            "data": {
                "__init__.py": "",
                "load_data.py": '"""Chargement des données."""\n\nimport pandas as pd\n\n\n'
                "def load_data(path: str) -> pd.DataFrame:\n    return pd.read_csv(path)\n",
                "preprocess.py": '"""Nettoyage des données."""\n\nimport pandas as pd\n'
                "from mlstrat import MissingValueImputer, OutlierClipper\n\n\n"
                "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    df = MissingValueImputer(df).impute_all().get_df()\n"
                "    return OutlierClipper(df).clip_all().get_df()\n",
            },
            "features": {
                "__init__.py": "",
                "build_features.py": '"""Ingénierie des features."""\n',
            },
            "models": {
                "__init__.py": "",
                "train_model.py": '"""Entraînement du modèle."""\n\n'
                "from mlstrat import ModelTrainer\n\n\n"
                "def train_model(model, X_train, y_train, **params) -> ModelTrainer:\n"
                "    return ModelTrainer(model, **params).fit(X_train, y_train)\n",
                "evaluate.py": '"""Évaluation du modèle."""\n\n\n'
                "def evaluate(trainer, X_test, y_test) -> dict:\n"
                "    return trainer.evaluate(X_test, y_test)\n",
                "predict.py": '"""Prédictions."""\n\n\n'
                "def predict(trainer, X):\n    return trainer.predict(X)\n",
            },
            "utils": {"__init__.py": "", "helpers.py": '"""Fonctions utilitaires."""\n'},
            "pipelines": {
                "__init__.py": "",
                "training_pipeline.py": '"""Pipeline d\'entraînement."""\n',
                "steps": {
                    "__init__.py": "",
                    "load_step.py": '"""Étape de chargement des données."""\n',
                    "train_step.py": '"""Étape d\'entraînement."""\n',
                    "eval_step.py": '"""Étape d\'évaluation."""\n',
                },
            },
        },
        "tests": {
            "__init__.py": "",
            "test_data.py": "def test_placeholder():\n    assert True\n",
            "test_model.py": "def test_placeholder():\n    assert True\n",
        },
        "reports": {"figures": None, "metrics.json": "{}\n"},
        "configs": {
            "config.yaml": CONFIG_TEMPLATE,
            "model_params.yaml": MODEL_PARAMS_TEMPLATE,
            "pipeline_config.yaml": PIPELINE_CONFIG_TEMPLATE,
        },
        "scripts": {
            "run_pipeline.py": '"""Lance le pipeline complet."""\n',
            "train_local.py": '"""Entraînement en local."""\n',
            "deploy_model.py": '"""Déploiement du modèle."""\n',
        },
    }


def _write(base: Path, tree: Tree, created: List[Path], skipped: List[Path]) -> None:
    for name, value in tree.items():
        path = base / name
        if isinstance(value, dict):
            path.mkdir(parents=True, exist_ok=True)
            _write(path, value, created, skipped)
        elif value is None:
            path.mkdir(parents=True, exist_ok=True)
            keep = path / ".gitkeep"
            if not keep.exists():
                keep.touch()
                created.append(keep)
        elif path.exists():
            skipped.append(path)
        else:
            path.write_text(value, encoding="utf-8")
            created.append(path)


@dataclass
class ScaffoldResult:
    """What :func:`create_project` did."""

    root: Path
    created: List[Path] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)  # files that already existed


def create_project(name: str, parent: Union[str, Path] = ".") -> ScaffoldResult:
    """Create the project folder ``parent/name`` with the standard structure.

    Existing files are never overwritten, so it is safe to run on an existing
    project to add the missing pieces.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            f"Invalid project name '{name}': use letters, digits, '.', '_' or '-' (no spaces or slashes)"
        )
    result = ScaffoldResult(root=Path(parent) / name)
    result.root.mkdir(parents=True, exist_ok=True)
    _write(result.root, project_tree(name), result.created, result.skipped)
    return result
