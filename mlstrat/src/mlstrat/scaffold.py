"""Generate an empty ML project structure (folders + starter files)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

Tree = Dict[str, Union[str, None, "Tree"]]  # str = file content, None = empty folder, dict = folder


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
        "README.md": f"# {name}\n\nDescription du projet.\n\n"
        "## Structure\n\n"
        "- `data/` : données brutes (`raw`), intermédiaires (`interim`), prêtes (`processed`)\n"
        "- `notebooks/` : exploration et essais\n"
        "- `src/` : code du projet (données, features, modèles, pipelines)\n"
        "- `configs/` : paramètres\n"
        "- `reports/` : figures et métriques\n"
        "- `tests/` : tests\n",
        "pyproject.toml": "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.9"\n'
        'dependencies = ["mlstrat"]\n\n'
        "[tool.setuptools.packages.find]\n"
        'include = ["src*"]\n',
        "requirements.txt": "mlstrat\n",
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
            "config.yaml": "# Configuration générale\ndata_path: data/raw/\n",
            "model_params.yaml": "# Hyperparamètres du modèle\n",
            "pipeline_config.yaml": "# Configuration des pipelines (ZenML, MLflow...)\n",
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
