import os

def create_project_structure(project_name):
    # Définir la structure des dossiers
    structure = {
        "README.md": "# " + project_name + "\n\nExplication du projet.",
        "requirements.txt": "# Liste des dépendances",
        "setup.py": """# setup.py
from setuptools import setup, find_packages

setup(
    name='{}',
    version='0.1',
    packages=find_packages(),
    install_requires=[]  # Ajoutez les dépendances ici
)
""".format(project_name),
        ".gitignore": """# Fichiers à ignorer par Git
*.pyc
__pycache__/
*.log
*.csv
*.sqlite
""",
        "data/raw": [],
        "data/interim": [],
        "data/processed": [],
        "notebooks": [
            "01_exploration.ipynb",
            "02_preprocessing.ipynb",
            "03_model_training.ipynb"
        ],
        "src": {
            "init.py": "",
            "data": {
                "load_data.py": "# Code pour charger les données",
                "preprocess.py": "# Code pour prétraiter les données"
            },
            "features": {
                "build_features.py": "# Code pour l'ingénierie des features"
            },
            "models": {
                "train_model.py": "# Code pour entraîner le modèle",
                "evaluate.py": "# Code pour évaluer le modèle",
                "predict.py": "# Code pour faire des prédictions"
            },
            "utils": {
                "helpers.py": "# Fonctions utilitaires"
            },
            "pipelines": {
                "init.py": "",
                "text_correction_pipeline.py": "# Pipeline pour la correction de texte",
                "steps": {
                    "load_step.py": "# Étape pour charger les données",
                    "train_step.py": "# Étape pour entraîner le modèle",
                    "eval_step.py": "# Étape pour évaluer le modèle"
                }
            }
        },
        "tests": [
            "test_data.py",
            "test_model.py"
        ],
        "reports": {
            "figures": [],
            "metrics.json": "{}"
        },
        "configs": {
            "config.yaml": "# Fichier de configuration générale",
            "model_params.yaml": "# Hyperparamètres du modèle",
            "pipeline_config.yaml": "# Configurations pour ZenML/MLflow"
        },
        "scripts": [
            "run_pipeline.py",
            "train_local.py",
            "deploy_model.py"
        ]
    }

    # Fonction récursive pour créer les dossiers et fichiers
    def create_files_and_folders(base_path, structure):
        for name, value in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(value, dict):
                # Créer un dossier et appeler la fonction récursivement
                os.makedirs(path, exist_ok=True)
                create_files_and_folders(path, value)
            elif isinstance(value, list):
                # Créer les fichiers dans le dossier
                os.makedirs(path, exist_ok=True)
                for file_name in value:
                    with open(os.path.join(path, file_name), 'w') as f:
                        f.write("# " + file_name)
            else:
                # Créer un fichier avec le contenu spécifié
                with open(path, 'w') as f:
                    f.write(value)
    
    # Créer le projet à la racine
    os.makedirs(project_name, exist_ok=True)
    create_files_and_folders(project_name, structure)
    print(f"Le projet {project_name} a été créé avec succès !")

# Exemple d'utilisation
create_project_structure("my_ml_project")
