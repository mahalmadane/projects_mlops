# mlstrat

Start a machine learning project in one command, then explore, clean and model tabular
data with ready-made tools built on pandas and scikit-learn.

Install from GitHub (requires git):

```bash
pip install "mlstrat @ git+https://github.com/mahalmadane/projects_mlops.git#subdirectory=mlstrat"
```

To pin a version, add a tag or commit after `.git`, e.g. `projects_mlops.git@mlstrat-v0.1.0#subdirectory=mlstrat`.

## Guide

A step-by-step guide (in French) ships with the package and opens offline in your browser:

```bash
mlstrat docs
```

If the `mlstrat` command is not found (pip's script folder is not on your PATH), use
`python3 -m mlstrat` instead, e.g. `python3 -m mlstrat new my_project`.

To update, add `-U`: `pip install -U "mlstrat @ git+https://github.com/mahalmadane/projects_mlops.git#subdirectory=mlstrat"`.

## Create a project

```bash
mlstrat new my_project
```

```
my_project/
├── data/            raw/ interim/ processed/
├── notebooks/       01_exploration, 02_preprocessing, 03_model_training
├── src/             data/ features/ models/ utils/ pipelines/steps/
├── configs/         config.yaml, model_params.yaml, pipeline_config.yaml
├── reports/         figures/, metrics.json
├── scripts/         run_pipeline.py, train_local.py, deploy_model.py
├── tests/
├── pyproject.toml   (depends on mlstrat)
├── requirements.txt
└── README.md
```

### Installing the generated project

Nothing heavy is installed for you. Installing mlstrat only brings in the analysis stack
(pandas, scikit-learn, matplotlib, seaborn, scipy), never ZenML or MLflow. The generated project
lists all its tools in `requirements.txt`, so you choose what to install:

```bash
cd my_project
python -m venv .venv && source .venv/bin/activate
# edit requirements.txt first: comment out what you don't need (zenml, mlflow, jupyter...)
pip install -r requirements.txt
pip install -e .        # makes src/ importable; installs no dependencies
```

| File | Role |
|---|---|
| `requirements.txt` | Every dependency, grouped and commented: analysis, notebooks, MLOps (ZenML, MLflow), tests, optional API/UI (FastAPI, Streamlit) |
| `pyproject.toml` | Empty `dependencies` on purpose, so `pip install -e .` downloads nothing |
| `README.md` | Step-by-step installation, usage and folder-by-folder description |

Existing files are never overwritten, so `mlstrat new` can also fill in the missing pieces of
an existing project. From Python: `from mlstrat import create_project; create_project("my_project")`.

## Overview of a dataset

```python
import pandas as pd
from mlstrat import DataOverview

df = pd.read_csv("data/raw/my_data.csv")
DataOverview().report(df)      # dtype, missing, missing_pct, n_unique per column
DataOverview().missing(df)     # columns sorted by missing values
```

## Cleaning

Both cleaners copy the DataFrame (the original is never modified) and can be chained.

```python
from mlstrat import MissingValueImputer, OutlierClipper

df_clean = (
    MissingValueImputer(df)
    .impute("Age", "median")        # mean / median / mode / min / max, or a constant
    .impute("Cabin", "Unknown")
    .impute_all()                   # every other column: mean (numeric) or mode
    .get_df()
)

clipper = OutlierClipper(df_clean, factor=1.5).clip_all(exclude=["Survived", "PassengerId"])
df_clean = clipper.get_df()
clipper.bounds                      # {"Age": (lower, upper), ...}
```

## Exploratory analysis

Strategies are picked from the column dtypes; every plot returns its matplotlib object.

```python
from mlstrat import univariate_strategy, bivariate_strategy, MultivariateAnalyzer

univariate_strategy(df["Age"]).plot(df, "Age")       # histogram + box plot
univariate_strategy(df["Sex"]).describe(df, "Sex")   # value counts

s = bivariate_strategy(df, "Fare", "Pclass")         # numeric first
s.plot(df, "Fare", "Pclass")
s.test(df, "Fare", "Pclass")                         # ANOVA / correlation / chi-square

MultivariateAnalyzer().correlation_heatmap(df)
```

| Columns | Strategy | `test()` |
|---|---|---|
| numeric / numeric | `NumericNumericBivariate` | Pearson or Spearman correlation |
| numeric / categorical | `NumericCategoricalBivariate` | one-way ANOVA |
| categorical / categorical | `CategoricalCategoricalBivariate` | chi-square |

## Training

`ModelTrainer` wraps any scikit-learn classifier in a preprocessing pipeline (impute + scale
numeric columns, impute + one-hot encode the others), so raw DataFrames can be passed directly.

```python
from sklearn.linear_model import LogisticRegression
from mlstrat import ModelTrainer

trainer = ModelTrainer(LogisticRegression, C=0.1, max_iter=1000)
trainer.fit(X_train, y_train)
trainer.evaluate(X_test, y_test)   # {"accuracy", "f1_score", "precision", "recall"}
trainer.model                      # the fitted sklearn Pipeline
```

`build_preprocessor()` and `make_model_pipeline(estimator)` give you the same pipeline
without the wrapper.

## License

MIT
