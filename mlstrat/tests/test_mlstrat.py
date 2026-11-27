import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlstrat
from mlstrat import (
    CategoricalCategoricalBivariate,
    CategoricalUnivariate,
    DataOverview,
    MissingValueImputer,
    ModelTrainer,
    MultivariateAnalyzer,
    NumericCategoricalBivariate,
    NumericNumericBivariate,
    NumericUnivariate,
    OutlierClipper,
    bivariate_strategy,
    univariate_strategy,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(30, 10, n)
    age[:10] = np.nan
    fare = rng.exponential(30, n)
    fare[0] = 1000.0
    sex = rng.choice(["male", "female"], n)
    embarked = rng.choice(["S", "C", "Q"], n).astype(object)
    embarked[:5] = None
    survived = ((sex == "female") | (rng.random(n) < 0.2)).astype(int)
    return pd.DataFrame({"Age": age, "Fare": fare, "Sex": sex, "Embarked": embarked, "Survived": survived})


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_version():
    assert isinstance(mlstrat.__version__, str)


def test_overview(df):
    ov = DataOverview()
    missing = ov.missing(df)
    assert missing.loc["Age", "missing"] == 10
    assert missing.loc["Embarked", "missing_pct"] == 2.5
    report = ov.report(df)
    assert set(report.columns) == {"dtype", "missing", "missing_pct", "n_unique"}
    assert ov.describe(df).shape[0] == df.shape[1]


def test_imputer_does_not_mutate_and_fills(df):
    before = df.isna().sum().sum()
    out = MissingValueImputer(df).impute("Age", "median").impute_all().get_df()
    assert df.isna().sum().sum() == before
    assert out.isna().sum().sum() == 0
    assert out.loc[0, "Age"] == pytest.approx(df["Age"].median())


def test_imputer_constant_and_errors(df):
    out = MissingValueImputer(df).impute("Embarked", "Unknown").get_df()
    assert (out["Embarked"] == "Unknown").sum() == 5
    with pytest.raises(ValueError):
        MissingValueImputer(df).impute("Embarked", "mean")
    with pytest.raises(ValueError):
        MissingValueImputer(df).impute("Age", "nope")
    with pytest.raises(KeyError):
        MissingValueImputer(df).impute("Missing")


def test_imputer_all_nan_column():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [None, None]})
    out = MissingValueImputer(df).impute_all().get_df()
    assert out.isna().all().all()


def test_outlier_clipper(df):
    clipper = OutlierClipper(df).clip_all(exclude=["Survived"])
    out = clipper.get_df()
    lower, upper = clipper.bounds["Fare"]
    assert out["Fare"].max() <= upper
    assert df["Fare"].max() == 1000.0
    assert "Survived" not in clipper.bounds
    OutlierClipper(df).clip("Sex")  # non-numeric: skipped, no error


def test_univariate(df):
    assert isinstance(univariate_strategy(df["Age"]), NumericUnivariate)
    assert isinstance(univariate_strategy(df["Sex"]), CategoricalUnivariate)
    num = NumericUnivariate(fill="median")
    assert num.impute(df, "Age").isna().sum() == 0
    assert df["Age"].isna().sum() == 10
    assert num.clip_outliers(df, "Fare").max() < 1000
    num.plot(df, "Age")
    cat = CategoricalUnivariate()
    assert cat.impute(df, "Embarked").isna().sum() == 0
    cat.plot(df, "Sex")
    assert cat.describe(df, "Sex").sum() == len(df)


def test_bivariate(df):
    assert isinstance(bivariate_strategy(df, "Age", "Fare"), NumericNumericBivariate)
    assert isinstance(bivariate_strategy(df, "Age", "Sex"), NumericCategoricalBivariate)
    assert isinstance(bivariate_strategy(df, "Sex", "Embarked"), CategoricalCategoricalBivariate)
    with pytest.raises(ValueError):
        bivariate_strategy(df, "Sex", "Age")

    nn = NumericNumericBivariate()
    res = nn.test(df, "Age", "Fare", method="spearman")
    assert -1 <= res["correlation"] <= 1
    nn.plot(df, "Age", "Fare")
    nn.regression_plot(df, "Age", "Fare")
    nn.heatmap(df, "Age", "Fare")

    nc = NumericCategoricalBivariate()
    assert 0 <= nc.test(df, "Fare", "Sex")["p_value"] <= 1
    nc.plot(df, "Fare", "Sex")
    nc.violin_plot(df, "Fare", "Sex")
    nc.mean_bar_plot(df, "Fare", "Sex")

    cc = CategoricalCategoricalBivariate()
    assert cc.test(df, "Sex", "Survived")["dof"] == 1
    cc.plot(df, "Sex", "Embarked")
    cc.stacked_bar_plot(df, "Sex", "Embarked")


def test_multivariate(df):
    m = MultivariateAnalyzer()
    corr = m.correlation(df)
    assert list(corr.columns) == ["Age", "Fare", "Survived"]
    m.correlation_heatmap(df)
    m.pairplot(df, columns=["Age", "Fare"], hue="Sex")


def test_trainer(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    metrics = ModelTrainer(LogisticRegression, max_iter=500).fit(X_train, y_train).evaluate(X_test, y_test)
    assert set(metrics) == {"accuracy", "f1_score", "precision", "recall"}
    assert metrics["accuracy"] > 0.6

    trainer = ModelTrainer(LogisticRegression(), C=0.5)
    assert trainer.model.named_steps["model"].C == 0.5
    with pytest.raises(TypeError):
        ModelTrainer(object())


def test_create_project(tmp_path):
    import json

    from mlstrat import create_project
    from mlstrat.cli import main

    result = create_project("demo", tmp_path)
    root = tmp_path / "demo"
    assert result.root == root
    assert (root / "data" / "raw" / ".gitkeep").exists()
    assert (root / "src" / "pipelines" / "steps" / "__init__.py").exists()
    nb = json.loads((root / "notebooks" / "01_exploration.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert "dependencies = []" in (root / "pyproject.toml").read_text()
    reqs = (root / "requirements.txt").read_text()
    for pkg in ("mlstrat", "zenml", "mlflow", "pytest"):
        assert f"\n{pkg}\n" in reqs
    assert "pip install -r requirements.txt" in (root / "README.md").read_text()
    for py in root.rglob("*.py"):
        compile(py.read_text(), str(py), "exec")

    (root / "README.md").write_text("mine")
    again = create_project("demo", tmp_path)
    assert (root / "README.md").read_text() == "mine"
    assert root / "README.md" in again.skipped
    assert not again.created

    with pytest.raises(ValueError):
        create_project("../evil", tmp_path)
    assert main(["new", "cli_demo", "--path", str(tmp_path)]) == 0
    assert (tmp_path / "cli_demo" / "pyproject.toml").exists()
    assert main(["new", "bad name", "--path", str(tmp_path)]) == 1
