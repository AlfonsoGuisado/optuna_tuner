import pytest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from hyperforge import forge_model


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


def test_randomforest_classifier(clf_data):
    X, y = clf_data
    result = forge_model(X, y, model_name="randomforest", task="classification", n_trials=5, verbose=False)
    assert result["best_value"] > 0

def test_gradientboosting_classifier(clf_data):
    X, y = clf_data
    result = forge_model(X, y, model_name="gradientboosting", task="classification", n_trials=5, verbose=False)
    assert result["best_value"] > 0

def test_randomforest_regressor(reg_data):
    X, y = reg_data
    result = forge_model(X, y, model_name="randomforest", task="regression", metric="r2", n_trials=5, verbose=False)
    assert "best_params" in result

def test_ridge_regressor(reg_data):
    X, y = reg_data
    result = forge_model(X, y, model_name="ridge", task="regression", n_trials=5, verbose=False)
    assert result["best_value"] is not None

def test_invalid_model(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError, match="no encontrado"):
        forge_model(X, y, model_name="modelo_falso", task="classification", n_trials=3, verbose=False)

def test_invalid_task(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError, match="task debe ser"):
        forge_model(X, y, model_name="randomforest", task="clustering", n_trials=3, verbose=False)

def test_invalid_metric(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError, match="no válida"):
        forge_model(X, y, model_name="randomforest", task="classification", metric="rmse", n_trials=3, verbose=False)

def test_result_keys(clf_data):
    X, y = clf_data
    result = forge_model(X, y, model_name="extratrees", task="classification", n_trials=5, verbose=False)
    for key in ("best_params", "best_value", "best_model", "study", "metric"):
        assert key in result