# 🔍 optuna_tuner

Personal library for automatic hyperparameter tuning with **Optuna**.  
Clone the repo, install it with a single command, and you’ll have `tune()` available in any project, both locally and in the cloud (Colab, Kaggle...).

---

## 📁 Repository structure

```
optuna_tuner/
├── optuna_tuner/
│   ├── assets/
│   │   ├── metrics.json          ← available metrics and their configuration
│   │   └── search_spaces.json    ← search ranges for each model
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py            ← reads JSON files and builds parameters for Optuna
│   │   ├── classifiers.py        ← classifiers registry
│   │   └── regressors.py         ← regressors registry
│   ├── __init__.py               ← public API: tune_model(), list_models(), list_metrics()
│   ├── callbacks.py              ← console progress for each trial
│   ├── metrics.py                ← loads metrics.json
│   └── tuner.py                  ← main function tune_model()
├── examples/
│   ├── ejemplo_clasificacion.py
│   └── ejemplo_regresion.py
├── tests/
│   └── test_tuner.py
├── setup.py
├── requirements.txt
└── .gitignore
```

**What does each part do?**

- `assets/` — JSON files with configuration. These are the only files you need to edit if you want to change search ranges or add metrics, without touching Python code.
- `models/` — contains available classifiers and regressors. To add a new model, you only need to modify this module.
- `tuner.py` — this is where the `tune()` function lives, the core of the library.
- `callbacks.py` — controls what gets printed to the console during the search.

---

## ⚡ Installation

### In a local project (VSCode)

```bash
pip install git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

With boosting libraries (XGBoost, LightGBM, CatBoost):

```bash
pip install "git+https://github.com/AlfonsoGuisado/optuna_tuner.git#egg=optuna_tuner[boosting]"
```

### Update to the latest version

```bash
pip install --upgrade git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

### In the cloud (Google Colab, Kaggle)

```bash
!pip install git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

---

## 🚀 Basic usage

```python
from optuna_tuner import tune

result = tune(
    X=X_train,           # DataFrame of features
    y=y_train,           # Series or array of labels
    model_name="xgboost",
    task="classification",
    metric="f1_micro",
    n_trials=100,
)

print(result["best_params"])   # best hyperparameters
print(result["best_value"])    # best score obtained
```

---

## 📖 `tune_model()` parameters

| Parámetro | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `X` | DataFrame | ✅ | Features de entrenamiento |
| `y` | Series / array | ✅ | Labels o target |
| `model_name` | str | ✅ | Nombre del modelo (ver tabla de modelos) |
| `task` | str | ✅ | `'classification'` o `'regression'` |
| `n_trials` | int | ✅ | Número de trials de Optuna |
| `metric` | str | ❌ | Métrica a optimizar (si no se indica usa la default) |
| `cv_folds` | int | ❌ | Número de folds para cross-validation (default: `5`) |
| `random_state` | int | ❌ | Semilla aleatoria (default: `42`) |
| `verbose` | bool | ❌ | Mostrar progreso en consola (default: `True`) |

---

## 📦 What `tune_model()` returns

`tune_model()` returns a dictionary with these keys:

```python
result = tune_model(...)

result["best_params"]   # dict with best hyperparameters, ready to use
result["best_value"]    # best score obtained during search
result["best_model"]    # model instance already configured with best params
result["study"]         # full optuna.Study object for advanced analysis
result["metric"]        # metric used
```

### Ejemplo de uso del resultado

```python
result = tune(X_train, y_train, model_name="randomforest", task="classification", n_trials=100)

# Train final model with full training data
best_model = result["best_model"]
best_model.fit(X_train, y_train)

# Evaluate on test
y_pred = best_model.predict(X_test)
```
