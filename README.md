# 🔍 optuna_tuner

Personal library for automatic hyperparameter tuning with **Optuna**.  
Clone the repo, install it with a single command, and you’ll have `tune()` available in any project, both locally and in the cloud (Colab, Kaggle...).

---

## 📁 Repository structure

```
optuna_tuner/
├── optuna_tuner/
│   ├── assets/
│   │   ├── metrics.json          ← métricas disponibles y su configuración
│   │   └── search_spaces.json    ← rangos de búsqueda de cada modelo
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py            ← lee los JSON y construye los parámetros para Optuna
│   │   ├── classifiers.py        ← registro de clasificadores
│   │   └── regressors.py         ← registro de regresores
│   ├── __init__.py               ← API pública: tune(), list_models(), list_metrics()
│   ├── callbacks.py              ← progreso por consola de cada trial
│   ├── metrics.py                ← carga metrics.json
│   └── tuner.py                  ← función principal tune()
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
