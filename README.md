# 🔍 optuna_tuner

Library for automatic hyperparameter search using **Optuna**.  
Install it with a single command and have `tune_model()` available in any project, both locally and in the cloud (Colab, Kaggle...).

> **Current version: 0.2.0**

---

## 📁 Repository Structure

```
optuna_tuner/
├── optuna_tuner/
│   ├── assets/
│   │   ├── metrics.json          ← available metrics and their configuration
│   │   └── search_spaces.json    ← search ranges for each model
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py            ← reads JSONs and builds Optuna parameter spaces
│   │   ├── classifiers.py        ← classifier registry
│   │   └── regressors.py         ← regressor registry
│   ├── __init__.py               ← public API: tune_model(), list_models(), list_metrics()
│   ├── callbacks.py              ← console progress per trial
│   ├── metrics.py                ← loads metrics.json
│   └── tuner.py                  ← main tune_model() function
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

- `assets/` — JSON configuration files. The only files you need to edit to change search ranges or add new metrics, without touching any Python code.
- `models/` — contains all available classifiers and regressors. To add a new model, only this module needs to be modified.
- `tuner.py` — where `tune_model()` lives, the core of the library.
- `callbacks.py` — controls what gets printed to the console during the search.

---

## ⚡ Installation

### Local project (VSCode)

```bash
pip install git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

With boosting libraries (XGBoost, LightGBM, CatBoost):

```bash
pip install "git+https://github.com/AlfonsoGuisado/optuna_tuner.git#egg=optuna_tuner[boosting]"
```

### Cloud (Google Colab, Kaggle)

```python
!pip install git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

### Update to the latest version

```bash
pip install --upgrade git+https://github.com/AlfonsoGuisado/optuna_tuner.git
```

---

## 🚀 Basic Usage

```python
from optuna_tuner import tune_model

result = tune_model(
    X=X_train,
    y=y_train,
    model_name="xgboost",
    task="classification",
    metric="f1_micro",
    n_trials=100,
    model_params={
        "objective":   "multi:softmax",
        "num_class":   y_train.nunique(),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs":      -1,
    }
)

print(result["best_params"])   # best hyperparameters found
print(result["best_value"])    # best score obtained
```

---

## 📖 `tune_model()` Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `X` | DataFrame | ✅ | — | Training features |
| `y` | Series / array | ✅ | — | Labels or target |
| `model_name` | str | ✅ | — | Model name (see model table below) |
| `task` | str | ✅ | — | `'classification'` or `'regression'` |
| `n_trials` | int | ✅ | — | Number of **successful** Optuna trials |
| `model_params` | dict | ✅ | — | Fixed model parameters not searched by Optuna. Pass `{}` if none needed |
| `metric` | str | ❌ | auto | Metric to optimize |
| `cv_folds` | int | ❌ | `5` | Number of cross-validation folds |
| `random_state` | int | ❌ | `42` | Random seed |
| `verbose` | bool | ❌ | `True` | Show trial progress in console |
| `study_name` | str | ❌ | auto | Name for the Optuna study. Required when using `storage` |
| `storage` | str | ❌ | `None` | SQLite path to persist trials on disk and resume if the process crashes |
| `timeout` | int | ❌ | `None` | Maximum search time in seconds. Stops after this time regardless of remaining trials |

> **Important:** `n_trials` counts only **successfully completed trials**. Failed trials are automatically retried and do not count towards the total.

---

## 📦 What `tune_model()` Returns

```python
result = tune_model(...)

result["best_params"]   # dict with the best hyperparameters, ready to use
result["best_value"]    # best score obtained during the search
result["best_model"]    # model instance configured with the best params
result["study"]         # complete optuna.Study object for advanced analysis
result["metric"]        # metric that was used
```

### Using the trained model

```python
result = tune_model(...)

best_model = result["best_model"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
```

---

## 💾 Crash Recovery — Never Lose a Search Again

If the process crashes or the kernel restarts mid-search, you can resume automatically from where it left off using `storage` and `study_name`:

```python
result = tune_model(
    X=X_train,
    y=y_train,
    model_name="randomforest",
    task="classification",
    metric="f1_micro",
    n_trials=200,
    model_params={"class_weight": "balanced"},
    study_name="rf_classification_v1",          # ← name of the study
    storage="sqlite:///optuna_studies.db",       # ← file where trials are saved on disk
    verbose=True,
)
```

If the process crashes and you re-run the exact same cell, Optuna will detect that `rf_classification_v1` already exists in the `.db` file, load the completed trials and **continue from where it left off**. No trial is ever lost.

---

## 🗄️ Inspecting the SQLite Database

Every time you use `storage`, Optuna saves all trials to a `.db` file in the same folder as your notebook. Here is everything you can do with it.

### List all studies saved in the database

```python
import optuna

studies = optuna.get_all_study_names(storage="sqlite:///optuna_studies.db")
print(studies)
# ['rf_classification_v1', 'xgb_clf_v1', 'catboost_classification_v1']
```

### Load a specific study

```python
study = optuna.load_study(
    study_name="catboost_classification_v1",
    storage="sqlite:///optuna_studies.db",
)
```

### Check trial counts and states

```python
df = study.trials_dataframe()
print(df["state"].value_counts())
# COMPLETE    28
# FAILED       2
```

### See all trials with their scores and duration

```python
df = study.trials_dataframe()
print(df[["number", "value", "state", "duration"]].to_string())
```

### See the best result

```python
print(f"Best value  : {study.best_value:.6f}")
print(f"Best trial  : {study.best_trial.number}")
print(f"Best params : {study.best_params}")
```

### See the top 5 trials

```python
df = study.trials_dataframe()
top5 = (
    df[df["state"] == "COMPLETE"]
    .sort_values("value", ascending=False)
    .head(5)
)
print(top5)
```

### See the parameters tested in every trial

```python
df = study.trials_dataframe()

# columns starting with "params_" contain the hyperparameters of each trial
param_cols = [col for col in df.columns if col.startswith("params_")]
print(df[["number", "value"] + param_cols].to_string())
```

### See the full detail of a specific trial

```python
trial = study.trials[10]   # trial number 10

print(f"Number   : {trial.number}")
print(f"State    : {trial.state}")
print(f"Score    : {trial.value}")
print(f"Duration : {trial.duration}")
print(f"Params   :")
for k, v in trial.params.items():
    print(f"  {k}: {v}")
```

### Export all trials to a CSV

```python
df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)
```

---

## ⏱️ Controlling Search Time

Use `timeout` to set a maximum search duration in seconds. The search will stop after that time regardless of how many successful trials remain, and will always return the best result found so far:

```python
result = tune_model(
    X=X_train,
    y=y_train,
    model_name="xgboost",
    task="classification",
    metric="f1_micro",
    n_trials=500,
    model_params={...},
    study_name="xgb_clf_v1",
    storage="sqlite:///optuna_studies.db",
    timeout=3600,                               # ← stop after 1 hour no matter what
    verbose=True,
)
```

Recommended timeouts based on dataset size:

| Dataset size | Recommended timeout |
|---|---|
| Small (< 10k rows) | 600 – 1800 seconds (10 – 30 min) |
| Medium (10k – 100k rows) | 1800 – 7200 seconds (30 min – 2 hours) |
| Large (> 100k rows) | 7200 – 18000 seconds (2 – 5 hours) |

---

## 🤖 Available Models

### Classification

| Name to use | Model |
|---|---|
| `randomforest` | RandomForestClassifier |
| `extratrees` | ExtraTreesClassifier |
| `gradientboosting` | GradientBoostingClassifier |
| `logisticregression` | LogisticRegression |
| `svm` | SVC |
| `knn` | KNeighborsClassifier |
| `xgboost` * | XGBClassifier |
| `lightgbm` * | LGBMClassifier |
| `catboost` * | CatBoostClassifier |

### Regression

| Name to use | Model |
|---|---|
| `randomforest` | RandomForestRegressor |
| `extratrees` | ExtraTreesRegressor |
| `gradientboosting` | GradientBoostingRegressor |
| `lasso` | Lasso |
| `ridge` | Ridge |
| `elasticnet` | ElasticNet |
| `svr` | SVR |
| `xgboost` * | XGBRegressor |
| `lightgbm` * | LGBMRegressor |
| `catboost` * | CatBoostRegressor |

`*` requires installation with `[boosting]`

```python
from optuna_tuner import list_models

list_models()                        # all models
list_models(task="classification")   # classification only
list_models(task="regression")       # regression only
```

---

## 📊 Available Metrics

### Classification — default: `f1_micro`

| Metric | Description | When to use it |
|---|---|---|
| `f1_micro` | Micro-averaged F1 | Multiclass with imbalanced classes |
| `f1_macro` | Macro-averaged F1 | Multiclass when all classes matter equally |
| `f1_weighted` | Support-weighted F1 | Multiclass with different sample counts per class |
| `accuracy` | Accuracy | Balanced classes |
| `balanced_accuracy` | Balanced accuracy | Imbalanced classes |
| `roc_auc` | AUC-ROC | Binary classification |
| `roc_auc_ovr` | AUC-ROC One-vs-Rest | Multiclass classification |
| `precision_macro` | Macro precision | When false positives are very costly |
| `precision_weighted` | Weighted precision | Same but with imbalanced classes |
| `recall_macro` | Macro recall | When false negatives are very costly |
| `recall_weighted` | Weighted recall | Same but with imbalanced classes |

### Regression — default: `neg_rmse`

| Metric | Description | When to use it |
|---|---|---|
| `neg_rmse` | Negative RMSE | General error, penalizes large errors |
| `neg_mae` | Negative MAE | General error, more robust to outliers |
| `neg_mse` | Negative MSE | When large errors are very costly |
| `neg_mape` | Negative MAPE | When relative error matters more than absolute |
| `neg_msle` | Negative MSLE | When the target has a logarithmic scale |
| `neg_medae` | Negative median absolute error | Very robust to extreme outliers |
| `r2` | Coefficient of determination | Proportion of explained variance |

> Regression metrics are negative because Optuna always maximizes internally. A lower RMSE is better, and its higher negative value is too.

```python
from optuna_tuner import list_metrics

list_metrics()                        # all metrics
list_metrics(task="classification")   # classification only
list_metrics(task="regression")       # regression only
```

---

## 🔎 View a Model's Search Ranges

```python
from optuna_tuner.models.builder import SEARCH_SPACES
import json

print(json.dumps(SEARCH_SPACES["classification"]["xgboost"], indent=2))
```

---

## ⚙️ Fixed Model Parameters (`model_params`)

Optuna searches for the best hyperparameters within the ranges defined in `search_spaces.json`, but some parameters **make no sense to search** because they are fixed based on your specific problem: the objective type, the internal metric, whether classes are imbalanced, etc.

These are passed via `model_params` and take **priority** over any value found by Optuna.

> **Note:** Internal logging parameters (`verbose`, `verbosity`, `silent`, `logging_level`, `verbose_eval`) are automatically handled by the library for all boosting models. You never need to include them in `model_params` — they will be silently ignored even if passed.

---

### 🧭 Three Questions Before Defining `model_params`

**1. How many classes does my problem have?**
```python
n_classes = y_train.nunique()
# 2 classes  → binary problem
# >2 classes → multiclass, you'll need num_class=n_classes in XGBoost and LightGBM
```

**2. Are the classes balanced?**
```python
y_train.value_counts(normalize=True)
# If any class represents less than 20% → imbalanced problem
# Add class_weight, is_unbalance or scale_pos_weight depending on the model
```

**3. Do you have a GPU available?**
```python
# Yes → "tree_method": "gpu_hist"   (XGBoost only)
# No  → "tree_method": "hist"       (always recommended on CPU)
```

---

### 🟠 XGBoost

**Binary classification**
```python
model_params={
    "objective":   "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs":      -1,
}
```

**Multiclass classification**
```python
model_params={
    "objective":   "multi:softmax",
  # "objective":  "multi:softprob",    # alternative: output probabilities per class
    "num_class":   y_train.nunique(),   # REQUIRED for multiclass
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs":      -1,
}
```

**Regression**
```python
model_params={
    "objective":   "reg:squarederror",
  # "objective":  "reg:absoluteerror",
  # "objective":  "reg:pseudohubererror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs":      -1,
}
```

**Imbalanced classes (binary only)**
```python
model_params={
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "tree_method":      "hist",
}
```

**With GPU**
```python
model_params={
    "tree_method": "gpu_hist",
    "objective":   "binary:logistic",
}
```

| Parameter | Options | Description |
|---|---|---|
| `objective` | `binary:logistic` · `multi:softmax` · `multi:softprob` · `reg:squarederror` · `reg:absoluteerror` · `reg:pseudohubererror` | Problem type and loss function |
| `num_class` | integer | Number of classes — only required for multiclass |
| `eval_metric` | `logloss` · `mlogloss` · `rmse` · `mae` · `auc` · `aucpr` | Internal evaluation metric |
| `tree_method` | `hist` · `gpu_hist` · `exact` · `approx` | Tree building algorithm |
| `scale_pos_weight` | float | Positive class weight — binary imbalanced only |
| `n_jobs` | `-1` | Cores to use |

---

### 🟢 LightGBM

**Binary classification**
```python
model_params={
    "objective": "binary",
    "metric":    "binary_logloss",
    "n_jobs":    -1,
}
```

**Multiclass classification**
```python
model_params={
    "objective": "multiclass",
    "num_class": y_train.nunique(),     # REQUIRED for multiclass
    "metric":    "multi_logloss",
    "n_jobs":    -1,
}
```

**Regression**
```python
model_params={
    "objective": "regression",
  # "objective": "regression_l1",
  # "objective": "huber",
  # "objective": "mape",
    "metric":    "rmse",
    "n_jobs":    -1,
}
```

**Imbalanced classes**
```python
model_params={
    "is_unbalance": True,
  # "class_weight": "balanced",
}
```

| Parameter | Options | Description |
|---|---|---|
| `objective` | `binary` · `multiclass` · `regression` · `regression_l1` · `huber` · `mape` | Problem type and loss function |
| `num_class` | integer | Number of classes — only required for multiclass |
| `metric` | `binary_logloss` · `multi_logloss` · `rmse` · `mae` · `auc` · `mape` | Internal evaluation metric |
| `is_unbalance` | `True` / `False` | Automatic class weight adjustment |
| `class_weight` | `"balanced"` | Alternative to `is_unbalance` |
| `n_jobs` | `-1` | Cores to use |

---

### 🔵 CatBoost

**Binary classification**
```python
model_params={
    "loss_function": "Logloss",
    "eval_metric":   "Logloss",
}
```

**Multiclass classification**
```python
model_params={
    "loss_function": "MultiClass",
    "eval_metric":   "Accuracy",
}
```

**Regression**
```python
model_params={
    "loss_function": "RMSE",
  # "loss_function": "MAE",
  # "loss_function": "Huber",
  # "loss_function": "MAPE",
    "eval_metric":   "RMSE",
}
```

**With categorical columns**
```python
model_params={
    "cat_features":  [0, 2, 5],
  # "cat_features":  ["city", "product", "category"],
    "loss_function": "MultiClass",
    "eval_metric":   "Accuracy",
}
```

**Imbalanced classes**
```python
model_params={
    "auto_class_weights": "Balanced",
    "loss_function":      "Logloss",
    "eval_metric":        "Logloss",
}
```

| Parameter | Options | Description |
|---|---|---|
| `loss_function` | `Logloss` · `MultiClass` · `RMSE` · `MAE` · `Huber` · `MAPE` | Loss function |
| `eval_metric` | `Logloss` · `Accuracy` · `RMSE` · `MAE` · `AUC` | Internal evaluation metric |
| `cat_features` | list of indices or names | Categorical columns — handled natively, no encoding needed |
| `auto_class_weights` | `"Balanced"` | Automatic per-class weight adjustment |

---

### 🔴 RandomForest / ExtraTrees / GradientBoosting

Sklearn detects the problem type automatically. You only need `model_params` in special cases:

**Imbalanced classes**
```python
# RandomForest and ExtraTrees
model_params={
    "class_weight": "balanced",
  # "class_weight": "balanced_subsample",
}

# GradientBoosting does not have class_weight
# Use sample_weight directly in .fit() instead
```

| Parameter | Options | Description |
|---|---|---|
| `class_weight` | `"balanced"` · `"balanced_subsample"` · dict | Per-class weights (not available in GradientBoosting) |

---

### 🟣 SVM / SVR

**Classification**
```python
model_params={
    "probability":  True,
    "class_weight": "balanced",
    "cache_size":   1000,
}
```

**Regression**
```python
model_params={
    "cache_size": 1000,
}
```

| Parameter | Options | Description |
|---|---|---|
| `probability` | `True` / `False` | Enables `predict_proba()` — SVC only |
| `class_weight` | `"balanced"` · dict | Per-class weights |
| `cache_size` | integer (MB) | Kernel cache. More MB = faster with large datasets |

---

### 🟡 Linear Models (Lasso, Ridge, ElasticNet)

```python
model_params={
    "fit_intercept": False,
    "positive":      True,
}
```

| Parameter | Options | Description |
|---|---|---|
| `fit_intercept` | `True` / `False` | Fit intercept (default `True`) |
| `positive` | `True` / `False` | Forces positive coefficients |

---

## 💡 Full Examples

### Multiclass Classification with XGBoost

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from optuna_tuner import tune_model

X_raw, y_raw = make_classification(
    n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42
)
X = pd.DataFrame(X_raw)
y = pd.Series(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

result = tune_model(
    X=X_train,
    y=y_train,
    model_name="xgboost",
    task="classification",
    metric="f1_micro",
    n_trials=100,
    model_params={
        "objective":   "multi:softmax",
        "num_class":   y_train.nunique(),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs":      -1,
    },
    study_name="xgb_clf_v1",
    storage="sqlite:///optuna_studies.db",
    timeout=3600,
    verbose=True,
)

model = result["best_model"]
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

### Regression with LightGBM

```python
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from optuna_tuner import tune_model

X_raw, y_raw = make_regression(n_samples=1000, n_features=20, random_state=42)
X = pd.DataFrame(X_raw)
y = pd.Series(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

result = tune_model(
    X=X_train,
    y=y_train,
    model_name="lightgbm",
    task="regression",
    metric="neg_rmse",
    n_trials=100,
    model_params={
        "objective": "regression",
        "metric":    "rmse",
        "n_jobs":    -1,
    },
    study_name="lgb_reg_v1",
    storage="sqlite:///optuna_studies.db",
    timeout=1800,
    verbose=True,
)

model = result["best_model"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
```

### Imbalanced Multiclass with CatBoost

```python
from optuna_tuner import tune_model

result = tune_model(
    X=X_train,
    y=y_train,
    model_name="catboost",
    task="classification",
    metric="f1_micro",
    n_trials=100,
    model_params={
        "loss_function":      "MultiClass",
        "eval_metric":        "Accuracy",
        "auto_class_weights": "Balanced",
    },
    study_name="catboost_clf_v1",
    storage="sqlite:///optuna_studies.db",
    timeout=3600,
    verbose=True,
)
```

### Balanced Binary Classification with RandomForest

```python
from optuna_tuner import tune_model

result = tune_model(
    X=X_train,
    y=y_train,
    model_name="randomforest",
    task="classification",
    metric="roc_auc",
    n_trials=100,
    model_params={
        "class_weight": "balanced",
    },
    study_name="rf_binary_v1",
    storage="sqlite:///optuna_studies.db",
    timeout=1800,
    verbose=True,
)
```

---

## ➕ How to Add a New Model

**1.** Add its search space in `optuna_tuner/assets/search_spaces.json`:

```json
"classification": {
    "mymodel": {
        "n_estimators":  {"type": "int",   "low": 100,  "high": 1000},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true}
    }
}
```

**2.** Add its entry in `optuna_tuner/models/classifiers.py`:

```python
from mylibrary import MyModel

CLASSIFIERS["mymodel"] = {
    "class":     MyModel,
    "params_fn": lambda trial, rs: build_params(trial, "mymodel", "classification", rs)
}
```

Done! Nothing else needs to be changed.

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📋 Requirements

```
optuna>=3.0
scikit-learn>=1.2
numpy>=1.23
pandas>=1.5
xgboost>=1.7
lightgbm>=3.3
catboost>=1.1
```