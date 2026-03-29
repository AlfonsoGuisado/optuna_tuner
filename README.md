# 🔍 optuna_tuner

Library for automatic hyperparameter search using **Optuna**.  
Install it with a single command and have `tune_model()` available in any project, both locally and in the cloud (Colab, Kaggle...).

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
│   │   ├── builder.py            ← reads JSONs and builds parameters for Optuna
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

- `assets/` — JSON configuration files. These are the only files you need to edit if you want to change search ranges or add metrics, without touching any Python code.
- `models/` — contains available classifiers and regressors. To add a new model, only this module needs to be modified.
- `tuner.py` — this is where `tune_model()` lives, the core of the library.
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

### Cloud (Google Colab, SageMaker... etc)

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
| `model_name` | str | ✅ | — | Model name (see model table) |
| `task` | str | ✅ | — | `'classification'` or `'regression'` |
| `n_trials` | int | ✅ | — | Number of Optuna trials |
| `model_params` | dict | ✅ | — | Fixed model parameters (see dedicated section) |
| `metric` | str | ❌ | auto | Metric to optimize |
| `cv_folds` | int | ❌ | `5` | Number of cross-validation folds |
| `random_state` | int | ❌ | `42` | Random seed |
| `verbose` | bool | ❌ | `True` | Show progress in console |

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
result = tune_model(X_train, y_train, model_name="xgboost", task="classification", n_trials=100)

best_model = result["best_model"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
```

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

Optuna searches for the best hyperparameters within the ranges defined in `search_spaces.json`, but some parameters **make no sense to search** because they are fixed based on your specific problem: the objective type, the model's internal metric, whether you have a GPU, whether classes are imbalanced, etc.

These parameters are passed via the `model_params` argument and take **priority** over any value found by Optuna.

```python
result = tune_model(
    X=X_train,
    y=y_train,
    model_name="xgboost",
    task="classification",
    metric="f1_micro",
    n_trials=100,
    model_params={                       # ← fixed params, Optuna won't touch these
        "objective":   "multi:softmax",
        "num_class":   3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }
)
```

---

### 🧭 Three Questions Before Defining `model_params`

**1. How many classes does my problem have?**
```python
n_classes = y_train.nunique()
# 2 classes  → binary problem
# >2 classes → multiclass problem, you'll need num_class=n_classes in XGBoost and LightGBM
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
    "objective":   "binary:logistic",   # output: probability between 0 and 1
    "eval_metric": "logloss",           # internal evaluation metric
    "tree_method": "hist",              # fastest tree algorithm on CPU
    "n_jobs":      -1,                  # use all available cores
    "verbosity":   0,                   # no internal XGBoost logs
}
```

**Multiclass classification**
```python
model_params={
    "objective":   "multi:softmax",     # output: predicted class as integer
  # "objective":  "multi:softprob",    # alternative: output probabilities per class
    "num_class":   y_train.nunique(),   # number of classes — REQUIRED for multiclass
    "eval_metric": "mlogloss",          # internal multiclass metric
    "tree_method": "hist",
    "n_jobs":      -1,
    "verbosity":   0,
}
```

**Regression**
```python
model_params={
    "objective":   "reg:squarederror",      # MSE regression — most common
  # "objective":  "reg:absoluteerror",     # MAE regression — more robust to outliers
  # "objective":  "reg:pseudohubererror",  # Huber regression — balance between MSE and MAE
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs":      -1,
    "verbosity":   0,
}
```

**Imbalanced classes (binary only)**
```python
# Automatically calculates the weight for the minority class
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
    "tree_method": "gpu_hist",          # greatly speeds up training
    "objective":   "binary:logistic",   # or whichever fits your problem
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
| `verbosity` | `0` | Internal log level |

---

### 🟢 LightGBM

**Binary classification**
```python
model_params={
    "objective": "binary",              # binary classification
    "metric":    "binary_logloss",      # internal metric
    "n_jobs":    -1,
    "verbose":   -1,                    # no internal logs
}
```

**Multiclass classification**
```python
model_params={
    "objective": "multiclass",          # multiclass classification
    "num_class": y_train.nunique(),     # number of classes — REQUIRED for multiclass
    "metric":    "multi_logloss",       # internal metric
    "n_jobs":    -1,
    "verbose":   -1,
}
```

**Regression**
```python
model_params={
    "objective": "regression",          # MSE regression — most common
  # "objective": "regression_l1",      # MAE regression — more robust to outliers
  # "objective": "huber",              # Huber regression — balance between MSE and MAE
  # "objective": "mape",               # mean absolute percentage error
    "metric":    "rmse",
    "n_jobs":    -1,
    "verbose":   -1,
}
```

**Imbalanced classes**
```python
model_params={
    "is_unbalance": True,               # automatic class weight adjustment
  # alternative:
  # "class_weight": "balanced",
}
```

| Parameter | Options | Description |
|---|---|---|
| `objective` | `binary` · `multiclass` · `regression` · `regression_l1` · `huber` · `mape` | Problem type and loss function |
| `num_class` | integer | Number of classes — only required for multiclass |
| `metric` | `binary_logloss` · `multi_logloss` · `rmse` · `mae` · `auc` · `mape` | Internal evaluation metric |
| `is_unbalance` | `True` / `False` | Automatic adjustment for imbalanced classes |
| `class_weight` | `"balanced"` | Alternative to `is_unbalance` |
| `n_jobs` | `-1` | Cores to use |
| `verbose` | `-1` | No internal logs |

---

### 🔵 CatBoost

**Binary classification**
```python
model_params={
    "loss_function": "Logloss",         # binary loss function
    "eval_metric":   "Logloss",         # evaluation metric
    "verbose":       False,             # no internal logs
}
```

**Multiclass classification**
```python
model_params={
    "loss_function": "MultiClass",      # multiclass loss function
    "eval_metric":   "Accuracy",        # evaluation metric
    "verbose":       False,
}
```

**Regression**
```python
model_params={
    "loss_function": "RMSE",            # MSE regression — most common
  # "loss_function": "MAE",            # MAE regression — more robust to outliers
  # "loss_function": "Huber",          # Huber regression
  # "loss_function": "MAPE",           # percentage error
    "eval_metric":   "RMSE",
    "verbose":       False,
}
```

**With categorical columns**
```python
# CatBoost natively handles categorical columns without prior encoding
model_params={
    "cat_features": [0, 2, 5],                              # by column index
  # "cat_features": ["city", "product", "category"],        # or by name if using DataFrame
    "loss_function": "MultiClass",
    "verbose":       False,
}
```

**Imbalanced classes**
```python
model_params={
    "auto_class_weights": "Balanced",   # automatic per-class weight adjustment
    "loss_function":      "Logloss",
    "verbose":            False,
}
```

| Parameter | Options | Description |
|---|---|---|
| `loss_function` | `Logloss` · `MultiClass` · `RMSE` · `MAE` · `Huber` · `MAPE` | Loss function |
| `eval_metric` | `Logloss` · `Accuracy` · `RMSE` · `MAE` · `AUC` | Internal evaluation metric |
| `cat_features` | list of indices or names | Categorical columns — CatBoost handles them natively |
| `auto_class_weights` | `"Balanced"` | Automatic adjustment for imbalanced classes |
| `verbose` | `False` | No internal logs |

---

### 🔴 RandomForest / ExtraTrees / GradientBoosting

Sklearn **detects the problem type automatically** based on the `task` you pass to `tune_model()`. You only need `model_params` in special cases:

**Imbalanced classes**
```python
# RandomForest and ExtraTrees
model_params={
    "class_weight": "balanced",             # weights each class inversely to its frequency
  # "class_weight": "balanced_subsample",   # same but recalculated for each tree
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
    "probability":  True,               # required to use predict_proba()
    "class_weight": "balanced",         # imbalanced classes
    "cache_size":   1000,               # kernel cache in MB — more = faster if you have RAM
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
| `probability` | `True` / `False` | Enables `predict_proba()` — SVC only, adds computational cost |
| `class_weight` | `"balanced"` · dict | Per-class weights |
| `cache_size` | integer (MB) | Kernel cache. More MB = faster with large datasets |

---

### 🟡 Linear Models (Lasso, Ridge, ElasticNet)

Rarely need `model_params`. Only in very specific cases:

```python
model_params={
    "fit_intercept": False,             # if data is already centered at 0
    "positive":      True,              # forces all coefficients to be positive
}
```

| Parameter | Options | Description |
|---|---|---|
| `fit_intercept` | `True` / `False` | Fit intercept (default `True`) |
| `positive` | `True` / `False` | Forces positive coefficients — useful when semantically meaningful |

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
        "verbosity":   0,
    }
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
        "verbose":   -1,
    }
)

model = result["best_model"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
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