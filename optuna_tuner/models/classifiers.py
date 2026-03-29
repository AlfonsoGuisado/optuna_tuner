from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from .builder import build_params
import optuna

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ── Modelos con lógica condicional (no se pueden expresar en JSON) ─────────────

def _params_logisticregression(trial: optuna.Trial, random_state: int = 42) -> dict:
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    penalty_options = ["l2", None] if solver == "lbfgs" else ["l1", "l2", "elasticnet", None]
    penalty = trial.suggest_categorical("penalty", penalty_options)
    params = {
        "C":            trial.suggest_float("C", 1e-4, 100.0, log=True),
        "solver":       solver,
        "penalty":      penalty,
        "max_iter":     trial.suggest_int("max_iter", 100, 1000, step=100),
        "random_state": random_state,
        "n_jobs":       -1,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params

def _params_svm(trial: optuna.Trial, random_state: int = 42) -> dict:
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    params = {
        "C":            trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel":       kernel,
        "probability":  True,
        "random_state": random_state,
    }
    params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    return params


# ── Registro ───────────────────────────────────────────────────────────────────

CLASSIFIERS = {
    "randomforest":       {"class": RandomForestClassifier,    "params_fn": lambda trial, rs: build_params(trial, "randomforest",     "classification", rs)},
    "extratrees":         {"class": ExtraTreesClassifier,      "params_fn": lambda trial, rs: build_params(trial, "extratrees",       "classification", rs)},
    "gradientboosting":   {"class": GradientBoostingClassifier,"params_fn": lambda trial, rs: build_params(trial, "gradientboosting", "classification", rs)},
    "knn":                {"class": KNeighborsClassifier,      "params_fn": lambda trial, rs: build_params(trial, "knn",              "classification", rs)},
    "logisticregression": {"class": LogisticRegression,        "params_fn": _params_logisticregression},
    "svm":                {"class": SVC,                       "params_fn": _params_svm},
}

if _HAS_XGB:
    CLASSIFIERS["xgboost"] = {
        "class": XGBClassifier,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "xgboost", "classification", rs),
            "verbosity": 0,
            "n_jobs":    -1,
        }
    }

if _HAS_LGB:
    CLASSIFIERS["lightgbm"] = {
        "class": LGBMClassifier,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "lightgbm", "classification", rs),
            "verbose": -1,
            "n_jobs":  -1,
        }
    }

if _HAS_CAT:
    CLASSIFIERS["catboost"] = {
        "class": CatBoostClassifier,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "catboost", "classification", rs),
            "verbose": False,
            "silent":  True,
        }
    }