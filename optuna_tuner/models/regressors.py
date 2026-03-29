from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from .builder import build_params
import optuna

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ── Modelos con lógica condicional ─────────────────────────────────────────────

def _params_svr(trial: optuna.Trial, random_state: int = 42) -> dict:
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    params = {
        "C":       trial.suggest_float("C", 1e-3, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "kernel":  kernel,
    }
    params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    return params

# ── Registro ───────────────────────────────────────────────────────────────────

REGRESSORS = {
    "randomforest": {"class": RandomForestRegressor, "params_fn": lambda trial, rs: build_params(trial, "randomforest", "regression", rs)},
    "extratrees": {"class": ExtraTreesRegressor, "params_fn": lambda trial, rs: build_params(trial, "extratrees", "regression", rs)},
    "gradientboosting": {"class": GradientBoostingRegressor, "params_fn": lambda trial, rs: build_params(trial, "gradientboosting", "regression", rs)},
    "lasso": {"class": Lasso, "params_fn": lambda trial, rs: build_params(trial, "lasso", "regression", rs)},
    "ridge": {"class": Ridge, "params_fn": lambda trial, rs: build_params(trial, "ridge", "regression", rs)},
    "elasticnet": {"class": ElasticNet, "params_fn": lambda trial, rs: build_params(trial, "elasticnet", "regression", rs)},
    "svr": {"class": SVR, "params_fn": _params_svr},
}

if _HAS_XGB:
    REGRESSORS["xgboost"] = {
        "class": XGBRegressor,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "xgboost", "regression", rs),
            "verbosity": 0,
            "n_jobs":    -1,
        }
    }

if _HAS_LGB:
    REGRESSORS["lightgbm"] = {
        "class": LGBMRegressor,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "lightgbm", "regression", rs),
            "verbose": -1,
            "n_jobs":  -1,
        }
    }

if _HAS_CAT:
    REGRESSORS["catboost"] = {
        "class": CatBoostRegressor,
        "params_fn": lambda trial, rs: {
            **build_params(trial, "catboost", "regression", rs),
            "verbose": False,
            "silent":  True,
        }
    }