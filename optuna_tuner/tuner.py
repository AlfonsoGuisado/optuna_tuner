from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from .models import CLASSIFIERS, REGRESSORS
from .metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from .callbacks import ProgressCallback

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def model_tune(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str,
    task: str,
    n_trials: int,
    model_params: dict,
    metric: str | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    study_name: str | None = None,
    storage: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Searches for the best hyperparameters for the given model using Optuna.

    Parameters
    ----------
    X            : Training features DataFrame
    y            : Labels or target Series / array
    model_name   : Model name — 'randomforest', 'xgboost', 'catboost', etc.
    task         : 'classification' or 'regression'
    n_trials     : Number of Optuna trials
    model_params : Fixed parameters passed directly to the model, not searched by Optuna.
                   Pass an empty dict {} if no fixed params are needed.
                   Example: {"objective": "multi:softmax", "num_class": 3}
    metric       : Metric to optimize (uses task default if not specified)
    cv_folds     : Number of cross-validation folds (default: 5)
    random_state : Random seed (default: 42)
    verbose      : Show progress in console (default: True)
    study_name   : Name for the Optuna study. Required if storage is used.
                   If the study already exists it resumes from where it left off.
                   Example: "rf_classification_v1"
    storage      : Path to a SQLite database to persist trials on disk.
                   If the process crashes, re-running with the same study_name
                   and storage will resume automatically.
                   Example: "sqlite:///optuna_studies.db"

    Returns
    -------
    dict with keys:
        best_params  - best hyperparameters found
        best_value   - best score obtained
        best_model   - model instance configured with best params
        study        - complete optuna.Study object
        metric       - metric used
    """

    # ── Validations ────────────────────────────────────────────────────────────
    task = task.lower().strip()
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', not '{task}'.")

    model_key = model_name.lower().replace(" ", "").replace("_", "")

    if task == "classification":
        registry = CLASSIFIERS
        valid_metrics = CLASSIFICATION_METRICS
        default_metric = "f1_micro"
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        registry = REGRESSORS
        valid_metrics = REGRESSION_METRICS
        default_metric = "neg_rmse"
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Model '{model_name}' not found for task '{task}'.\n"
            f"Available: {available}"
        )

    metric = metric or default_metric
    if metric not in valid_metrics:
        raise ValueError(
            f"Metric '{metric}' not valid for '{task}'.\n"
            f"Valid options: {', '.join(valid_metrics.keys())}"
        )

    sklearn_scoring = valid_metrics[metric]["sklearn_scoring"]
    direction       = valid_metrics[metric]["direction"]
    model_entry     = registry[model_key]

    # ── Objective ──────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = model_entry["params_fn"](trial, random_state)
        params = {**params, **model_params}
        model  = model_entry["class"](**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=sklearn_scoring, n_jobs=-1)
        return float(scores.mean())

    # ── Study ──────────────────────────────────────────────────────────────────
    _study_name = study_name or f"{model_name}_{task}"

    study = optuna.create_study(
        direction=direction,
        study_name=_study_name,
        storage=storage,
        load_if_exists=True,      # ← si ya existe lo retoma, no lo sobreescribe
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = n_trials - completed

    if verbose:
        _print_header(model_name, task, metric, n_trials, cv_folds, completed)

    if remaining <= 0:
        if verbose:
            print(f"  Study already has {completed} completed trials. Nothing to run.")
            print(f"  To run more trials increase n_trials above {completed}.\n")
    else:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=timeout,
            callbacks=[ProgressCallback(n_trials, completed, verbose=verbose)],
            show_progress_bar=False,
        )

    # ── Result ─────────────────────────────────────────────────────────────────
    best_params = study.best_params
    best_model  = model_entry["class"](**{**best_params, **model_params})

    if verbose:
        _print_results(study, metric)

    return {
        "best_params": best_params,
        "best_value":  study.best_value,
        "best_model":  best_model,
        "study":       study,
        "metric":      metric,
    }


# ── Console helpers ────────────────────────────────────────────────────────────

def _print_header(model_name, task, metric, n_trials, cv_folds, completed):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Model   : {model_name}")
    print(f"  Task    : {task}")
    print(f"  Metric  : {metric}")
    print(f"  Trials  : {n_trials}  |  CV folds: {cv_folds}")
    if completed > 0:
        print(f"  Resuming from trial {completed + 1} ({completed} already completed)")
    print(f"{sep}")


def _print_results(study: optuna.Study, metric: str):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Best {metric}: {study.best_value:.6f}")
    print(f"  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{sep}\n")