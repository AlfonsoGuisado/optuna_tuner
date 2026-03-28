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


def tune_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str,
    task: str,
    n_trials: int,
    metric: str | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Busca los mejores hiperparámetros para el modelo indicado usando Optuna.

    Parámetros
    ----------
    X            : DataFrame de features de entrenamiento
    y            : Serie o array de labels / target
    model_name   : 'randomforest', 'xgboost', 'catboost', 'lightgbm', ...
    task         : 'classification' o 'regression'
    metric       : métrica a optimizar (si None usa la default de cada tarea)
    n_trials     : número de trials de Optuna
    cv_folds     : número de folds para cross-validation (default 5)
    random_state : semilla aleatoria (default 42)
    verbose      : muestra progreso en consola (default True)

    Retorna
    -------
    dict con:
        best_params  – hiperparámetros óptimos listos para usar
        best_value   – mejor score obtenido
        best_model   – instancia del modelo con los mejores params
        study        – objeto optuna.Study completo
        metric       – métrica usada
    """

    # ── Validaciones ───────────────────────────────────────────────────────────
    task = task.lower().strip()
    if task not in ("classification", "regression"):
        raise ValueError(f"task debe ser 'classification' o 'regression', no '{task}'.")

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
            f"Modelo '{model_name}' no encontrado para '{task}'.\n"
            f"Disponibles: {available}"
        )

    metric = metric or default_metric
    if metric not in valid_metrics:
        raise ValueError(
            f"Métrica '{metric}' no válida para '{task}'.\n"
            f"Válidas: {', '.join(valid_metrics.keys())}"
        )

    sklearn_scoring = valid_metrics[metric]["sklearn_scoring"]
    direction       = valid_metrics[metric]["direction"]
    model_entry     = registry[model_key]

    # ── Objective ──────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = model_entry["params_fn"](trial, random_state)
        model  = model_entry["class"](**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=sklearn_scoring, n_jobs=-1)
        return float(scores.mean())

    # ── Study ──────────────────────────────────────────────────────────────────
    study = optuna.create_study(direction=direction)

    if verbose:
        _print_header(model_name, task, metric, n_trials, cv_folds)

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[ProgressCallback(n_trials, verbose=verbose)],
        show_progress_bar=False,
    )

    # ── Resultado ──────────────────────────────────────────────────────────────
    best_params = study.best_params
    best_model  = model_entry["class"](**best_params)

    if verbose:
        _print_results(study, metric)

    return {
        "best_params": best_params,
        "best_value":  study.best_value,
        "best_model":  best_model,
        "study":       study,
        "metric":      metric,
    }


# ── Helpers de consola ─────────────────────────────────────────────────────────

def _print_header(model_name, task, metric, n_trials, cv_folds):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Modelo  : {model_name}")
    print(f"  Tarea   : {task}")
    print(f"  Métrica : {metric}")
    print(f"  Trials  : {n_trials}  |  CV folds: {cv_folds}")
    print(f"{sep}")


def _print_results(study: optuna.Study, metric: str):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Mejor {metric}: {study.best_value:.6f}")
    print(f"  Mejores hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{sep}\n")