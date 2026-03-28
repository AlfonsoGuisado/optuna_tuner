from .tuner import tune
from .models import CLASSIFIERS, REGRESSORS
from .metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS

__version__ = "0.1.2"

__all__ = [
    "tune",
    "CLASSIFIERS",
    "REGRESSORS",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "list_models",
    "list_metrics",
]


def list_models(task: str | None = None) -> None:
    """Muestra los modelos disponibles por tarea."""
    if task is None or task == "classification":
        print("── Clasificación ──")
        for m in sorted(CLASSIFIERS.keys()):
            print(f"  {m}")
    if task is None or task == "regression":
        print("── Regresión ──")
        for m in sorted(REGRESSORS.keys()):
            print(f"  {m}")


def list_metrics(task: str | None = None) -> None:
    """Muestra las métricas disponibles por tarea."""
    if task is None or task == "classification":
        print("── Clasificación ──")
        for m in CLASSIFICATION_METRICS:
            print(f"  {m}")
    if task is None or task == "regression":
        print("── Regresión ──")
        for m in REGRESSION_METRICS:
            print(f"  {m}")