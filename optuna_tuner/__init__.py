from .tuner import model_tune
from .models import CLASSIFIERS, REGRESSORS
from .metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS

__version__ = "0.1.5"

__all__ = [
    "model_tune",
    "CLASSIFIERS",
    "REGRESSORS",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "list_models",
    "list_metrics",
]


def list_models(task: str | None = None) -> None:
    """Shows available models by task."""
    if task is None or task == "classification":
        print("── Classification ──")
        for m in sorted(CLASSIFIERS.keys()):
            print(f"  {m}")
    if task is None or task == "regression":
        print("── Regression ──")
        for m in sorted(REGRESSORS.keys()):
            print(f"  {m}")


def list_metrics(task: str | None = None) -> None:
    """Shows available metrics by task."""
    if task is None or task == "classification":
        print("── Classification ──")
        for m in CLASSIFICATION_METRICS:
            print(f"  {m}")
    if task is None or task == "regression":
        print("── Regression ──")
        for m in REGRESSION_METRICS:
            print(f"  {m}")