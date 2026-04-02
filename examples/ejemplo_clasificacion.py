import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from hyperforge import forge_model, list_models, list_metrics

# ── Datos ──────────────────────────────────────────────────────────────────────
X_raw, y_raw = make_classification(
    n_samples=1_000, n_features=20, n_informative=10, n_classes=3, random_state=42
)
X = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(X_raw.shape[1])])
y = pd.Series(y_raw, name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Ver opciones disponibles ───────────────────────────────────────────────────
list_models(task="classification")
list_metrics(task="classification")

# ── Afinar modelo ──────────────────────────────────────────────────────────────
result = forge_model(
    X=X_train,
    y=y_train,
    model_name="randomforest",
    task="classification",
    metric="f1_micro",
    n_trials=50,
    cv_folds=5,
    verbose=True,
)

# ── Evaluar en test ────────────────────────────────────────────────────────────
model = result["best_model"]
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))