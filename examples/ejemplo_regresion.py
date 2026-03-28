import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from optuna_tuner import tune_model

# ── Datos ──────────────────────────────────────────────────────────────────────
X_raw, y_raw = make_regression(
    n_samples=1_000, n_features=20, n_informative=10, noise=0.1, random_state=42
)
X = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(X_raw.shape[1])])
y = pd.Series(y_raw, name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Afinar modelo ──────────────────────────────────────────────────────────────
result = tune_model(
    X=X_train,
    y=y_train,
    model_name="xgboost",
    task="regression",
    metric="neg_rmse",
    n_trials=50,
    cv_folds=5,
    verbose=True,
)

# ── Evaluar en test ────────────────────────────────────────────────────────────
model = result["best_model"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")