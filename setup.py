from setuptools import setup, find_packages

setup(
    name="optuna_tuner",
    version="0.1.1",
    description="Búsqueda automática de hiperparámetros con Optuna",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "optuna>=3.0",
        "scikit-learn>=1.2",
        "numpy>=1.23",
        "pandas>=1.5",
    ],
    extras_require={
        "boosting": [
            "xgboost>=1.7",
            "lightgbm>=3.3",
            "catboost>=1.1",
        ],
    },
)