from setuptools import setup, find_packages

setup(
    name="hyperforge",
    version="0.2.1",
    description="Búsqueda automática de hiperparámetros con Optuna",
    packages=find_packages(),
    python_requires=">=3.9",
    package_data={
        "hyperforge": ["assets/*.json"],
    },
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