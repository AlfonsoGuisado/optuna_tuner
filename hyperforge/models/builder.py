import json
import optuna
from pathlib import Path

_path = Path(__file__).parent.parent / "assets" / "search_spaces.json"
with open(_path) as f:
    SEARCH_SPACES = json.load(f)


def build_params(trial: optuna.Trial, model_name: str, task: str, random_state: int = 42) -> dict:
    space = SEARCH_SPACES[task][model_name]
    params = {}

    for param_name, config in space.items():
        t = config["type"]
        if t == "int":
            kwargs = {"step": config["step"]} if "step" in config else {}
            params[param_name] = trial.suggest_int(param_name, config["low"], config["high"], **kwargs)
        elif t == "float":
            kwargs = {"log": config["log"]} if "log" in config else {}
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], **kwargs)
        elif t == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config["choices"])

    return params