import json
from pathlib import Path

_path = Path(__file__).parent / "assets" / "metrics.json"
with open(_path) as f:
    _data = json.load(f)

CLASSIFICATION_METRICS = _data["classification"]
REGRESSION_METRICS     = _data["regression"]