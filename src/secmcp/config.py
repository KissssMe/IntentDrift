from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_cfg(name: str) -> SimpleNamespace:
    models = _load_yaml(CONFIGS_DIR / "models.yaml")
    if name not in models:
        raise KeyError(f"Model '{name}' not found in models.yaml. Available: {list(models)}")
    return _to_namespace(models[name])


def load_data_cfg() -> SimpleNamespace:
    return _to_namespace(_load_yaml(CONFIGS_DIR / "data.yaml"))


def load_training_cfg() -> SimpleNamespace:
    return _to_namespace(_load_yaml(CONFIGS_DIR / "training.yaml"))


def load_eval_cfg() -> SimpleNamespace:
    return _to_namespace(_load_yaml(CONFIGS_DIR / "eval.yaml"))


def all_model_names() -> list[str]:
    return list(_load_yaml(CONFIGS_DIR / "models.yaml").keys())
