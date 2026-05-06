from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from secmcp.config import load_model_cfg

# Process-level model cache — both LocalHFLLM and ActivationDetectorElement share
# the same weights.  Without this, Llama-3.3-70B would be loaded twice (≈280 GB).
_MODEL_CACHE: dict[str, "LoadedModel"] = {}


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    cfg: SimpleNamespace


def torch_dtype(dtype_name: str):
    import torch

    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def namespace_to_plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: namespace_to_plain(val) for key, val in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_plain(item) for item in value]
    return value


def load_model(name: str) -> LoadedModel:
    """Load a configured HuggingFace causal LM and tokenizer (no caching).

    Prefer load_shared_model() in application code to avoid loading the same
    weights multiple times.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_model_cfg(name)
    model_path = Path(cfg.path)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Configured model path for {name!r} does not exist: {cfg.path}. "
            "If this is a local Hugging Face checkpoint, make sure the path is mounted "
            "and visible from the Python process running SecMCP."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.path,
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(cfg, "pad_token_id", None) is not None:
        tokenizer.pad_token_id = cfg.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.path,
        dtype=torch_dtype(cfg.dtype),
        device_map=namespace_to_plain(cfg.device_map),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    model.eval()
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, cfg=cfg)


def load_shared_model(name: str) -> LoadedModel:
    """Return a cached LoadedModel, loading from disk only on the first call.

    All callers within the same process (LocalHFLLM, ActivationDetectorElement,
    extraction scripts) will receive the same instance.
    """
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = load_model(name)
    return _MODEL_CACHE[name]


def clear_model_cache() -> None:
    """Release all cached models (useful in tests and multi-model scripts)."""
    _MODEL_CACHE.clear()
