"""Phase 0 tests: config loading, path validation, yaml consistency."""
from __future__ import annotations

import pytest
from pathlib import Path

from secmcp.config import (
    PROJECT_ROOT,
    CONFIGS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    all_model_names,
    load_model_cfg,
    load_data_cfg,
    load_training_cfg,
    load_eval_cfg,
)


def test_project_root_is_secmcp():
    """PROJECT_ROOT must point to the SecMCP/ directory."""
    assert PROJECT_ROOT.name == "SecMCP"
    assert (PROJECT_ROOT / "configs" / "models.yaml").exists()


def test_configs_dir_exists():
    for name in ("models.yaml", "data.yaml", "training.yaml", "eval.yaml"):
        assert (CONFIGS_DIR / name).exists(), f"Missing {name}"


def test_all_yaml_load():
    load_data_cfg()
    load_training_cfg()
    load_eval_cfg()


def test_model_names():
    names = all_model_names()
    assert set(names) == {"llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"}


@pytest.mark.parametrize("name", ["llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"])
def test_model_cfg_required_fields(name):
    cfg = load_model_cfg(name)
    for field in (
        "path", "dtype", "device_map", "layers",
        "num_hidden_layers", "hidden_dim",
        "detect_max_tokens", "truncation",
        "supports_tools_in_chat_template", "tool_call_format",
    ):
        assert hasattr(cfg, field), f"Model '{name}' missing field '{field}'"


@pytest.mark.parametrize("name", ["llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"])
def test_model_layers_within_bounds(name):
    cfg = load_model_cfg(name)
    for layer_idx in cfg.layers:
        assert 0 <= layer_idx < cfg.num_hidden_layers, (
            f"Model '{name}': layer index {layer_idx} out of range "
            f"[0, {cfg.num_hidden_layers})"
        )


_VALID_DTYPES = {"float32", "float16", "bfloat16", "float64", "int8"}


@pytest.mark.parametrize("name", ["llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"])
def test_model_dtype_valid(name):
    cfg = load_model_cfg(name)
    assert cfg.dtype in _VALID_DTYPES, (
        f"Model '{name}' has unknown dtype '{cfg.dtype}'. Expected one of {_VALID_DTYPES}"
    )


@pytest.mark.parametrize("name", ["llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"])
def test_model_tool_call_format_valid(name):
    cfg = load_model_cfg(name)
    valid_formats = {"llama3_json", "hermes", "mistral", "prompting"}
    assert cfg.tool_call_format in valid_formats, (
        f"Model '{name}' has unknown tool_call_format '{cfg.tool_call_format}'"
    )


@pytest.mark.parametrize("name", ["llama3_3_70b", "qwen3_32b", "mistral_7b_v03", "gemma2_9b"])
def test_model_path_exists(name):
    """Skip if model weights are not mounted; otherwise verify the directory exists."""
    cfg = load_model_cfg(name)
    model_path = Path(cfg.path)
    if not model_path.parent.exists():
        pytest.skip(f"Model weight directory not mounted at {model_path}")
    assert model_path.exists(), f"Model path does not exist: {model_path}"


def test_load_unknown_model_raises():
    with pytest.raises(KeyError, match="not found"):
        load_model_cfg("nonexistent_model")


def test_outputs_dir_exists():
    for subdir in ("splits", "activations", "drift_activations", "detectors", "baselines", "eval", "logs"):
        assert (OUTPUTS_DIR / subdir).exists(), f"Missing outputs/{subdir}"
