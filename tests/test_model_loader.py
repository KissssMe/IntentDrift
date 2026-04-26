from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from secmcp.models.loader import (
    LoadedModel,
    _MODEL_CACHE,
    clear_model_cache,
    load_model,
    load_shared_model,
    namespace_to_plain,
    torch_dtype,
)


def test_torch_dtype_valid():
    torch = pytest.importorskip("torch")
    assert torch_dtype("float32") is torch.float32
    assert torch_dtype("bfloat16") is torch.bfloat16


def test_torch_dtype_invalid():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        torch_dtype("not_a_dtype")


def test_namespace_to_plain_converts_device_map():
    value = SimpleNamespace(device_map=SimpleNamespace(**{"": 0}), nested=[SimpleNamespace(x=1)])
    assert namespace_to_plain(value) == {"device_map": {"": 0}, "nested": [{"x": 1}]}


def test_namespace_to_plain_string_passthrough():
    assert namespace_to_plain("auto") == "auto"


def test_namespace_to_plain_list():
    assert namespace_to_plain([1, "a"]) == [1, "a"]


def test_load_shared_model_caches():
    """load_shared_model must call load_model only once for the same name."""
    clear_model_cache()
    fake = LoadedModel(name="fake", model=None, tokenizer=None, cfg=SimpleNamespace())
    with patch("secmcp.models.loader.load_model", return_value=fake) as mock_load:
        r1 = load_shared_model("fake")
        r2 = load_shared_model("fake")
    assert mock_load.call_count == 1, "load_model should be called exactly once"
    assert r1 is r2, "Both calls must return the same object"
    clear_model_cache()


def test_load_shared_model_different_names_load_separately():
    """Two different model names must each get their own load_model call."""
    clear_model_cache()
    fake_a = LoadedModel(name="a", model=None, tokenizer=None, cfg=SimpleNamespace())
    fake_b = LoadedModel(name="b", model=None, tokenizer=None, cfg=SimpleNamespace())

    def _fake_load(name: str) -> LoadedModel:
        return fake_a if name == "a" else fake_b

    with patch("secmcp.models.loader.load_model", side_effect=_fake_load) as mock_load:
        r_a = load_shared_model("a")
        r_b = load_shared_model("b")
        # Second call for each — should still be cached
        load_shared_model("a")
        load_shared_model("b")
    assert mock_load.call_count == 2
    assert r_a is fake_a
    assert r_b is fake_b
    clear_model_cache()


def test_clear_model_cache():
    clear_model_cache()
    fake = LoadedModel(name="x", model=None, tokenizer=None, cfg=SimpleNamespace())
    with patch("secmcp.models.loader.load_model", return_value=fake):
        load_shared_model("x")
        assert "x" in _MODEL_CACHE
        clear_model_cache()
        assert "x" not in _MODEL_CACHE


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("SECMCP_RUN_GPU_TESTS") != "1", reason="set SECMCP_RUN_GPU_TESTS=1 to load real models")
def test_load_model_real_mistral_smoke():
    loaded = load_model("mistral_7b_v03")
    assert loaded.model is not None
    assert loaded.tokenizer is not None
    assert loaded.cfg.hidden_dim == 4096
