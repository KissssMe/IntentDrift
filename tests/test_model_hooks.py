from __future__ import annotations

from types import SimpleNamespace

import pytest

from secmcp.models.hooks import (
    extract_last_token_from_hidden_states,
    last_token_hidden_states,
    normalize_chat_messages,
    prepare_inputs,
)


torch = pytest.importorskip("torch")


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(max(1, len(text))))

    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=False, tokenize=True):
        self.last_messages = messages
        length = max(1, sum(len(str(m.get("content", ""))) for m in messages))
        return torch.arange(length, dtype=torch.long).unsqueeze(0)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, n_layers: int = 4, hidden_dim: int = 3):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.called_eval = False

    def __call__(self, inputs, output_hidden_states=False, use_cache=False):
        assert output_hidden_states is True
        seq_len = inputs.shape[-1]
        hidden_states = []
        for layer in range(self.n_layers + 1):
            base = torch.arange(seq_len * self.hidden_dim, dtype=torch.float32).reshape(1, seq_len, self.hidden_dim)
            hidden_states.append(base + layer * 100)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def _cfg(**overrides):
    data = {
        "truncation": "head_tail",
        "detect_max_tokens": 1000,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_normalize_chat_messages_text():
    assert normalize_chat_messages("hello", _cfg()) == [{"role": "user", "content": "hello"}]


def test_normalize_chat_messages_no_system_role_folds_system_into_user():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
    ]
    out = normalize_chat_messages(messages, _cfg(chat_template_quirks="no_system_role"))
    assert [m["role"] for m in out] == ["user"]
    assert "system prompt" in out[0]["content"]
    assert "question" in out[0]["content"]


def test_normalize_chat_messages_trailing_system_appended_at_end():
    """Trailing system message with no following user turn must appear AFTER
    already-normalized turns, not be inserted at position 0."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "system", "content": "trailing note"},
    ]
    out = normalize_chat_messages(messages, _cfg(chat_template_quirks="no_system_role"))
    # The trailing system content should appear in the LAST message
    roles = [m["role"] for m in out]
    assert "trailing note" in out[-1]["content"]
    # Existing user/assistant order must be preserved before it
    assert roles[0] == "user"
    assert roles[1] == "assistant"


def test_prepare_inputs_calls_chat_template():
    tokenizer = FakeTokenizer()
    inputs = prepare_inputs(tokenizer, "hello", _cfg())
    assert tuple(inputs.shape) == (1, 5)
    assert tokenizer.last_messages == [{"role": "user", "content": "hello"}]


def test_extract_last_token_from_hidden_states_shape_and_values():
    hidden_states = tuple(torch.ones(1, 2, 4) * i for i in range(4))
    out = extract_last_token_from_hidden_states(hidden_states, [0, 2])
    assert tuple(out.shape) == (2, 4)
    assert torch.all(out[0] == 1)
    assert torch.all(out[1] == 3)


def test_extract_last_token_rejects_bad_layer():
    hidden_states = tuple(torch.ones(1, 2, 4) for _ in range(3))
    with pytest.raises(ValueError, match="out of range"):
        extract_last_token_from_hidden_states(hidden_states, [2])


def test_last_token_hidden_states_end_to_end_fake():
    model = FakeModel(n_layers=4, hidden_dim=3)
    tokenizer = FakeTokenizer()
    out = last_token_hidden_states(model, tokenizer, "hello", [0, 3], _cfg())
    assert tuple(out.shape) == (2, 3)
    assert torch.all(out[0] >= 100)
    assert torch.all(out[1] >= 400)
