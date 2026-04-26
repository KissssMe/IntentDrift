from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from secmcp.data.schema import format_messages, normalize_messages_for_chat
from secmcp.models.truncate import truncate_messages


def normalize_chat_messages(text_or_messages: str | list[dict[str, Any]], cfg: SimpleNamespace) -> list[dict[str, Any]]:
    """Normalize text or chat messages before tokenization.

    Gemma/Mistral configs are marked no_system_role; if a caller passes system
    messages for those models, fold them into user text rather than emitting a
    system role that the chat template may reject.
    """
    if isinstance(text_or_messages, str):
        messages = [{"role": "user", "content": text_or_messages}]
    else:
        messages = normalize_messages_for_chat([dict(m) for m in text_or_messages])

    quirks = str(getattr(cfg, "chat_template_quirks", ""))
    if "no_system_role" not in quirks:
        return messages

    normalized: list[dict[str, Any]] = []
    pending_system: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if content:
                pending_system.append(str(content))
            continue
        if pending_system and role == "user":
            prefix = "\n".join(pending_system)
            msg = dict(msg)
            msg["content"] = f"{prefix}\n\n{content}" if content else prefix
            pending_system.clear()
        normalized.append(msg)
    # Any system messages that never found a following user turn are appended
    # at the end as a user message rather than inserted at position 0, which
    # would break role ordering for already-normalized turns.
    if pending_system:
        normalized.append({"role": "user", "content": "\n".join(pending_system)})
    return normalized


def prepare_inputs(
    tokenizer: Any,
    text_or_messages: str | list[dict[str, Any]],
    cfg: SimpleNamespace,
):
    messages = normalize_chat_messages(text_or_messages, cfg)
    messages = truncate_messages(messages, tokenizer, cfg)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        # Some benchmark traces have tool-role details that a given HF chat
        # template rejects. Fall back to a stable text rendering so extraction
        # remains consistent across training and runtime.
        rendered = format_messages(messages)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
    if hasattr(encoded, "to"):
        return encoded
    # Some tokenizers return a list when mocked or configured differently.
    import torch

    return torch.tensor([encoded], dtype=torch.long)


def _model_device(model: Any):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _move_to_device(inputs: Any, device: Any):
    if device is None or not hasattr(inputs, "to"):
        return inputs
    return inputs.to(device)


def validate_layers(layers: list[int], hidden_state_count: int) -> None:
    # hidden_states includes embeddings at index 0, so valid layer ids are
    # [0, hidden_state_count - 2] and are read from hidden_states[layer + 1].
    max_layer = hidden_state_count - 2
    invalid = [layer for layer in layers if layer < 0 or layer > max_layer]
    if invalid:
        raise ValueError(f"Layer indices out of range for hidden_states: {invalid}; max layer={max_layer}")


def extract_last_token_from_hidden_states(hidden_states: tuple[Any, ...], layers: list[int]):
    import torch

    validate_layers(layers, len(hidden_states))
    return torch.stack([hidden_states[layer + 1][0, -1].detach().cpu() for layer in layers])


def last_token_hidden_states(
    model: Any,
    tokenizer: Any,
    text_or_messages: str | list[dict[str, Any]],
    layers: list[int],
    cfg: SimpleNamespace,
):
    """Return stacked last-token hidden states with shape [len(layers), hidden_dim]."""
    import torch

    inputs = prepare_inputs(tokenizer, text_or_messages, cfg)
    inputs = _move_to_device(inputs, _model_device(model))
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True, use_cache=False)
    return extract_last_token_from_hidden_states(outputs.hidden_states, layers)
