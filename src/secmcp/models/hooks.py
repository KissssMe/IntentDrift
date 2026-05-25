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
        if isinstance(inputs, dict):
            return {key: _move_to_device(value, device) for key, value in inputs.items()}
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


# ── task-anchored hidden states ────────────────────────────────────────────
#
# In a causal LM, hidden states at position t see only tokens 0..t. So you
# cannot probe how "context that comes later" affects the representation of an
# earlier task token directly. The trick (Abdelnabi 2024, TaskTracker) is to
# *re-append* the task description at the end of the rendered context and
# probe the appended copy. Those tokens sit at the tail and therefore attend
# to all preceding history (system, original user task, assistant turns, tool
# outputs); the difference between probes under different prefixes isolates
# how added context shifts the model's understanding of the task.

TASK_ANCHOR_SEPARATOR = "\n\n[Recap of original user task]: "
TASK_ANCHOR_EXTRACTION_MODE = "task_anchor_mean"


def _apply_chat_template_to_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    if hasattr(out, "tolist"):
        out = out.tolist()
    if isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    return [int(t) for t in out]


def _encode_text_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(t) for t in ids]


def task_anchored_hidden_states(
    model: Any,
    tokenizer: Any,
    prefix_messages: list[dict[str, Any]],
    task_text: str,
    layers: list[int],
    cfg: SimpleNamespace,
):
    """Probe hidden states of the user-task tokens re-appended at the end of `prefix_messages`.

    Output shape: ``[len(layers), hidden_dim]`` — mean-pooled across the
    appended task tokens at each requested layer.

    If ``task_text`` is empty (degenerate trajectory) we fall back to the
    last-token hidden state of the rendered prefix so the runtime detector
    keeps working rather than crashing.
    """
    import torch

    if not task_text:
        return last_token_hidden_states(model, tokenizer, prefix_messages, layers, cfg)

    normalized = normalize_chat_messages(prefix_messages, cfg)

    sep_ids = _encode_text_ids(tokenizer, TASK_ANCHOR_SEPARATOR)
    task_ids = _encode_text_ids(tokenizer, task_text)
    if not task_ids:
        return last_token_hidden_states(model, tokenizer, prefix_messages, layers, cfg)

    # Reserve headroom for separator + task tokens so head_tail truncation
    # never eats into the appended anchor.
    base_max = int(getattr(cfg, "detect_max_tokens", 1024))
    headroom = len(sep_ids) + len(task_ids) + 8
    truncate_cfg = SimpleNamespace(**{**vars(cfg), "detect_max_tokens": max(64, base_max - headroom)})
    truncated = truncate_messages(normalized, tokenizer, truncate_cfg)

    try:
        prefix_ids = _apply_chat_template_to_ids(tokenizer, truncated, add_generation_prompt=False)
    except Exception:
        rendered = format_messages(truncated)
        prefix_ids = _apply_chat_template_to_ids(
            tokenizer,
            [{"role": "user", "content": rendered}],
            add_generation_prompt=False,
        )

    full_ids = prefix_ids + sep_ids + task_ids
    task_token_count = len(task_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    input_ids = _move_to_device(input_ids, _model_device(model))
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)

    validate_layers(layers, len(outputs.hidden_states))
    pooled = [
        outputs.hidden_states[layer + 1][0, -task_token_count:].mean(dim=0).detach().cpu()
        for layer in layers
    ]
    return torch.stack(pooled)


def _pad_token_id(tokenizer: Any, cfg: SimpleNamespace) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        token_id = getattr(cfg, "pad_token_id", None)
    return int(token_id) if token_id is not None else 0


def _task_anchor_ids(
    tokenizer: Any,
    prefix_messages: list[dict[str, Any]],
    task_text: str,
    cfg: SimpleNamespace,
) -> tuple[list[int], int] | None:
    if not task_text:
        return None

    normalized = normalize_chat_messages(prefix_messages, cfg)
    sep_ids = _encode_text_ids(tokenizer, TASK_ANCHOR_SEPARATOR)
    task_ids = _encode_text_ids(tokenizer, task_text)
    if not task_ids:
        return None

    base_max = int(getattr(cfg, "detect_max_tokens", 1024))
    headroom = len(sep_ids) + len(task_ids) + 8
    truncate_cfg = SimpleNamespace(**{**vars(cfg), "detect_max_tokens": max(64, base_max - headroom)})
    truncated = truncate_messages(normalized, tokenizer, truncate_cfg)

    try:
        prefix_ids = _apply_chat_template_to_ids(tokenizer, truncated, add_generation_prompt=False)
    except Exception:
        rendered = format_messages(truncated)
        prefix_ids = _apply_chat_template_to_ids(
            tokenizer,
            [{"role": "user", "content": rendered}],
            add_generation_prompt=False,
        )

    return prefix_ids + sep_ids + task_ids, len(task_ids)


def task_anchored_hidden_states_batch(
    model: Any,
    tokenizer: Any,
    requests: list[tuple[list[dict[str, Any]], str]],
    layers: list[int],
    cfg: SimpleNamespace,
) -> list[Any]:
    """Batch version of task_anchored_hidden_states.

    Returns one ``[len(layers), hidden_dim]`` tensor per request. Requests with
    an empty task text fall back to the scalar implementation, preserving the
    existing degenerate behavior.
    """
    import torch

    if not requests:
        return []

    prepared: list[tuple[list[int], int] | None] = [
        _task_anchor_ids(tokenizer, prefix_messages, task_text, cfg)
        for prefix_messages, task_text in requests
    ]

    batch_items = [(idx, item) for idx, item in enumerate(prepared) if item is not None]
    results: list[Any | None] = [None] * len(requests)

    if batch_items:
        pad_token_id = _pad_token_id(tokenizer, cfg)
        max_len = max(len(ids) for _, (ids, _) in batch_items)
        input_rows = []
        mask_rows = []
        lengths = []
        task_token_counts = []
        for _, (ids, task_token_count) in batch_items:
            pad_len = max_len - len(ids)
            input_rows.append(ids + [pad_token_id] * pad_len)
            mask_rows.append([1] * len(ids) + [0] * pad_len)
            lengths.append(len(ids))
            task_token_counts.append(task_token_count)

        inputs = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }
        inputs = _move_to_device(inputs, _model_device(model))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        validate_layers(layers, len(outputs.hidden_states))
        for row_idx, (request_idx, _) in enumerate(batch_items):
            seq_len = lengths[row_idx]
            task_token_count = task_token_counts[row_idx]
            pooled = [
                outputs.hidden_states[layer + 1][row_idx, seq_len - task_token_count : seq_len].mean(dim=0).detach().cpu()
                for layer in layers
            ]
            results[request_idx] = torch.stack(pooled)

    for idx, item in enumerate(prepared):
        if item is None:
            prefix_messages, task_text = requests[idx]
            results[idx] = task_anchored_hidden_states(model, tokenizer, prefix_messages, task_text, layers, cfg)

    if any(result is None for result in results):
        raise RuntimeError("Missing task-anchor batch result")
    return [result for result in results if result is not None]
