"""head_tail truncation for long agent contexts.

Strategy:
  - Reserve marker_cost tokens for the truncation marker upfront.
  - Head (25% of remaining budget): taken from the front; first message that
    overflows is content-truncated via binary search on characters.
  - Tail (75% of remaining budget): taken from the back, same logic.
  - Truncation marker is folded into adjacent message *content* — no extra
    role="system" message is inserted, keeping chat-template compatibility
    for models that forbid system roles (Mistral, Gemma).
  - Output is guaranteed to be <= detect_max_tokens tokens.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

TRUNCATION_MARKER = "[... truncated ...]"
_HEAD_FRAC = 0.25
_MIN_REMAINING = 10  # skip content-truncation when remaining budget < this


def _count_tokens_text(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_tokens_msg(msg: dict, tokenizer: Any) -> int:
    return _count_tokens_text(msg.get("content") or "", tokenizer)


def _truncate_content(msg: dict, tokenizer: Any, token_budget: int) -> dict:
    """Return a copy of *msg* with content trimmed to fit *token_budget* tokens.

    Binary-searches on character index — works for any tokenizer.
    """
    content = msg.get("content") or ""
    if not content or token_budget <= 0:
        return {**msg, "content": ""}
    if _count_tokens_text(content, tokenizer) <= token_budget:
        return msg
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens_text(content[:mid], tokenizer) <= token_budget:
            lo = mid
        else:
            hi = mid - 1
    return {**msg, "content": content[:lo]}


def truncate_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    cfg: SimpleNamespace,
) -> list[dict[str, Any]]:
    """Return messages fitting within cfg.detect_max_tokens (head_tail strategy).

    The output is guaranteed: sum of per-message token counts <= detect_max_tokens.
    """
    if cfg.truncation != "head_tail":
        return messages
    if not messages:
        return messages

    max_tokens: int = cfg.detect_max_tokens
    total = sum(_count_tokens_msg(m, tokenizer) for m in messages)
    if total <= max_tokens:
        return messages

    # Reserve space for the marker that will be injected at the boundary.
    # Use the longer form (with newline) to be conservative.
    marker_text_head = f"\n{TRUNCATION_MARKER}"   # appended to last head msg
    marker_text_tail = f"{TRUNCATION_MARKER}\n"   # prepended to first tail msg
    marker_cost = _count_tokens_text(marker_text_head, tokenizer)

    content_budget = max(1, max_tokens - marker_cost)
    head_budget = max(1, int(content_budget * _HEAD_FRAC))
    tail_budget = content_budget - head_budget

    # ── Head: consume from front ────────────────────────────────────────────
    head_msgs: list[dict] = []
    head_used = 0
    head_end_idx = 0
    for i, msg in enumerate(messages):
        cost = _count_tokens_msg(msg, tokenizer)
        remaining = head_budget - head_used
        if cost <= remaining:
            head_msgs.append(msg)
            head_used += cost
            head_end_idx = i + 1
        elif remaining >= _MIN_REMAINING:
            head_msgs.append(_truncate_content(msg, tokenizer, remaining))
            head_end_idx = i + 1
            break
        else:
            break

    # ── Tail: consume from back, skip already-head messages ─────────────────
    tail_msgs: list[dict] = []
    tail_used = 0
    for i in range(len(messages) - 1, head_end_idx - 1, -1):
        msg = messages[i]
        cost = _count_tokens_msg(msg, tokenizer)
        remaining = tail_budget - tail_used
        if cost <= remaining:
            tail_msgs.insert(0, msg)
            tail_used += cost
        elif remaining >= _MIN_REMAINING:
            tail_msgs.insert(0, _truncate_content(msg, tokenizer, remaining))
            break
        else:
            break

    # ── Attach marker to content boundary — no new role message ─────────────
    if head_msgs:
        last = dict(head_msgs[-1])
        last["content"] = (last.get("content") or "").rstrip() + marker_text_head
        head_msgs[-1] = last
    elif tail_msgs:
        first = dict(tail_msgs[0])
        first["content"] = marker_text_tail + (first.get("content") or "").lstrip()
        tail_msgs[0] = first

    result = head_msgs + tail_msgs

    # Safety fallback: both loops produced nothing (e.g. single msg << _MIN_REMAINING)
    if not result:
        return [_truncate_content(messages[-1], tokenizer, max_tokens)]

    return result


def count_tokens(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    """Sum per-message token counts (best-effort, no chat-template overhead)."""
    return sum(_count_tokens_msg(m, tokenizer) for m in messages)
