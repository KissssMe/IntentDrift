"""Phase 0 tests: truncate_messages head_tail strategy."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from secmcp.models.truncate import (
    TRUNCATION_MARKER,
    count_tokens,
    truncate_messages,
)

# ---------------------------------------------------------------------------
# Minimal fake tokenizer: each character == one token
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ):
        text = "".join(m.get("content", "") or "" for m in messages)
        ids = list(range(len(text)))
        return ids if tokenize else text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARKER_COST = len(f"\n{TRUNCATION_MARKER}")  # 19 with FakeTokenizer


def _cfg(detect_max_tokens: int = 100) -> SimpleNamespace:
    return SimpleNamespace(truncation="head_tail", detect_max_tokens=detect_max_tokens)


def _msgs(sizes: list[int]) -> list[dict]:
    roles = ["system", "user", "assistant", "tool"]
    return [
        {"role": roles[i % len(roles)], "content": "x" * sz}
        for i, sz in enumerate(sizes)
    ]


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_no_truncation_when_short():
    """Messages that fit within budget are returned unchanged."""
    tok = FakeTokenizer()
    msgs = _msgs([10, 20, 30])
    assert truncate_messages(msgs, tok, _cfg(200)) is msgs


def test_output_strictly_within_budget():
    """Output token count must be <= detect_max_tokens — including marker."""
    tok = FakeTokenizer()
    # 3 messages × 100 chars = 300 tokens; budget 100
    msgs = _msgs([100, 100, 100])
    result = truncate_messages(msgs, tok, _cfg(100))
    assert count_tokens(result, tok) <= 100, (
        f"Output exceeded budget: {count_tokens(result, tok)} > 100"
    )


def test_output_within_budget_various():
    """Parametrize over several (sizes, budget) combos to stress-test strict enforcement."""
    tok = FakeTokenizer()
    cases = [
        ([20, 20, 20, 20, 20], 50),
        ([100, 100, 100], 80),
        ([10, 10, 10, 10, 10, 10, 10, 10], 50),
        ([500], 100),
        ([30, 30, 30], 60),
    ]
    for sizes, budget in cases:
        msgs = _msgs(sizes)
        result = truncate_messages(msgs, tok, _cfg(budget))
        total = count_tokens(result, tok)
        assert total <= budget, (
            f"sizes={sizes}, budget={budget}: output={total} > budget"
        )


def test_truncation_marker_in_content():
    """Marker must appear in some message's content (not as a new role message)."""
    tok = FakeTokenizer()
    msgs = _msgs([20, 20, 20, 20, 20])
    result = truncate_messages(msgs, tok, _cfg(50))
    all_content = " ".join(m.get("content", "") or "" for m in result)
    assert TRUNCATION_MARKER in all_content


def test_no_extra_system_role_inserted():
    """truncate_messages must not introduce new system-role messages."""
    tok = FakeTokenizer()
    msgs = _msgs([20, 20, 20, 20, 20])
    original_n_system = sum(1 for m in msgs if m["role"] == "system")
    result = truncate_messages(msgs, tok, _cfg(50))
    result_n_system = sum(1 for m in result if m["role"] == "system")
    assert result_n_system <= original_n_system


# ---------------------------------------------------------------------------
# Head / tail preservation
# ---------------------------------------------------------------------------


def test_head_preserved():
    """First message should appear (possibly with marker appended) at result[0].

    Budget chosen so that the first message (10 chars) fits entirely in the
    head allocation: content_budget = 70-19 = 51, head_budget = 12 > 10.
    """
    tok = FakeTokenizer()
    cfg = _cfg(detect_max_tokens=70)        # 8 × 10 = 80 > 70, needs truncation
    msgs = _msgs([10, 10, 10, 10, 10, 10, 10, 10])
    result = truncate_messages(msgs, tok, cfg)
    # result[0] keeps msgs[0] content; marker is appended, so it *starts with* original
    assert result[0]["content"].startswith(msgs[0]["content"])


def test_tail_preserved():
    """Last message must appear in the truncated tail."""
    tok = FakeTokenizer()
    cfg = _cfg(detect_max_tokens=60)        # 8 × 10 = 80 > 60
    msgs = _msgs([10, 10, 10, 10, 10, 10, 10, 10])
    result = truncate_messages(msgs, tok, cfg)
    last_content = msgs[-1]["content"]
    result_contents = [m.get("content", "") or "" for m in result]
    assert any(last_content in c for c in result_contents)


# ---------------------------------------------------------------------------
# Single oversized message
# ---------------------------------------------------------------------------


def test_single_large_message_strictly_within_budget():
    """A 1000-char message with budget=100 must produce output <= 100 tokens."""
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "x" * 1000}]
    result = truncate_messages(msgs, tok, _cfg(100))
    assert count_tokens(result, tok) <= 100


def test_single_large_message_nonempty():
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "y" * 500}]
    result = truncate_messages(msgs, tok, _cfg(50))
    assert len(result) >= 1
    assert len(result[0].get("content", "")) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_marker_when_fits():
    tok = FakeTokenizer()
    msgs = _msgs([5, 5, 5])
    result = truncate_messages(msgs, tok, _cfg(500))
    all_content = " ".join(m.get("content", "") or "" for m in result)
    assert TRUNCATION_MARKER not in all_content


def test_non_head_tail_passthrough():
    """cfg.truncation != 'head_tail' returns messages unchanged."""
    tok = FakeTokenizer()
    cfg = SimpleNamespace(truncation="none", detect_max_tokens=10)
    msgs = _msgs([100, 100])
    assert truncate_messages(msgs, tok, cfg) is msgs


def test_empty_messages():
    tok = FakeTokenizer()
    assert truncate_messages([], tok, _cfg(50)) == []


def test_single_small_message_unchanged():
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "x" * 30}]
    assert truncate_messages(msgs, tok, _cfg(50)) == msgs


# ---------------------------------------------------------------------------
# count_tokens helper
# ---------------------------------------------------------------------------


def test_count_tokens_single():
    tok = FakeTokenizer()
    assert count_tokens([{"role": "user", "content": "hello"}], tok) == 5


def test_count_tokens_multiple():
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]
    assert count_tokens(msgs, tok) == 5


def test_count_tokens_empty():
    tok = FakeTokenizer()
    assert count_tokens([], tok) == 0
