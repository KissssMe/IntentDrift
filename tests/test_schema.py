from __future__ import annotations

import pytest

from secmcp.data.schema import UnifiedSample, format_messages, _content_to_str


def test_sample_roundtrip():
    sample = UnifiedSample(
        label=1,
        text="attack",
        source="unit",
        kind="conversation",
        sample_type="case",
        metadata={"id": 1},
    )
    assert UnifiedSample.from_dict(sample.to_dict()) == sample


def test_invalid_label_rejected():
    with pytest.raises(ValueError, match="label"):
        UnifiedSample(label=2, text="x", source="s", kind="k", sample_type="t")


def test_format_messages_roles():
    text = format_messages([
        {"role": "user", "content": "hello"},
        {"from": "gpt", "value": "hi"},
    ])
    assert "[USER]:" in text
    assert "[GPT]:" in text


def test_content_to_str_plain():
    assert _content_to_str("hello") == "hello"


def test_content_to_str_none():
    assert _content_to_str(None) == ""


def test_content_to_str_agentdojo_list():
    """AgentDojo uses list[{"type": "text", "content": "..."}] — must flatten to plain text."""
    content = [{"type": "text", "content": "You are an AI assistant."}]
    result = _content_to_str(content)
    assert result == "You are an AI assistant."
    # Must NOT look like a Python repr
    assert "[{" not in result
    assert "type" not in result


def test_content_to_str_openai_list():
    """OpenAI uses list[{"type": "text", "text": "..."}]."""
    content = [{"type": "text", "text": "Hello world"}]
    result = _content_to_str(content)
    assert result == "Hello world"


def test_content_to_str_skips_non_text_blocks():
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "tool_use", "id": "call_1", "name": "search"},
        {"type": "text", "text": "keep me"},
    ]
    result = _content_to_str(content)
    assert result == "keep me"
    assert "image_url" not in result
    assert "tool_use" not in result


def test_format_messages_with_list_content():
    """format_messages must produce readable text when content is list[dict]."""
    messages = [
        {"role": "system", "content": [{"type": "text", "content": "System prompt here."}]},
        {"role": "user", "content": [{"type": "text", "content": "User question."}]},
    ]
    text = format_messages(messages)
    assert "System prompt here." in text
    assert "User question." in text
    assert "[{" not in text  # no Python repr garbage
