from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class UnifiedSample:
    label: int
    text: str = ""
    source: str = ""
    kind: str = ""
    sample_type: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {self.label!r}")
        messages = [dict(m) for m in self.messages]
        text = self.text
        if not text and messages:
            text = format_messages(messages)
            object.__setattr__(self, "text", text)
        if not messages and text:
            messages = [{"role": "user", "content": text}]
            object.__setattr__(self, "messages", messages)
        if not self.text or not self.text.strip():
            raise ValueError("text or messages must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.kind:
            raise ValueError("kind must be non-empty")
        if not self.sample_type:
            raise ValueError("sample_type must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedSample":
        return cls(
            label=int(data["label"]),
            text=str(data.get("text") or ""),
            source=str(data["source"]),
            kind=str(data["kind"]),
            sample_type=str(data["sample_type"]),
            messages=list(data.get("messages") or []),
            tools=data.get("tools"),
            metadata=dict(data.get("metadata") or {}),
        )


def _content_to_str(content: Any) -> str:
    """Flatten message content to plain text.

    Handles three shapes:
    - str  → return as-is
    - list[dict{"type":"text","content":...}]  → AgentDojo / OpenAI-style content blocks
    - anything else → str()
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                # AgentDojo uses {"type": "text", "content": "..."}.
                # OpenAI uses {"type": "text", "text": "..."}.
                # Non-text blocks are intentionally skipped to avoid leaking
                # Python reprs for image/tool-use structures into training text.
                if "content" in item:
                    parts.append(_content_to_str(item.get("content")).strip())
                elif "text" in item:
                    parts.append(_content_to_str(item.get("text")).strip())
            elif isinstance(item, str):
                parts.append(str(item).strip())
        return " ".join(p for p in parts if p)
    return str(content)


def format_messages(messages: list[dict[str, Any]], role_separator: str = "\n[{role}]: ") -> str:
    """Format a list of message dicts as a single string for text-based processing."""
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or msg.get("from") or "unknown").upper()
        # Support both {"content": ...} and {"value": ...} (AgentTraj-L uses "value")
        raw = msg.get("content")
        if raw is None:
            raw = msg.get("value", "")
        content = _content_to_str(raw).strip()
        if not content:
            continue
        parts.append(f"{role_separator.format(role=role)}{content}")
    return "".join(parts).strip()


def normalize_messages_for_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return OpenAI-style messages with plain string content.

    AgentDojo often stores content as list[{"type": "text", ...}] blocks and
    AgentTraj-L uses human/gpt roles. HF chat templates expect a smaller role
    set and text content, so training/extraction paths should normalize before
    tokenization.
    """
    role_map = {"human": "user", "gpt": "assistant", "tool_result": "tool"}
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or msg.get("from") or "user")
        role = role_map.get(role, role)
        content = msg.get("content")
        if content is None:
            content = msg.get("value", "")
        new_msg = dict(msg)
        new_msg["role"] = role
        new_msg["content"] = _content_to_str(content)
        normalized.append(new_msg)
    return normalized


def format_key_values(items: dict[str, Any]) -> str:
    lines = []
    for key, value in items.items():
        if value is None or value == "":
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines).strip()


def balanced_cap(samples: list[UnifiedSample], max_samples: int | None) -> list[UnifiedSample]:
    """Cap sample list to max_samples while maintaining rough label balance."""
    if max_samples is None or len(samples) <= max_samples:
        return samples
    benign = [s for s in samples if s.label == 0]
    malicious = [s for s in samples if s.label == 1]
    half = max_samples // 2
    capped = benign[:half] + malicious[: max_samples - half]
    if len(capped) < max_samples:
        used = {id(s) for s in capped}
        capped.extend(s for s in samples if id(s) not in used)
    return capped[:max_samples]
