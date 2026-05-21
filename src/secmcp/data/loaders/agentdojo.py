from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from secmcp.data.schema import UnifiedSample, format_messages, normalize_messages_for_chat


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _injection_text(obj: dict[str, Any]) -> str:
    values = obj.get("injections") or {}
    if not values:
        return ""
    return "\n".join(str(v) for _, v in sorted(values.items())).strip()


def _injection_fragments(text: str, min_len: int = 40) -> list[str]:
    fragments = [part.strip() for part in re.split(r"\n+", text) if len(part.strip()) >= min_len]
    return fragments or ([text.strip()] if text.strip() else [])


def _label(obj: dict[str, Any]) -> int:
    """Positive only when the run actually delivered an injection.

    Previously this returned 1 for any ``user_task_id`` starting with
    ``injection_task_`` even when the run carried no ``injections`` content
    (e.g. clean control reruns of an injection scenario). Those samples have no
    locatable tool step and only contaminate the step-level supervision.
    """
    has_injection_content = bool(obj.get("injections"))
    has_injection_tag = bool(obj.get("injection_task_id") or obj.get("attack_type"))
    return 1 if (has_injection_content and has_injection_tag) else 0


def _malicious_tool_indices(messages: list[dict[str, Any]], fragments: list[str]) -> list[int]:
    """Indices of tool messages whose content contains any injection fragment.

    Returned in message order; downstream code uses ``min(indices)`` as the
    first hijack point. Empty list when no fragment matches — those positives
    will be dropped by ``drop_unlocatable_positives``.
    """
    if not fragments:
        return []
    matched: list[int] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        if any(fragment in content for fragment in fragments):
            matched.append(idx)
    return matched


def load(root: Path, max_samples: int | None = None, pipeline: str | None = None) -> list[UnifiedSample]:
    samples: list[UnifiedSample] = []
    runs_root = root / "runs"
    pattern = f"{pipeline}/**/*.json" if pipeline else "**/*.json"
    for path in sorted(runs_root.glob(pattern)):
        obj = json.load(path.open(encoding="utf-8"))
        messages = normalize_messages_for_chat(obj.get("messages") or [])
        text = format_messages(messages)
        if not text:
            continue
        label = _label(obj)
        injection_text = _injection_text(obj)
        injection_fragments = _injection_fragments(injection_text)
        malicious_tool_indices = (
            _malicious_tool_indices(messages, injection_fragments) if label == 1 else []
        )
        if injection_text:
            split_group = f"injection:{_hash_text(injection_text)}"
        else:
            split_group = f"user:{obj.get('suite_name')}:{obj.get('user_task_id')}"
        samples.append(
            UnifiedSample(
                label=label,
                text=text,
                messages=messages,
                source="agentdojo",
                kind="conversation",
                sample_type=str(obj.get("attack_type") or "none"),
                metadata={
                    "suite_name": obj.get("suite_name"),
                    "pipeline_name": obj.get("pipeline_name"),
                    "user_task_id": obj.get("user_task_id"),
                    "injection_task_id": obj.get("injection_task_id"),
                    "attack_type": obj.get("attack_type"),
                    "utility": obj.get("utility"),
                    "security": obj.get("security"),
                    "num_turns": len(messages),
                    "file": str(path.relative_to(root)),
                    "injection_hash": _hash_text(injection_text) if injection_text else None,
                    "injection_text": injection_text,
                    "injection_fragments": injection_fragments,
                    "malicious_tool_message_indices": malicious_tool_indices,
                    "split_group": split_group,
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            return samples
    return samples
