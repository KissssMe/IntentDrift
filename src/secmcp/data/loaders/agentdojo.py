from __future__ import annotations

import hashlib
import json
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


def _label(obj: dict[str, Any]) -> int:
    user_task_id = str(obj.get("user_task_id") or "")
    if obj.get("injection_task_id") or obj.get("attack_type"):
        return 1
    if user_task_id.startswith("injection_task_"):
        return 1
    return 0


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
                    "split_group": split_group,
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            return samples
    return samples
