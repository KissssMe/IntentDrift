from __future__ import annotations

import json
from pathlib import Path

from secmcp.data.schema import UnifiedSample, balanced_cap, format_key_values


def _load_behavior_file(path: Path, label: int, sample_type: str) -> list[UnifiedSample]:
    obj = json.load(path.open(encoding="utf-8"))
    rows = obj.get("behaviors", [])
    samples: list[UnifiedSample] = []
    for row in rows:
        text = format_key_values(
            {
                "Prompt": row.get("prompt"),
                "Name": row.get("name"),
                "Target Functions": row.get("target_functions"),
            }
        )
        messages = [
            {
                "role": "system",
                "content": "You are a tool-using assistant. Refuse unsafe requests and follow applicable safety policy.",
            },
            {"role": "user", "content": str(row.get("prompt") or "")},
        ]
        samples.append(
            UnifiedSample(
                label=label,
                text=text,
                messages=messages,
                source="agentharm",
                kind="user_prompt",
                sample_type=sample_type,
                metadata={
                    "original_id": row.get("id"),
                    "id_original": row.get("id_original"),
                    "category": row.get("category"),
                    "grading_function": row.get("grading_function"),
                    "file": path.name,
                },
            )
        )
    return samples


def load(root: Path, max_samples: int | None = None) -> list[UnifiedSample]:
    benchmark = root / "benchmark"
    samples: list[UnifiedSample] = []
    for path in sorted(benchmark.glob("benign_behaviors_*.json")):
        samples.extend(_load_behavior_file(path, 0, path.stem))
    for path in sorted(benchmark.glob("harmful_behaviors_*.json")):
        samples.extend(_load_behavior_file(path, 1, path.stem))
    return balanced_cap(samples, max_samples)
