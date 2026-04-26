from __future__ import annotations

import json
import random
from pathlib import Path

from secmcp.data.schema import UnifiedSample, format_messages, normalize_messages_for_chat


def load(
    root: Path,
    sample_frac: float = 0.25,
    seed: int = 42,
    max_samples: int | None = None,
) -> list[UnifiedSample]:
    rng = random.Random(seed)
    samples: list[UnifiedSample] = []
    for path in sorted(root.glob("*_train.json")):
        domain = path.stem.removesuffix("_train")
        rows = json.load(path.open(encoding="utf-8"))
        if sample_frac < 1.0:
            keep = max(1, int(len(rows) * sample_frac))
            indices = set(rng.sample(range(len(rows)), keep))
            selected = (row for idx, row in enumerate(rows) if idx in indices)
        else:
            selected = iter(rows)
        for idx, row in enumerate(selected):
            messages = normalize_messages_for_chat([
                {"role": turn.get("from", "unknown"), "content": turn.get("value", "")}
                for turn in row.get("conversations", [])
            ])
            text = format_messages(messages)
            if not text:
                continue
            samples.append(
                UnifiedSample(
                    label=0,
                    text=text,
                    messages=messages,
                    source="agenttraj_l",
                    kind="conversation",
                    sample_type=domain,
                    metadata={
                        "domain": domain,
                        "item_id": row.get("item_id"),
                        "num_turns": len(messages),
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples
