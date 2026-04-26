from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from secmcp.data.schema import UnifiedSample


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_samples_jsonl(path: Path, samples: Iterable[UnifiedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def read_samples_jsonl(path: Path) -> list[UnifiedSample]:
    return [UnifiedSample.from_dict(row) for row in iter_jsonl(path)]
