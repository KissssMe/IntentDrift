from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secmcp.activations.drift import completed_drift_shards, drift_output_dir, drift_shard_paths
from secmcp.config import OUTPUTS_DIR


@dataclass(frozen=True)
class DriftActivationSplit:
    task: Any
    history: Any
    post: Any
    labels: Any
    metas: list[dict[str, Any]]


def read_meta_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_drift_split(model_name: str, split: str, root: Path | None = None) -> DriftActivationSplit:
    import torch

    out_dir = drift_output_dir(model_name, split, root or OUTPUTS_DIR / "drift_activations")
    shard_indices = completed_drift_shards(out_dir)
    if not shard_indices:
        raise FileNotFoundError(f"No complete drift activation shards found in {out_dir}")

    task = []
    history = []
    post = []
    labels = []
    metas: list[dict[str, Any]] = []
    for idx in shard_indices:
        drift_path, label_path, meta_path = drift_shard_paths(out_dir, idx)
        payload = torch.load(drift_path, map_location="cpu", weights_only=True)
        y = torch.load(label_path, map_location="cpu", weights_only=True)
        meta_rows = read_meta_jsonl(meta_path)
        n = int(y.shape[0])
        if payload["task"].shape[0] != n or payload["history"].shape[0] != n or payload["post"].shape[0] != n:
            raise ValueError(f"Shard {idx:05d} activation count mismatch")
        if len(meta_rows) != n:
            raise ValueError(f"Shard {idx:05d} metadata count mismatch")
        task.append(payload["task"])
        history.append(payload["history"])
        post.append(payload["post"])
        labels.append(y)
        metas.extend(meta_rows)

    return DriftActivationSplit(
        task=torch.cat(task, dim=0),
        history=torch.cat(history, dim=0),
        post=torch.cat(post, dim=0),
        labels=torch.cat(labels, dim=0).long(),
        metas=metas,
    )
