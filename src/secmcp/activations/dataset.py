from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secmcp.config import OUTPUTS_DIR
from secmcp.activations.extract import completed_shards, shard_paths


@dataclass(frozen=True)
class ActivationSplit:
    activations: Any
    labels: Any
    metas: list[dict[str, Any]]


def activation_dir(model_name: str, split: str, root: Path | None = None) -> Path:
    return (root or OUTPUTS_DIR / "activations") / model_name / split


def read_meta_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activation_split(
    model_name: str,
    split: str,
    root: Path | None = None,
    show_progress: bool = False,
) -> ActivationSplit:
    import torch

    out_dir = activation_dir(model_name, split, root)
    shard_indices = completed_shards(out_dir)
    if not shard_indices:
        raise FileNotFoundError(f"No complete activation shards found in {out_dir}")
    if show_progress:
        print(
            f"[load] activation split={split} shards={len(shard_indices)} dir={out_dir}",
            file=sys.stderr,
            flush=True,
        )

    activations = []
    labels = []
    metas: list[dict[str, Any]] = []
    iterator = shard_indices
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(
            shard_indices,
            desc=f"load {model_name}/{split}",
            unit="shard",
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=1.0,
        )
    for idx in iterator:
        act_path, label_path, meta_path = shard_paths(out_dir, idx)
        x = torch.load(act_path, map_location="cpu", weights_only=True)
        y = torch.load(label_path, map_location="cpu", weights_only=True)
        meta_rows = read_meta_jsonl(meta_path)
        if x.shape[0] != y.shape[0] or x.shape[0] != len(meta_rows):
            raise ValueError(
                f"Shard {idx:05d} count mismatch: activations={x.shape[0]} "
                f"labels={y.shape[0]} meta={len(meta_rows)}"
            )
        activations.append(x)
        labels.append(y)
        metas.extend(meta_rows)
    if show_progress:
        print(
            f"[load] activation split={split} samples={sum(int(y.shape[0]) for y in labels)}",
            file=sys.stderr,
            flush=True,
        )

    return ActivationSplit(
        activations=torch.cat(activations, dim=0),
        labels=torch.cat(labels, dim=0).long(),
        metas=metas,
    )
