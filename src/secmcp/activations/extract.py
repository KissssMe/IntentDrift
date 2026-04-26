from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from secmcp.config import OUTPUTS_DIR
from secmcp.data.io import read_samples_jsonl
from secmcp.data.schema import UnifiedSample
from secmcp.models.hooks import last_token_hidden_states


@dataclass(frozen=True)
class ExtractionSummary:
    model_name: str
    split: str
    output_dir: Path
    total_samples: int
    skipped_samples: int
    extracted_samples: int
    shards_written: int


def split_path(split: str, splits_dir: Path | None = None) -> Path:
    root = splits_dir or OUTPUTS_DIR / "splits"
    return root / f"{split}.jsonl"


def activation_output_dir(model_name: str, split: str, output_root: Path | None = None) -> Path:
    root = output_root or OUTPUTS_DIR / "activations"
    return root / model_name / split


def shard_paths(output_dir: Path, shard_idx: int) -> tuple[Path, Path, Path]:
    stem = f"{shard_idx:05d}"
    return (
        output_dir / f"shard_{stem}.pt",
        output_dir / f"labels_{stem}.pt",
        output_dir / f"meta_{stem}.jsonl",
    )


def completed_shards(output_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in output_dir.glob("shard_*.pt"):
        try:
            idx = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        labels_path = output_dir / f"labels_{idx:05d}.pt"
        meta_path = output_dir / f"meta_{idx:05d}.jsonl"
        if labels_path.exists() and meta_path.exists():
            indices.append(idx)
    return sorted(indices)


def count_completed_samples(output_dir: Path) -> int:
    import torch

    total = 0
    for idx in completed_shards(output_dir):
        labels_path = output_dir / f"labels_{idx:05d}.pt"
        labels = torch.load(labels_path, map_location="cpu", weights_only=True)
        total += int(labels.shape[0])
    return total


def next_shard_index(output_dir: Path) -> int:
    indices = completed_shards(output_dir)
    return (max(indices) + 1) if indices else 0


def clear_activation_shards(output_dir: Path) -> None:
    for pattern in ("shard_*.pt", "labels_*.pt", "meta_*.jsonl"):
        for path in output_dir.glob(pattern):
            path.unlink()


def sample_meta(sample: UnifiedSample, sample_index: int) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "label": sample.label,
        "source": sample.source,
        "kind": sample.kind,
        "sample_type": sample.sample_type,
        "metadata": sample.metadata,
        "text_len": len(sample.text),
    }


def write_shard(
    output_dir: Path,
    shard_idx: int,
    activations: list[Any],
    labels: list[int],
    metas: list[dict[str, Any]],
) -> None:
    import torch

    if not activations:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path, labels_path, meta_path = shard_paths(output_dir, shard_idx)
    tensor = torch.stack([act.detach().cpu() for act in activations])
    label_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(tensor, activations_path)
    torch.save(label_tensor, labels_path)
    with meta_path.open("w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def iter_pending_samples(samples: list[UnifiedSample], skip: int) -> Iterable[tuple[int, UnifiedSample]]:
    for idx, sample in enumerate(samples):
        if idx < skip:
            continue
        yield idx, sample


def extract_split_activations(
    *,
    model_name: str,
    split: str,
    model: Any,
    tokenizer: Any,
    cfg: Any,
    layers: list[int],
    splits_dir: Path | None = None,
    output_root: Path | None = None,
    shard_size: int = 1000,
    resume: bool = True,
    show_progress: bool = False,
    log_every: int = 1,
    activation_fn: Callable[[Any, Any, str, list[int], Any], Any] = last_token_hidden_states,
) -> ExtractionSummary:
    samples = read_samples_jsonl(split_path(split, splits_dir))
    out_dir = activation_output_dir(model_name, split, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        clear_activation_shards(out_dir)
    skipped = count_completed_samples(out_dir) if resume else 0
    if skipped > len(samples):
        raise ValueError(f"Completed shard sample count {skipped} exceeds split size {len(samples)}")
    shard_idx = next_shard_index(out_dir) if resume else 0
    if show_progress:
        print(
            f"[extract] model={model_name} split={split} total={len(samples)} "
            f"skipped={skipped} output_dir={out_dir}",
            file=sys.stderr,
            flush=True,
        )

    activations: list[Any] = []
    labels: list[int] = []
    metas: list[dict[str, Any]] = []
    written = 0
    extracted = 0

    iterator = iter_pending_samples(samples, skipped)
    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(
            total=len(samples),
            initial=skipped,
            desc=f"{model_name}/{split}",
            unit="sample",
            dynamic_ncols=True,
            file=sys.stderr,
            disable=False,
            mininterval=1.0,
        )

    try:
        for sample_index, sample in iterator:
            if show_progress and log_every > 0 and (extracted == 0 or extracted % log_every == 0):
                print(
                    f"[extract] starting sample_index={sample_index} "
                    f"done={skipped + extracted}/{len(samples)} "
                    f"source={sample.source} label={sample.label} text_len={len(sample.text)}",
                    file=sys.stderr,
                    flush=True,
                )
            hidden = activation_fn(model, tokenizer, sample.text, layers, cfg)
            activations.append(hidden)
            labels.append(sample.label)
            metas.append(sample_meta(sample, sample_index))
            extracted += 1
            if progress is not None:
                progress.update(1)

            if len(activations) >= shard_size:
                write_shard(out_dir, shard_idx, activations, labels, metas)
                written += 1
                shard_idx += 1
                activations, labels, metas = [], [], []
                if progress is not None:
                    progress.set_postfix(shards=shard_idx)
    finally:
        if progress is not None:
            progress.close()

    if activations:
        write_shard(out_dir, shard_idx, activations, labels, metas)
        written += 1

    return ExtractionSummary(
        model_name=model_name,
        split=split,
        output_dir=out_dir,
        total_samples=len(samples),
        skipped_samples=skipped,
        extracted_samples=extracted,
        shards_written=written,
    )
