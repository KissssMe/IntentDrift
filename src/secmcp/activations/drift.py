from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from secmcp.config import OUTPUTS_DIR
from secmcp.data.io import read_samples_jsonl
from secmcp.data.schema import UnifiedSample, normalize_messages_for_chat
from secmcp.models.hooks import last_token_hidden_states


@dataclass(frozen=True)
class ToolStep:
    sample_index: int
    step_index: int
    task_prefix: list[dict[str, Any]]
    history_prefix: list[dict[str, Any]]
    post_tool_prefix: list[dict[str, Any]]
    label: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class DriftExtractionSummary:
    model_name: str
    split: str
    output_dir: Path
    total_samples: int
    total_steps: int
    skipped_steps: int
    extracted_steps: int
    shards_written: int


def drift_output_dir(model_name: str, split: str, output_root: Path | None = None) -> Path:
    root = output_root or OUTPUTS_DIR / "drift_activations"
    return root / model_name / split


def drift_shard_paths(output_dir: Path, shard_idx: int) -> tuple[Path, Path, Path]:
    stem = f"{shard_idx:05d}"
    return (
        output_dir / f"drift_{stem}.pt",
        output_dir / f"labels_{stem}.pt",
        output_dir / f"meta_{stem}.jsonl",
    )


def completed_drift_shards(output_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in output_dir.glob("drift_*.pt"):
        try:
            idx = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (output_dir / f"labels_{idx:05d}.pt").exists() and (output_dir / f"meta_{idx:05d}.jsonl").exists():
            indices.append(idx)
    return sorted(indices)


def count_completed_steps(output_dir: Path) -> int:
    import torch

    total = 0
    for idx in completed_drift_shards(output_dir):
        _, labels_path, _ = drift_shard_paths(output_dir, idx)
        labels = torch.load(labels_path, map_location="cpu", weights_only=True)
        total += int(labels.shape[0])
    return total


def next_drift_shard_index(output_dir: Path) -> int:
    indices = completed_drift_shards(output_dir)
    return (max(indices) + 1) if indices else 0


def clear_drift_shards(output_dir: Path) -> None:
    for pattern in ("drift_*.pt", "labels_*.pt", "meta_*.jsonl"):
        for path in output_dir.glob(pattern):
            path.unlink()


def _task_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            prefix.append(msg)
            continue
        if msg.get("role") == "user":
            prefix.append(msg)
            break
    return prefix or messages[:1]


def _tool_name(msg: dict[str, Any]) -> str | None:
    tool_call = msg.get("tool_call") or {}
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            return function.get("name")
        if isinstance(function, str):
            return function
    return msg.get("name")


def tool_steps_for_sample(sample: UnifiedSample, sample_index: int) -> list[ToolStep]:
    messages = normalize_messages_for_chat(sample.messages)
    task_prefix = _task_prefix(messages)
    steps: list[ToolStep] = []
    step_index = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        meta = {
            "sample_index": sample_index,
            "step_index": step_index,
            "tool_message_index": idx,
            "label": sample.label,
            "source": sample.source,
            "kind": sample.kind,
            "sample_type": sample.sample_type,
            "tool_name": _tool_name(msg),
            "metadata": sample.metadata,
        }
        steps.append(
            ToolStep(
                sample_index=sample_index,
                step_index=step_index,
                task_prefix=task_prefix,
                history_prefix=messages[:idx],
                post_tool_prefix=messages[: idx + 1],
                label=sample.label,
                meta=meta,
            )
        )
        step_index += 1
    return steps


def iter_tool_steps(samples: list[UnifiedSample]) -> Iterable[ToolStep]:
    for sample_index, sample in enumerate(samples):
        yield from tool_steps_for_sample(sample, sample_index)


def write_drift_shard(output_dir: Path, shard_idx: int, records: list[dict[str, Any]]) -> None:
    import torch

    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    drift_path, labels_path, meta_path = drift_shard_paths(output_dir, shard_idx)
    torch.save(
        {
            "task": torch.stack([r["task"].detach().cpu() for r in records]),
            "history": torch.stack([r["history"].detach().cpu() for r in records]),
            "post": torch.stack([r["post"].detach().cpu() for r in records]),
        },
        drift_path,
    )
    torch.save(torch.tensor([int(r["label"]) for r in records], dtype=torch.long), labels_path)
    with meta_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record["meta"], ensure_ascii=False) + "\n")


def extract_split_drift_activations(
    *,
    model_name: str,
    split: str,
    model: Any,
    tokenizer: Any,
    cfg: Any,
    layers: list[int],
    splits_dir: Path | None = None,
    output_root: Path | None = None,
    shard_size: int = 500,
    resume: bool = True,
    show_progress: bool = False,
    activation_fn: Callable[[Any, Any, list[dict[str, Any]], list[int], Any], Any] = last_token_hidden_states,
) -> DriftExtractionSummary:
    root = splits_dir or OUTPUTS_DIR / "splits"
    samples = read_samples_jsonl(root / f"{split}.jsonl")
    steps = list(iter_tool_steps(samples))
    out_dir = drift_output_dir(model_name, split, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        clear_drift_shards(out_dir)
    skipped = count_completed_steps(out_dir) if resume else 0
    if skipped > len(steps):
        raise ValueError(f"Completed step count {skipped} exceeds step count {len(steps)}")

    shard_idx = next_drift_shard_index(out_dir) if resume else 0
    records: list[dict[str, Any]] = []
    written = 0
    extracted = 0

    if show_progress:
        print(
            f"[drift] model={model_name} split={split} samples={len(samples)} "
            f"steps={len(steps)} skipped={skipped} output_dir={out_dir}",
            file=sys.stderr,
            flush=True,
        )

    for step in steps[skipped:]:
        task = activation_fn(model, tokenizer, step.task_prefix, layers, cfg)
        history = activation_fn(model, tokenizer, step.history_prefix, layers, cfg)
        post = activation_fn(model, tokenizer, step.post_tool_prefix, layers, cfg)
        records.append({"task": task, "history": history, "post": post, "label": step.label, "meta": step.meta})
        extracted += 1
        if len(records) >= shard_size:
            write_drift_shard(out_dir, shard_idx, records)
            written += 1
            shard_idx += 1
            records = []

    if records:
        write_drift_shard(out_dir, shard_idx, records)
        written += 1

    return DriftExtractionSummary(
        model_name=model_name,
        split=split,
        output_dir=out_dir,
        total_samples=len(samples),
        total_steps=len(steps),
        skipped_steps=skipped,
        extracted_steps=extracted,
        shards_written=written,
    )
