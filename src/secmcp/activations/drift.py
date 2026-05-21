from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from secmcp.config import OUTPUTS_DIR
from secmcp.data.io import read_samples_jsonl
from secmcp.data.schema import UnifiedSample, normalize_messages_for_chat
from secmcp.models.hooks import TASK_ANCHOR_EXTRACTION_MODE, task_anchored_hidden_states


@dataclass(frozen=True)
class ToolStep:
    sample_index: int
    step_index: int
    task_prefix: list[dict[str, Any]]
    history_prefix: list[dict[str, Any]]
    post_tool_prefix: list[dict[str, Any]]
    task_text: str
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


def validate_completed_drift_shards(
    output_dir: Path,
    *,
    expected_extraction_mode: str = TASK_ANCHOR_EXTRACTION_MODE,
) -> None:
    """Reject resumable shards extracted with an incompatible representation.

    The task-drift activation format was changed from last-token prefixes to
    task-anchored mean pooling. A shape-compatible old shard would otherwise
    be silently mixed into a new run when ``resume=True``.
    """
    for idx in completed_drift_shards(output_dir):
        _, _, meta_path = drift_shard_paths(output_dir, idx)
        with meta_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                mode = meta.get("extraction_mode")
                if mode != expected_extraction_mode:
                    raise ValueError(
                        f"Drift shard {meta_path} line {line_no} has extraction_mode={mode!r}; "
                        f"expected {expected_extraction_mode!r}. Re-run extraction with --no-resume "
                        "or use a fresh --output-root."
                    )


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
    """System-only prefix. The user task is re-appended via the task-anchored
    hook's recap; including it both inside the prefix and at the appended
    recap (the previous behavior) made ``task_anchor`` "the task looking at
    itself" — effectively a constant baseline that carried no information
    about the trajectory.
    """
    return [msg for msg in messages if msg.get("role") == "system"]


def _task_text(messages: list[dict[str, Any]]) -> str:
    """First user message content; this is the task description we re-append
    at each anchor to probe how the model's understanding of it shifts."""
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    for msg in messages:
        if msg.get("role") == "system":
            return str(msg.get("content") or "")
    return ""


def _tool_name(msg: dict[str, Any]) -> str | None:
    tool_call = msg.get("tool_call") or {}
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            return function.get("name")
        if isinstance(function, str):
            return function
    return msg.get("name")


def _matched_injection_indices(messages: list[dict[str, Any]], sample: UnifiedSample) -> set[int] | None:
    metadata = sample.metadata or {}
    explicit = metadata.get("malicious_tool_message_indices")
    if explicit is not None:
        return {int(idx) for idx in explicit}

    fragments = metadata.get("injection_fragments") or []
    injection_text = str(metadata.get("injection_text") or "").strip()
    if not fragments and injection_text:
        fragments = [part.strip() for part in injection_text.splitlines() if len(part.strip()) >= 40]
        if not fragments:
            fragments = [injection_text]
    fragments = [str(fragment).strip() for fragment in fragments if str(fragment).strip()]
    if not fragments:
        return None

    matched: set[int] = set()
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        if any(fragment in content for fragment in fragments):
            matched.add(idx)
    return matched or None


def tool_steps_for_sample(sample: UnifiedSample, sample_index: int) -> list[ToolStep]:
    messages = normalize_messages_for_chat(sample.messages)
    task_prefix = _task_prefix(messages)
    task_text = _task_text(messages)
    malicious_tool_indices = _matched_injection_indices(messages, sample) if sample.label == 1 else None
    # Forward-propagate the positive label: once an injected tool message enters
    # the agent's context, every subsequent tool step is operating on a hijacked
    # state and must also count as positive. Without this, the same drift pattern
    # is taught to the classifier as both 0 and 1.
    first_injection_index = min(malicious_tool_indices) if malicious_tool_indices else None
    steps: list[ToolStep] = []
    step_index = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        if sample.label == 1 and first_injection_index is not None:
            step_label = 1 if idx >= first_injection_index else 0
            step_label_source = "matched_injection_tool"
        elif sample.label == 1:
            # Positive trajectory but no locator: marking every step as 1 yields
            # contradictory supervision (pre-injection steps look benign yet
            # carry label 1). Emit ``-1`` so the trainer can drop these steps
            # from the loss while validation still aggregates them by
            # trajectory_label.
            step_label = -1
            step_label_source = "trajectory_label_unlocatable"
        else:
            step_label = sample.label
            step_label_source = "trajectory_label"
        meta = {
            "sample_index": sample_index,
            "step_index": step_index,
            "tool_message_index": idx,
            "label": step_label,
            "trajectory_label": sample.label,
            "step_label_source": step_label_source,
            "source": sample.source,
            "kind": sample.kind,
            "sample_type": sample.sample_type,
            "tool_name": _tool_name(msg),
            "metadata": sample.metadata,
            "extraction_mode": TASK_ANCHOR_EXTRACTION_MODE,
        }
        steps.append(
            ToolStep(
                sample_index=sample_index,
                step_index=step_index,
                task_prefix=task_prefix,
                history_prefix=messages[:idx],
                post_tool_prefix=messages[: idx + 1],
                task_text=task_text,
                label=step_label,
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
    log_every: int = 1,
    activation_fn: Callable[..., Any] | None = None,
) -> DriftExtractionSummary:
    if activation_fn is None:
        activation_fn = task_anchored_hidden_states
    root = splits_dir or OUTPUTS_DIR / "splits"
    samples = read_samples_jsonl(root / f"{split}.jsonl")
    steps = list(iter_tool_steps(samples))
    out_dir = drift_output_dir(model_name, split, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        clear_drift_shards(out_dir)
    if resume:
        validate_completed_drift_shards(out_dir)
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

    pending_steps = steps[skipped:]
    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(
            total=len(steps),
            initial=skipped,
            desc=f"{model_name}/{split}/task_drift",
            unit="step",
            dynamic_ncols=True,
            file=sys.stderr,
            disable=False,
            mininterval=1.0,
        )

    try:
        for step in pending_steps:
            if show_progress and log_every > 0 and (extracted == 0 or extracted % log_every == 0):
                print(
                    f"[drift] starting sample_index={step.sample_index} "
                    f"step_index={step.step_index} "
                    f"tool_message_index={step.meta.get('tool_message_index')} "
                    f"done={skipped + extracted}/{len(steps)} label={step.label}",
                    file=sys.stderr,
                    flush=True,
                )
            task = activation_fn(model, tokenizer, step.task_prefix, step.task_text, layers, cfg)
            history = activation_fn(model, tokenizer, step.history_prefix, step.task_text, layers, cfg)
            post = activation_fn(model, tokenizer, step.post_tool_prefix, step.task_text, layers, cfg)
            records.append({"task": task, "history": history, "post": post, "label": step.label, "meta": step.meta})
            extracted += 1
            if progress is not None:
                progress.update(1)
            if len(records) >= shard_size:
                write_drift_shard(out_dir, shard_idx, records)
                written += 1
                shard_idx += 1
                records = []
                if progress is not None:
                    progress.set_postfix(shards=shard_idx)
    finally:
        if progress is not None:
            progress.close()

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
