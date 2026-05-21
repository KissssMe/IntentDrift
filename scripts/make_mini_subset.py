#!/usr/bin/env python
"""Carve a smaller, distribution-matched experiment set out of existing splits.

Reuses already-extracted drift activation shards: no model forward passes are
required. The mini set lives at ``outputs/splits_mini/`` and
``outputs/drift_activations_mini/{model}/{split}/`` so the full-scale artifacts
stay intact.

Stratification is by ``(source, label)`` at the *group* level: counterfactual
(clean, attacked) pairs and AgentDojo ``split_group`` clusters always move
together. Step rows in mini shards have ``sample_index`` renumbered to match
the new mini jsonl line indices, and ``original_sample_index`` is recorded for
traceability so we can reconcile against the full-set artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from secmcp.activations.drift import completed_drift_shards, drift_output_dir, drift_shard_paths
from secmcp.config import OUTPUTS_DIR


def _group_key(sample: dict, line_index: int) -> str:
    """Return an atomicity key for stratified subsampling.

    Counterfactual ``(clean, attacked)`` pairs are paired training data; their
    value comes from being scored together, so we must keep them on the same
    side of the mini cut. Everything else — including AgentDojo's
    ``split_group`` — exists only to prevent *cross-split* leakage between
    train/val/test, which is already enforced by the upstream splitter; within
    a single split we are free to subsample members of an AgentDojo group
    independently, which is necessary for the mini fraction to match the
    target (those groups contain up to ~178 samples each).
    """
    metadata = sample.get("metadata") or {}
    pair_id = metadata.get("pair_id")
    if pair_id:
        return f"pair::{pair_id}"
    grp = metadata.get("split_group")
    if grp and str(grp).startswith("cf:"):
        return f"pair::{grp}"
    return f"single::{line_index}"


def _group_stratum(members: list[dict]) -> tuple:
    composition = Counter((str(m.get("source", "?")), int(m.get("label", -1))) for m in members)
    return tuple(sorted((source, label, count) for (source, label), count in composition.items()))


def stratified_group_subsample(
    samples: list[dict],
    pos_fraction: float,
    neg_fraction: float,
    seed: int,
) -> list[int]:
    """Return sorted list of original line indices to keep.

    Per-label fractions: typically ``pos_fraction << neg_fraction`` (e.g.
    20% positives, 100% negatives) so the mini set has enough benign
    trajectories to calibrate thresholds — the previous uniform 20% sampler
    left val with only 71 benign samples, which is too sparse for FPR
    control. Groups remain atomic: counterfactual ``(clean, attacked)`` pairs
    are kept together by routing them through the ``pos_fraction`` budget
    (any member is a positive).
    """
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        groups[_group_key(sample, idx)].append((idx, sample))

    by_stratum: dict[tuple, list[tuple[str, list[tuple[int, dict]]]]] = defaultdict(list)
    for gid, members in groups.items():
        stratum = _group_stratum([m for _, m in members])
        by_stratum[stratum].append((gid, members))

    def group_target_weight(members: list[tuple[int, dict]]) -> float:
        """Each member contributes its label-specific fraction; the group's
        total weight is the expected number of *sampled* rows if it were
        kept. Greedy add until accumulated ≥ summed weight."""
        return sum(
            (pos_fraction if int(m.get("label", 0)) == 1 else neg_fraction)
            for _, m in members
        )

    rng = random.Random(seed)
    selected_indices: set[int] = set()
    for stratum, group_list in by_stratum.items():
        group_list = sorted(group_list, key=lambda kv: kv[0])
        rng.shuffle(group_list)
        target_samples = sum(group_target_weight(members) for _, members in group_list)
        # Round up so we never starve a small stratum into zero coverage.
        target_samples = max(1, round(target_samples))
        accumulated = 0
        for _, members in group_list:
            if accumulated >= target_samples:
                break
            for line_idx, _ in members:
                selected_indices.add(line_idx)
            accumulated += len(members)
    return sorted(selected_indices)


def read_split_jsonl(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_mini_split(out_path: Path, selected_samples: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _clear_mini_shards(dst_dir: Path) -> None:
    for pattern in ("drift_*.pt", "labels_*.pt", "meta_*.jsonl"):
        for path in dst_dir.glob(pattern):
            path.unlink()


def subsample_activation_shards(
    *,
    model_name: str,
    split: str,
    selected_original_indices: list[int],
    src_root: Path,
    dst_root: Path,
    shard_size: int,
) -> tuple[int, int, int]:
    import torch

    src_dir = drift_output_dir(model_name, split, src_root)
    dst_dir = drift_output_dir(model_name, split, dst_root)
    dst_dir.mkdir(parents=True, exist_ok=True)
    _clear_mini_shards(dst_dir)

    remap = {orig: new for new, orig in enumerate(selected_original_indices)}
    selected_set = set(selected_original_indices)

    buf_task: list[Any] = []
    buf_history: list[Any] = []
    buf_post: list[Any] = []
    buf_labels: list[int] = []
    buf_meta: list[dict] = []
    out_shard_idx = 0
    total_in = 0
    total_out = 0

    def flush() -> None:
        nonlocal out_shard_idx, buf_task, buf_history, buf_post, buf_labels, buf_meta
        if not buf_meta:
            return
        drift_path, labels_path, meta_path = drift_shard_paths(dst_dir, out_shard_idx)
        torch.save(
            {
                "task": torch.stack(buf_task),
                "history": torch.stack(buf_history),
                "post": torch.stack(buf_post),
            },
            drift_path,
        )
        torch.save(torch.tensor(buf_labels, dtype=torch.long), labels_path)
        with meta_path.open("w", encoding="utf-8") as f:
            for record in buf_meta:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_shard_idx += 1
        buf_task = []
        buf_history = []
        buf_post = []
        buf_labels = []
        buf_meta = []

    for src_shard_idx in completed_drift_shards(src_dir):
        drift_path, labels_path, meta_path = drift_shard_paths(src_dir, src_shard_idx)
        payload = torch.load(drift_path, map_location="cpu", weights_only=True)
        labels = torch.load(labels_path, map_location="cpu", weights_only=True)
        with meta_path.open(encoding="utf-8") as f:
            metas = [json.loads(line) for line in f if line.strip()]
        n = int(labels.shape[0])
        total_in += n
        for row in range(n):
            meta = metas[row]
            orig_idx = int(meta.get("sample_index", -1))
            if orig_idx not in selected_set:
                continue
            new_meta = dict(meta)
            new_meta["sample_index"] = remap[orig_idx]
            new_meta["original_sample_index"] = orig_idx
            buf_task.append(payload["task"][row])
            buf_history.append(payload["history"][row])
            buf_post.append(payload["post"][row])
            buf_labels.append(int(labels[row].item()))
            buf_meta.append(new_meta)
            total_out += 1
            if len(buf_meta) >= shard_size:
                flush()
    flush()
    return total_in, total_out, out_shard_idx


def distribution_summary(samples: list[dict]) -> dict[tuple[str, int], int]:
    counts: Counter = Counter()
    for sample in samples:
        counts[(str(sample.get("source", "?")), int(sample.get("label", -1)))] += 1
    return dict(counts)


def fmt_distribution(before: dict, after: dict) -> str:
    keys = sorted(set(before) | set(after))
    lines = []
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        ratio = (a / b) if b else 0.0
        lines.append(f"    {key[0]:<22s} label={key[1]} before={b:>6d} after={a:>5d} ({ratio:.1%})")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.20,
        help="Legacy uniform fraction. Overridden by --pos-fraction/--neg-fraction when provided.",
    )
    parser.add_argument(
        "--pos-fraction",
        type=float,
        default=0.20,
        help="Sampling fraction for positive (label==1) groups.",
    )
    parser.add_argument(
        "--neg-fraction",
        type=float,
        default=1.00,
        help="Sampling fraction for negative (label==0) groups. Default keeps all benign "
        "so threshold calibration has enough samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--src-splits-dir", type=Path, default=OUTPUTS_DIR / "splits")
    parser.add_argument("--dst-splits-dir", type=Path, default=OUTPUTS_DIR / "splits_mini")
    parser.add_argument("--src-activation-root", type=Path, default=OUTPUTS_DIR / "drift_activations")
    parser.add_argument("--dst-activation-root", type=Path, default=OUTPUTS_DIR / "drift_activations_mini")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys whose drift_activations/{model}/{split} should be subset. "
        "Defaults to every model directory found under --src-activation-root.",
    )
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument(
        "--no-activations",
        action="store_true",
        help="Only write mini splits jsonl; skip activation shard subsetting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.models is None:
        if args.src_activation_root.exists():
            args.models = sorted(p.name for p in args.src_activation_root.iterdir() if p.is_dir())
        else:
            args.models = []

    print(
        f"[mini] pos_fraction={args.pos_fraction:.2f} neg_fraction={args.neg_fraction:.2f} "
        f"seed={args.seed} splits={args.splits} models={args.models or '(none)'}",
        file=sys.stderr,
        flush=True,
    )

    for split in args.splits:
        src_path = args.src_splits_dir / f"{split}.jsonl"
        if not src_path.exists():
            print(f"[mini] skip split={split}: {src_path} not found", file=sys.stderr)
            continue

        all_samples = read_split_jsonl(src_path)
        # Per-split seed perturbation so train / val / test use independent
        # group orderings under one user-supplied --seed.
        split_seed = args.seed + (hash(split) & 0xFFFF)
        selected_indices = stratified_group_subsample(
            all_samples,
            pos_fraction=args.pos_fraction,
            neg_fraction=args.neg_fraction,
            seed=split_seed,
        )
        selected_samples = [all_samples[i] for i in selected_indices]
        dst_path = args.dst_splits_dir / f"{split}.jsonl"
        write_mini_split(dst_path, selected_samples)

        before = distribution_summary(all_samples)
        after = distribution_summary(selected_samples)
        print(
            f"[mini] split={split}: {len(selected_samples)}/{len(all_samples)} samples "
            f"({len(selected_samples) / max(1, len(all_samples)):.1%}) → {dst_path}",
            file=sys.stderr,
        )
        print(fmt_distribution(before, after), file=sys.stderr)

        if args.no_activations:
            continue
        for model in args.models:
            src_dir = drift_output_dir(model, split, args.src_activation_root)
            if not src_dir.exists():
                print(f"[mini] skip model={model} split={split}: {src_dir} missing", file=sys.stderr)
                continue
            n_in, n_out, n_shards = subsample_activation_shards(
                model_name=model,
                split=split,
                selected_original_indices=selected_indices,
                src_root=args.src_activation_root,
                dst_root=args.dst_activation_root,
                shard_size=args.shard_size,
            )
            ratio = (n_out / n_in) if n_in else 0.0
            print(
                f"[mini] acts model={model} split={split}: {n_out}/{n_in} steps "
                f"({ratio:.1%}) shards_written={n_shards}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
