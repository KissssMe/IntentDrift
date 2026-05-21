from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections import Counter
from typing import Iterable

from secmcp.data.schema import UnifiedSample


def sample_id(sample: UnifiedSample) -> str:
    raw = "\n".join([sample.source, str(sample.label), sample.text])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def deduplicate(samples: Iterable[UnifiedSample]) -> list[UnifiedSample]:
    seen: set[str] = set()
    result: list[UnifiedSample] = []
    for sample in samples:
        key = sample_id(sample)
        if key in seen:
            continue
        seen.add(key)
        result.append(sample)
    return result


def stratified_split(
    samples: list[UnifiedSample],
    train_ratio: float,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list[UnifiedSample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.source, sample.label)].append(sample)

    left: list[UnifiedSample] = []
    right: list[UnifiedSample] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        if len(bucket) == 1:
            split_idx = 1
        else:
            split_idx = max(1, min(len(bucket) - 1, int(round(len(bucket) * train_ratio))))
        left.extend(bucket[:split_idx])
        right.extend(bucket[split_idx:])
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def stratified_grouped_split(
    samples: list[UnifiedSample],
    train_ratio: float,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    """Like ``stratified_split`` but keeps samples sharing ``metadata.split_group``
    together. Counterfactual (clean, attacked) pairs and any other
    pre-grouped samples therefore always land on the same side.

    Falls back to per-sample atomicity when ``split_group`` is unset. Groups
    are bucketed by their full source/label composition rather than by the
    first member, so mixed-label counterfactual pairs remain atomic without
    being mis-stratified as only the clean side.
    """
    rng = random.Random(seed)
    grouped: dict[str, list[UnifiedSample]] = defaultdict(list)
    for sample in samples:
        group_key = str((sample.metadata or {}).get("split_group") or sample_id(sample))
        grouped[group_key].append(sample)

    buckets: dict[tuple[tuple[str, int, int], ...], list[tuple[str, list[UnifiedSample]]]] = defaultdict(list)
    for group_key, members in grouped.items():
        composition = Counter((member.source, int(member.label)) for member in members)
        bucket_key = tuple(sorted((source, label, count) for (source, label), count in composition.items()))
        buckets[bucket_key].append((group_key, members))

    left: list[UnifiedSample] = []
    right: list[UnifiedSample] = []
    for groups_list in buckets.values():
        rng.shuffle(groups_list)
        n = len(groups_list)
        if n == 1:
            split_idx = 1
        else:
            split_idx = max(1, min(n - 1, int(round(n * train_ratio))))
        for _, members in groups_list[:split_idx]:
            left.extend(members)
        for _, members in groups_list[split_idx:]:
            right.extend(members)
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def split_agentdojo_heldout(
    samples: list[UnifiedSample],
    train_ratio: float,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[UnifiedSample]] = defaultdict(list)
    for sample in samples:
        group = str(sample.metadata.get("split_group") or sample_id(sample))
        grouped[group].append(sample)
    groups = list(grouped)
    rng.shuffle(groups)
    if len(groups) <= 1:
        return samples, []
    split_idx = max(1, min(len(groups) - 1, int(round(len(groups) * train_ratio))))
    train_groups = set(groups[:split_idx])
    train = [s for group in train_groups for s in grouped[group]]
    test = [s for group in groups[split_idx:] for s in grouped[group]]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def split_agentdojo_three_way(
    samples: list[UnifiedSample],
    train_pool_ratio: float,
    train_val_ratio: float,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample], list[UnifiedSample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[UnifiedSample]] = defaultdict(list)
    for sample in samples:
        group = str(sample.metadata.get("split_group") or sample_id(sample))
        grouped[group].append(sample)
    groups = list(grouped)
    rng.shuffle(groups)
    if len(groups) <= 2:
        train, test = split_agentdojo_heldout(samples, train_pool_ratio, seed)
        train, val = stratified_split(train, train_val_ratio, seed)
        return train, val, test

    train_pool_count = max(2, min(len(groups) - 1, int(round(len(groups) * train_pool_ratio))))
    train_count = max(1, min(train_pool_count - 1, int(round(train_pool_count * train_val_ratio))))
    train_groups = set(groups[:train_count])
    val_groups = set(groups[train_count:train_pool_count])
    test_groups = set(groups[train_pool_count:])
    train = [s for group in train_groups for s in grouped[group]]
    val = [s for group in val_groups for s in grouped[group]]
    test = [s for group in test_groups for s in grouped[group]]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def make_splits(
    samples: list[UnifiedSample],
    agentdojo_train_ratio: float = 0.8,
    train_val_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[UnifiedSample]]:
    samples = deduplicate(samples)
    agentdojo_samples = [s for s in samples if s.source == "agentdojo"]
    other_samples = [s for s in samples if s.source != "agentdojo"]
    agentdojo_train, agentdojo_val, test = split_agentdojo_three_way(
        agentdojo_samples,
        train_pool_ratio=agentdojo_train_ratio,
        train_val_ratio=train_val_ratio,
        seed=seed,
    )
    other_train, other_val = stratified_grouped_split(other_samples, train_val_ratio, seed)
    rng = random.Random(seed)
    train = other_train + agentdojo_train
    val = other_val + agentdojo_val
    rng.shuffle(train)
    rng.shuffle(val)
    splits = {"train": train, "val": val, "test": test}
    _warn_on_missing_labels(samples, splits)
    return splits


def _warn_on_missing_labels(
    samples: list[UnifiedSample],
    splits: dict[str, list[UnifiedSample]],
) -> None:
    """Surface debug-mode mishaps where a too-small input or aggressive
    grouping leaves a split with only one label. Threshold calibration and
    AUROC computation both need both labels present, but failures otherwise
    show up far downstream as cryptic sklearn errors. We warn rather than
    raise so legitimate single-label corner cases still run.
    """
    import sys

    input_labels = {s.label for s in samples}
    if len(input_labels) < 2:
        return
    for name, split_samples in splits.items():
        if not split_samples:
            continue
        present = {s.label for s in split_samples}
        missing = input_labels - present
        if missing:
            print(
                f"[make_splits] WARNING: split={name!r} is missing label(s) {sorted(missing)} "
                f"(input had {sorted(input_labels)}). Threshold calibration / AUROC will fail for this split. "
                "Consider raising --max-per-source, lowering train_val_ratio, or disabling --counterfactual-pairs.",
                file=sys.stderr,
                flush=True,
            )
