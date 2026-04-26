from __future__ import annotations

import hashlib
import random
from collections import defaultdict
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


def make_splits(
    samples: list[UnifiedSample],
    agentdojo_train_ratio: float = 0.8,
    train_val_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[UnifiedSample]]:
    samples = deduplicate(samples)
    agentdojo_samples = [s for s in samples if s.source == "agentdojo"]
    other_samples = [s for s in samples if s.source != "agentdojo"]
    agentdojo_train_pool, test = split_agentdojo_heldout(agentdojo_samples, agentdojo_train_ratio, seed)
    train_pool = other_samples + agentdojo_train_pool
    train, val = stratified_split(train_pool, train_val_ratio, seed)
    return {"train": train, "val": val, "test": test}
