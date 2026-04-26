from __future__ import annotations

from secmcp.data.schema import UnifiedSample
from secmcp.data.split import deduplicate, make_splits


def _sample(source: str, label: int, idx: int, group: str | None = None) -> UnifiedSample:
    metadata = {}
    if group is not None:
        metadata["split_group"] = group
    return UnifiedSample(
        label=label,
        text=f"{source}-{label}-{idx}",
        source=source,
        kind="unit",
        sample_type="unit",
        metadata=metadata,
    )


def test_deduplicate_removes_exact_duplicate():
    sample = _sample("x", 0, 1)
    assert deduplicate([sample, sample]) == [sample]


def test_make_splits_agentdojo_group_disjoint():
    samples = []
    for i in range(20):
        samples.append(_sample("agentdojo", i % 2, i, group=f"group-{i // 2}"))
    for i in range(20):
        samples.append(_sample("other", i % 2, i))
    splits = make_splits(samples, agentdojo_train_ratio=0.7, train_val_ratio=0.8, seed=1)
    train_val_groups = {
        s.metadata.get("split_group")
        for s in splits["train"] + splits["val"]
        if s.source == "agentdojo"
    }
    test_groups = {s.metadata.get("split_group") for s in splits["test"]}
    assert train_val_groups.isdisjoint(test_groups)
    assert splits["train"]
    assert splits["val"]
    assert splits["test"]


def test_make_splits_train_val_have_both_labels():
    """train and val must contain both benign (0) and malicious (1) samples."""
    samples = []
    for i in range(30):
        samples.append(_sample("agentdojo", i % 2, i, group=f"group-{i // 3}"))
    for i in range(30):
        samples.append(_sample("other", i % 2, i))
    splits = make_splits(samples, agentdojo_train_ratio=0.8, train_val_ratio=0.8, seed=0)
    for split_name in ("train", "val"):
        labels = {s.label for s in splits[split_name]}
        assert labels == {0, 1}, f"{split_name} split missing a label: {labels}"
