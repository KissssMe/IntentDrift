from __future__ import annotations

import json

import pytest

from secmcp.activations.dataset import load_activation_split
from secmcp.activations.extract import shard_paths, write_shard


torch = pytest.importorskip("torch")


def test_load_activation_split_roundtrip(tmp_path):
    root = tmp_path / "activations"
    out_dir = root / "fake" / "train"
    write_shard(
        out_dir,
        0,
        [torch.ones(2, 3), torch.zeros(2, 3)],
        [0, 1],
        [{"sample_index": 0}, {"sample_index": 1}],
    )
    split = load_activation_split("fake", "train", root)
    assert tuple(split.activations.shape) == (2, 2, 3)
    assert split.labels.tolist() == [0, 1]
    assert split.metas == [{"sample_index": 0}, {"sample_index": 1}]


def test_load_activation_split_detects_count_mismatch(tmp_path):
    root = tmp_path / "activations"
    out_dir = root / "fake" / "train"
    write_shard(out_dir, 0, [torch.ones(2, 3)], [0], [{"sample_index": 0}])
    _, _, meta_path = shard_paths(out_dir, 0)
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_index": 1}) + "\n")
    with pytest.raises(ValueError, match="count mismatch"):
        load_activation_split("fake", "train", root)
