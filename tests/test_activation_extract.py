from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from secmcp.activations.extract import (
    clear_activation_shards,
    count_completed_samples,
    extract_split_activations,
    shard_paths,
    write_shard,
)
from secmcp.data.io import write_samples_jsonl
from secmcp.data.schema import UnifiedSample


torch = pytest.importorskip("torch")


def _samples(n: int) -> list[UnifiedSample]:
    return [
        UnifiedSample(
            label=i % 2,
            text=f"sample text {i}",
            source="unit",
            kind="unit",
            sample_type="unit",
            metadata={"id": i},
        )
        for i in range(n)
    ]


def _activation_fn(model, tokenizer, text: str, layers: list[int], cfg):
    base = float(len(text))
    return torch.ones(len(layers), cfg.hidden_dim) * base


def test_write_shard_roundtrip(tmp_path):
    out_dir = tmp_path / "acts"
    activations = [torch.ones(2, 3), torch.zeros(2, 3)]
    labels = [1, 0]
    metas = [{"sample_index": 0}, {"sample_index": 1}]
    write_shard(out_dir, 0, activations, labels, metas)
    act_path, labels_path, meta_path = shard_paths(out_dir, 0)
    loaded = torch.load(act_path, map_location="cpu", weights_only=True)
    loaded_labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    loaded_meta = [json.loads(line) for line in meta_path.read_text().splitlines()]
    assert tuple(loaded.shape) == (2, 2, 3)
    assert loaded_labels.tolist() == labels
    assert loaded_meta == metas


def test_extract_split_writes_shards(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", _samples(5))
    cfg = SimpleNamespace(hidden_dim=4)
    summary = extract_split_activations(
        model_name="fake_model",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=cfg,
        layers=[0, 2],
        splits_dir=splits_dir,
        output_root=tmp_path / "activations",
        shard_size=2,
        activation_fn=_activation_fn,
    )
    assert summary.total_samples == 5
    assert summary.extracted_samples == 5
    assert summary.shards_written == 3
    assert count_completed_samples(summary.output_dir) == 5

    act0, labels0, meta0 = shard_paths(summary.output_dir, 0)
    tensor0 = torch.load(act0, map_location="cpu", weights_only=True)
    assert tuple(tensor0.shape) == (2, 2, 4)
    assert torch.load(labels0, map_location="cpu", weights_only=True).tolist() == [0, 1]
    first_meta = json.loads(meta0.read_text().splitlines()[0])
    assert first_meta["source"] == "unit"
    assert first_meta["sample_index"] == 0


def test_extract_split_resume_skips_completed_samples(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", _samples(5))
    calls: list[str] = []

    def activation_fn(model, tokenizer, text: str, layers: list[int], cfg):
        calls.append(text)
        return torch.ones(len(layers), cfg.hidden_dim)

    common = dict(
        model_name="fake_model",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=SimpleNamespace(hidden_dim=2),
        layers=[0],
        splits_dir=splits_dir,
        output_root=tmp_path / "activations",
        shard_size=2,
        activation_fn=activation_fn,
    )
    first = extract_split_activations(**common)
    assert first.extracted_samples == 5
    assert len(calls) == 5
    calls.clear()
    second = extract_split_activations(**common)
    assert second.skipped_samples == 5
    assert second.extracted_samples == 0
    assert calls == []


def test_extract_split_partial_resume_continues_at_next_shard(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", _samples(5))
    out_dir = tmp_path / "activations" / "fake_model" / "train"
    write_shard(out_dir, 0, [torch.ones(1, 2), torch.ones(1, 2)], [0, 1], [{"sample_index": 0}, {"sample_index": 1}])
    cfg = SimpleNamespace(hidden_dim=2)
    summary = extract_split_activations(
        model_name="fake_model",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=cfg,
        layers=[0],
        splits_dir=splits_dir,
        output_root=tmp_path / "activations",
        shard_size=2,
        activation_fn=_activation_fn,
    )
    assert summary.skipped_samples == 2
    assert summary.extracted_samples == 3
    assert (out_dir / "shard_00001.pt").exists()
    assert (out_dir / "shard_00002.pt").exists()


def test_no_resume_clears_stale_shards(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", _samples(1))
    out_dir = tmp_path / "activations" / "fake_model" / "train"
    write_shard(out_dir, 0, [torch.ones(1, 2)], [0], [{"sample_index": 0}])
    write_shard(out_dir, 9, [torch.ones(1, 2)], [1], [{"sample_index": 9}])
    summary = extract_split_activations(
        model_name="fake_model",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=SimpleNamespace(hidden_dim=2),
        layers=[0],
        splits_dir=splits_dir,
        output_root=tmp_path / "activations",
        shard_size=2,
        resume=False,
        activation_fn=_activation_fn,
    )
    assert summary.skipped_samples == 0
    assert (out_dir / "shard_00000.pt").exists()
    assert not (out_dir / "shard_00009.pt").exists()
    assert count_completed_samples(out_dir) == 1


def test_clear_activation_shards_only_removes_activation_files(tmp_path):
    out_dir = tmp_path / "acts"
    write_shard(out_dir, 0, [torch.ones(1, 2)], [0], [{"sample_index": 0}])
    keep = out_dir / "notes.txt"
    keep.write_text("keep", encoding="utf-8")
    clear_activation_shards(out_dir)
    assert keep.exists()
    assert not (out_dir / "shard_00000.pt").exists()


def test_extract_split_progress_enabled(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", _samples(2))
    summary = extract_split_activations(
        model_name="fake_model",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=SimpleNamespace(hidden_dim=2),
        layers=[0],
        splits_dir=splits_dir,
        output_root=tmp_path / "activations",
        shard_size=10,
        show_progress=True,
        activation_fn=_activation_fn,
    )
    assert summary.extracted_samples == 2
