from __future__ import annotations

from types import SimpleNamespace

import pytest

from secmcp.activations.drift import extract_split_drift_activations, tool_steps_for_sample
from secmcp.activations.drift_dataset import load_drift_split
from secmcp.data.io import write_samples_jsonl
from secmcp.data.schema import UnifiedSample


torch = pytest.importorskip("torch")


def _sample(label: int = 1) -> UnifiedSample:
    return UnifiedSample(
        label=label,
        source="unit",
        kind="conversation",
        sample_type="case",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "call tool", "tool_calls": []},
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "content": "final", "tool_calls": []},
        ],
    )


def test_tool_steps_prefix_boundaries():
    step = tool_steps_for_sample(_sample(), 3)[0]
    assert step.sample_index == 3
    assert [m["role"] for m in step.task_prefix] == ["system", "user"]
    assert [m["role"] for m in step.history_prefix] == ["system", "user", "assistant"]
    assert [m["role"] for m in step.post_tool_prefix] == ["system", "user", "assistant", "tool"]
    assert step.label == 1


def test_extract_drift_split_writes_records(tmp_path):
    splits_dir = tmp_path / "splits"
    write_samples_jsonl(splits_dir / "train.jsonl", [_sample(0), _sample(1)])

    def activation_fn(model, tokenizer, messages, layers, cfg):
        return torch.ones(len(layers), cfg.hidden_dim) * len(messages)

    summary = extract_split_drift_activations(
        model_name="fake",
        split="train",
        model=object(),
        tokenizer=object(),
        cfg=SimpleNamespace(hidden_dim=3),
        layers=[0, 1],
        splits_dir=splits_dir,
        output_root=tmp_path / "drift",
        shard_size=1,
        activation_fn=activation_fn,
    )
    assert summary.total_steps == 2
    assert summary.shards_written == 2

    split = load_drift_split("fake", "train", tmp_path / "drift")
    assert tuple(split.task.shape) == (2, 2, 3)
    assert split.labels.tolist() == [0, 1]
    assert split.metas[0]["tool_message_index"] == 3
