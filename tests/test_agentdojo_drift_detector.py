from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import secmcp.integrations.agentdojo_drift_detector as runtime_mod
from secmcp.integrations.agentdojo_drift_detector import SecMCPTaskDriftDetector


def test_runtime_rejects_legacy_detector_without_self_baseline_field(tmp_path):
    detector_path = tmp_path / "legacy.pkl"
    legacy = SimpleNamespace(
        threshold=0.5,
        feature_config=SimpleNamespace(feature_mode="concat", include_anchor_distances=True, aggregation="max"),
    )
    detector_path.write_bytes(pickle.dumps(legacy))

    with pytest.raises(ValueError, match="predates per-trajectory self-baseline"):
        SecMCPTaskDriftDetector(detector_path, model_name="fake")


def test_runtime_score_uses_trained_global_baseline(monkeypatch):
    captured = {}

    class Classifier:
        def predict_proba(self, features):
            return np.array([[0.25, 0.75]])

    detector = SecMCPTaskDriftDetector.__new__(SecMCPTaskDriftDetector)
    detector.detector = SimpleNamespace(
        classifier=Classifier(),
        benign_incremental_anchors=None,
        benign_global_anchors=None,
        feature_config=SimpleNamespace(
            feature_mode="concat",
            include_anchor_distances=False,
            include_self_baseline=True,
            aggregation="max",
        ),
        global_baseline={"inc_norm": "sentinel"},
    )
    detector.loaded = SimpleNamespace(model=None, tokenizer=None, cfg=SimpleNamespace(layers=[0]))

    def fake_hidden_states(*args, **kwargs):
        return torch.zeros(1, 2)

    def fake_feature_matrix(split, **kwargs):
        captured["global_baseline"] = kwargs.get("global_baseline")
        return torch.zeros(1, 2)

    monkeypatch.setattr(runtime_mod, "task_anchored_hidden_states", fake_hidden_states)
    monkeypatch.setattr(runtime_mod, "drift_feature_matrix", fake_feature_matrix)

    score, split = detector._score(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "tool", "content": "observation"},
        ]
    )

    assert score == 0.75
    assert split is not None
    assert captured["global_baseline"] == {"inc_norm": "sentinel"}
