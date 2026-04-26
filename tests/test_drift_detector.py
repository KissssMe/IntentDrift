from __future__ import annotations

import pytest

from secmcp.activations.drift_dataset import DriftActivationSplit
from secmcp.detectors.drift import (
    aggregate_split_scores,
    aggregate_step_scores,
    drift_feature_matrix,
    train_hist_gradient_drift,
)


torch = pytest.importorskip("torch")


def _split(n: int, offset: float = 2.5) -> DriftActivationSplit:
    n0 = n // 2
    n1 = n - n0
    task = torch.zeros(n, 2, 4)
    history = torch.randn(n, 2, 4) * 0.05
    benign_post = history[:n0] + torch.randn(n0, 2, 4) * 0.05
    malicious_post = history[n0:] + offset + torch.randn(n1, 2, 4) * 0.05
    post = torch.cat([benign_post, malicious_post], dim=0)
    labels = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    return DriftActivationSplit(task=task, history=history, post=post, labels=labels, metas=[{} for _ in range(n)])


def test_drift_feature_matrix_shape():
    split = _split(10)
    anchors = (split.post - split.history)[:3]
    features = drift_feature_matrix(
        split,
        benign_incremental_anchors=anchors,
        benign_global_anchors=anchors,
        feature_mode="concat",
    )
    assert features.shape[0] == 10
    assert features.shape[1] > 8


def test_drift_feature_matrix_uses_per_layer_stats_and_optional_anchors():
    split = _split(10)
    features = drift_feature_matrix(
        split,
        feature_mode="concat",
        include_anchor_distances=False,
    )
    # inc_flat + glob_flat + five per-layer stat groups:
    # 8 + 8 + (5 * 2) for [N, 2, 4] activations.
    assert features.shape == (10, 26)

    last_only = drift_feature_matrix(
        split,
        feature_mode="last_only",
        include_anchor_distances=False,
    )
    assert last_only.shape == (10, 13)


def test_aggregate_step_scores():
    assert aggregate_step_scores([], aggregation="max") == 0.0
    assert aggregate_step_scores([0.1, 0.8, 0.4], aggregation="max") == 0.8
    assert aggregate_step_scores([0.1, 0.8, 0.4], aggregation="top2_mean") == pytest.approx(0.6)


def test_aggregate_split_scores_groups_by_sample_index():
    split = _split(4)
    split.metas[0]["sample_index"] = 10
    split.metas[1]["sample_index"] = 10
    split.metas[2]["sample_index"] = 20
    split.metas[3]["sample_index"] = 20
    labels, scores = aggregate_split_scores(split, [0.1, 0.7, 0.4, 0.8], aggregation="top2_mean")
    assert labels.tolist() == [0, 1]
    assert scores.tolist() == pytest.approx([0.4, 0.6])


def test_train_hist_gradient_drift_synthetic_high_auc():
    detector = train_hist_gradient_drift(
        model_name="fake",
        train=_split(80),
        val=_split(40),
        test=_split(40),
        n_anchors=10,
        params={"max_iter": 50, "random_state": 0},
    )
    assert detector.val_metrics["auroc"] > 0.95
    assert detector.test_metrics is not None
    assert detector.test_metrics["auroc"] > 0.95


def test_train_hist_gradient_drift_can_disable_anchor_distances():
    detector = train_hist_gradient_drift(
        model_name="fake",
        train=_split(80),
        val=_split(40),
        n_anchors=10,
        include_anchor_distances=False,
        aggregation="top2_mean",
        params={"max_iter": 50, "random_state": 0},
    )
    assert detector.benign_incremental_anchors is None
    assert detector.benign_global_anchors is None
    assert detector.feature_config.include_anchor_distances is False
    assert detector.feature_config.aggregation == "top2_mean"
