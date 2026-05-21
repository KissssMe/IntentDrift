from __future__ import annotations

import pytest

from secmcp.activations.drift_dataset import DriftActivationSplit
from secmcp.detectors.drift import (
    TrajectoryPriorNorms,
    aggregate_split_records,
    aggregate_split_scores,
    aggregate_step_scores,
    confusion_at_threshold,
    drift_feature_matrix,
    group_diagnostics,
    threshold_tradeoffs,
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
        include_self_baseline=False,
    )
    # inc_flat + glob_flat + five per-layer stat groups:
    # 8 + 8 + (5 * 2) for [N, 2, 4] activations.
    assert features.shape == (10, 26)

    last_only = drift_feature_matrix(
        split,
        feature_mode="last_only",
        include_anchor_distances=False,
        include_self_baseline=False,
    )
    assert last_only.shape == (10, 13)


def test_drift_feature_matrix_includes_self_baseline_columns():
    """With include_self_baseline=True we add 4 z-scored columns per layer
    (incremental-norm, global-norm, history-norm, relative-norm)."""
    split = _split(10)
    base = drift_feature_matrix(
        split,
        feature_mode="concat",
        include_anchor_distances=False,
        include_self_baseline=False,
    )
    with_self = drift_feature_matrix(
        split,
        feature_mode="concat",
        include_anchor_distances=False,
        include_self_baseline=True,
    )
    assert with_self.shape == (base.shape[0], base.shape[1] + 4 * 2)


def test_self_baseline_first_two_steps_per_trajectory_are_zero():
    """The first two steps of every trajectory must contribute zero
    self-baseline columns: with 0 or 1 prior data points, std is 0 and a
    real z-score would blow up to ~1e6 / eps. Real z-scores only kick in
    from step 3 onwards (>=2 prior steps)."""
    n = 9
    task = torch.zeros(n, 2, 4)
    history = torch.randn(n, 2, 4) * 0.05
    post = history + torch.randn(n, 2, 4) * 0.5
    metas = []
    for traj in range(3):
        for step in range(3):
            metas.append({"sample_index": traj, "step_index": step})
    split = DriftActivationSplit(
        task=task, history=history, post=post, labels=torch.zeros(n).long(), metas=metas,
    )
    base_cols = drift_feature_matrix(
        split, feature_mode="concat", include_anchor_distances=False, include_self_baseline=False
    ).shape[1]
    with_self = drift_feature_matrix(
        split, feature_mode="concat", include_anchor_distances=False, include_self_baseline=True
    )
    self_block = with_self[:, base_cols:]
    # Steps 0 and 1 of every trajectory must be exactly zero.
    for row, meta in enumerate(metas):
        if meta["step_index"] < 2:
            assert torch.allclose(self_block[row], torch.zeros_like(self_block[row])), (
                f"row {row} (step_index={meta['step_index']}) should have zero self-baseline"
            )
    # At least one step-2 row should be non-zero (drift was randomized).
    step2_rows = [row for row, meta in enumerate(metas) if meta["step_index"] == 2]
    assert any(not torch.allclose(self_block[row], torch.zeros_like(self_block[row])) for row in step2_rows)
    # And bounded: no single value should be in the runaway 1e3+ regime that
    # the previous implementation produced when prior length was 1.
    assert torch.isfinite(self_block).all()
    assert self_block.abs().max() < 1e3


def test_drift_feature_matrix_rejects_prior_state_on_batch():
    """``prior_state`` is the runtime single-step contract; passing it on a
    multi-row split would silently apply one external prior to every row."""
    split = _split(4)
    state = TrajectoryPriorNorms()
    with pytest.raises(ValueError, match="single-step path"):
        drift_feature_matrix(
            split,
            feature_mode="concat",
            include_anchor_distances=False,
            include_self_baseline=True,
            prior_state=state,
        )


def test_self_baseline_first_step_per_trajectory_is_zero():
    """The very first tool step of a trajectory has no within-trajectory
    prior, so its z-scored block must be all zeros."""
    split = _split(4)
    split.metas[0]["sample_index"] = 1
    split.metas[0]["step_index"] = 0
    split.metas[1]["sample_index"] = 1
    split.metas[1]["step_index"] = 1
    split.metas[2]["sample_index"] = 2
    split.metas[2]["step_index"] = 0
    split.metas[3]["sample_index"] = 2
    split.metas[3]["step_index"] = 1
    base_cols = drift_feature_matrix(
        split,
        feature_mode="concat",
        include_anchor_distances=False,
        include_self_baseline=False,
    ).shape[1]
    with_self = drift_feature_matrix(
        split,
        feature_mode="concat",
        include_anchor_distances=False,
        include_self_baseline=True,
    )
    self_block_first_traj_first_step = with_self[0, base_cols:]
    self_block_second_traj_first_step = with_self[2, base_cols:]
    assert torch.allclose(self_block_first_traj_first_step, torch.zeros_like(self_block_first_traj_first_step))
    assert torch.allclose(self_block_second_traj_first_step, torch.zeros_like(self_block_second_traj_first_step))


def test_aggregate_step_scores():
    assert aggregate_step_scores([], aggregation="max") == 0.0
    assert aggregate_step_scores([0.1, 0.8, 0.4], aggregation="max") == 0.8
    assert aggregate_step_scores([0.1, 0.8, 0.4], aggregation="top2_mean") == pytest.approx(0.6)


def test_aggregate_step_scores_first_exceed_k_bounds_detection_budget():
    """``first_exceed_K`` looks only at the first K tool steps. A benign
    spike in step 5 cannot trigger when K=3, but the same spike at step 2
    does. This is the structural fix for ``max``'s long-trajectory FPR
    amplification."""
    early_attack = [0.1, 0.9, 0.2, 0.1, 0.1]
    late_spike = [0.2, 0.2, 0.3, 0.1, 0.95]
    assert aggregate_step_scores(early_attack, aggregation="first_exceed_3") == pytest.approx(0.9)
    assert aggregate_step_scores(late_spike, aggregation="first_exceed_3") == pytest.approx(0.3)
    # max would have flagged both.
    assert aggregate_step_scores(late_spike, aggregation="max") == pytest.approx(0.95)


def test_aggregate_step_scores_cusum_rewards_sustained_elevation():
    """CUSUM accumulates above the reference; an isolated spike decays back,
    but several elevated steps in a row produce a large s_max."""
    isolated_spike = [0.2, 0.3, 0.85, 0.2, 0.3]
    sustained = [0.55, 0.6, 0.65, 0.7, 0.75]
    score_spike = aggregate_step_scores(isolated_spike, aggregation="cusum")
    score_sustained = aggregate_step_scores(sustained, aggregation="cusum")
    # Both > 0; sustained should produce a larger CUSUM than a single spike.
    assert score_sustained > score_spike
    assert score_spike < 0.5  # reference=0.5; spike alone barely registers


def test_aggregate_step_scores_cusum_window_truncates():
    scores = [0.7, 0.7, 0.2, 0.2, 0.2]
    full = aggregate_step_scores(scores, aggregation="cusum")
    windowed = aggregate_step_scores(scores, aggregation="cusum_w2")
    assert full == pytest.approx(windowed)


def test_aggregate_step_scores_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        aggregate_step_scores([0.5], aggregation="not_a_strategy")


def test_aggregate_split_scores_groups_by_sample_index():
    split = _split(4)
    split.metas[0]["sample_index"] = 10
    split.metas[1]["sample_index"] = 10
    split.metas[2]["sample_index"] = 20
    split.metas[3]["sample_index"] = 20
    labels, scores = aggregate_split_scores(split, [0.1, 0.7, 0.4, 0.8], aggregation="top2_mean")
    assert labels.tolist() == [0, 1]
    assert scores.tolist() == pytest.approx([0.4, 0.6])


def test_aggregate_split_scores_uses_trajectory_label_when_step_labels_differ():
    base = _split(3)
    split = DriftActivationSplit(
        task=base.task,
        history=base.history,
        post=base.post,
        labels=torch.tensor([0, 1, 0]),
        metas=[
            {"sample_index": 10, "trajectory_label": 1},
            {"sample_index": 10, "trajectory_label": 1},
            {"sample_index": 20, "trajectory_label": 0},
        ],
    )
    records = aggregate_split_records(split, [0.2, 0.9, 0.4], aggregation="max")
    assert records[0]["label"] == 1
    assert records[0]["step_labels"] == [0, 1]
    assert records[0]["score"] == pytest.approx(0.9)

    labels, scores = aggregate_split_scores(split, [0.2, 0.9, 0.4], aggregation="max")
    assert labels.tolist() == [1, 0]
    assert scores.tolist() == pytest.approx([0.9, 0.4])


def test_aggregate_split_records_and_group_diagnostics():
    split = _split(4)
    split.metas[0].update({"sample_index": 10, "source": "a", "sample_type": "x", "metadata": {"suite_name": "s1"}})
    split.metas[1].update({"sample_index": 10, "source": "a", "sample_type": "x", "metadata": {"suite_name": "s1"}})
    split.metas[2].update({"sample_index": 20, "source": "b", "sample_type": "y", "metadata": {"suite_name": "s2"}})
    split.metas[3].update({"sample_index": 20, "source": "b", "sample_type": "y", "metadata": {"suite_name": "s2"}})
    records = aggregate_split_records(split, [0.1, 0.7, 0.4, 0.8], aggregation="max")
    assert records[0]["score"] == pytest.approx(0.7)
    assert records[1]["score"] == pytest.approx(0.8)

    groups = group_diagnostics(records, threshold=0.75)
    assert groups["source"]["a"]["tn"] == 1
    assert groups["source"]["b"]["tp"] == 1
    assert groups["metadata.suite_name"]["s2"]["tpr"] == 1.0


def test_threshold_diagnostics_report_confusion_and_tradeoffs():
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.7, 0.6, 0.9])
    confusion = confusion_at_threshold(labels, scores, 0.65)
    assert confusion["tp"] == 1
    assert confusion["fp"] == 1
    assert confusion["fn"] == 1
    assert confusion["fpr"] == pytest.approx(0.5)

    tradeoffs = threshold_tradeoffs(labels, scores, [0.0, 0.5])
    assert tradeoffs[0]["fpr"] == 0.0
    assert tradeoffs[1]["fpr"] <= 0.5
    assert tradeoffs[1]["tpr"] >= tradeoffs[0]["tpr"]


def test_train_hist_gradient_drift_synthetic_high_auc():
    detector = train_hist_gradient_drift(
        model_name="fake",
        train=_split(80),
        val=_split(40),
        test=_split(40),
        n_anchors=10,
        params={"max_iter": 50, "random_state": 0},
        sample_weight="balanced",
    )
    assert detector.val_metrics["auroc"] > 0.95
    assert detector.test_metrics is not None
    assert detector.test_metrics["auroc"] > 0.95
    assert detector.diagnostics["sample_weight"] == "balanced"
    assert "val_threshold_tradeoffs" in detector.diagnostics


def test_train_hist_gradient_drift_can_disable_anchor_distances():
    detector = train_hist_gradient_drift(
        model_name="fake",
        train=_split(80),
        val=_split(40),
        n_anchors=10,
        include_anchor_distances=False,
        aggregation="top2_mean",
        params={"max_iter": 50, "random_state": 0},
        threshold_max_fpr=0.25,
    )
    assert detector.benign_incremental_anchors is None
    assert detector.benign_global_anchors is None
    assert detector.feature_config.include_anchor_distances is False
    assert detector.feature_config.aggregation == "top2_mean"
    assert detector.params["threshold_max_fpr"] == 0.25
