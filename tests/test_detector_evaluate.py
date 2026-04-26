from __future__ import annotations

import numpy as np

from secmcp.detectors.evaluate import binary_metrics, fpr_at_tpr


def test_binary_metrics_perfect_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, scores)
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.fpr_at_95tpr == 0.0


def test_fpr_at_tpr_returns_float():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.8])
    value = fpr_at_tpr(labels, scores, 0.5)
    assert isinstance(value, float)
