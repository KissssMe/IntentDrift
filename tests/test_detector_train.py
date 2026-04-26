from __future__ import annotations

import pickle

import pytest

from secmcp.activations.dataset import ActivationSplit
from secmcp.detectors.train import save_detector, train_logistic_diff, train_rf_anchor


torch = pytest.importorskip("torch")


def _synthetic_split(n: int, offset: float = 2.5) -> ActivationSplit:
    n0 = n // 2
    n1 = n - n0
    benign = torch.randn(n0, 2, 4) * 0.1
    malicious = torch.randn(n1, 2, 4) * 0.1 + offset
    x = torch.cat([benign, malicious], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    metas = [{"sample_index": i} for i in range(n)]
    return ActivationSplit(x, y, metas)


def test_train_rf_anchor_synthetic_high_auc(tmp_path):
    train = _synthetic_split(60)
    val = _synthetic_split(30)
    test = _synthetic_split(30)
    detector = train_rf_anchor(
        model_name="fake",
        train=train,
        val=val,
        test=test,
        n_anchors=10,
        rf_params={"n_estimators": 20, "max_depth": 5, "random_state": 0, "class_weight": "balanced"},
    )
    assert detector.val_metrics["auroc"] > 0.95
    assert detector.test_metrics is not None
    assert detector.test_metrics["auroc"] > 0.95
    model_path, metrics_path = save_detector(detector, tmp_path)
    assert model_path.exists()
    assert metrics_path.exists()
    with model_path.open("rb") as f:
        loaded = pickle.load(f)
    assert loaded.detector_type == "rf_anchor"


def test_train_logistic_diff_synthetic_high_auc():
    train = _synthetic_split(40)
    val = _synthetic_split(20)
    detector = train_logistic_diff(
        model_name="fake",
        train=train,
        val=val,
        n_anchors=8,
        logistic_params={"max_iter": 200, "solver": "lbfgs", "class_weight": "balanced"},
    )
    assert detector.val_metrics["auroc"] > 0.95
    assert detector.feature_type == "diff"
