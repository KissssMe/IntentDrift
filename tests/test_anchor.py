from __future__ import annotations

import pytest

from secmcp.activations.anchor import diff_features, distance_features, flatten_activations, select_anchors


torch = pytest.importorskip("torch")


def test_flatten_activations_concat_and_last_only():
    x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert tuple(flatten_activations(x, "concat").shape) == (2, 12)
    assert tuple(flatten_activations(x, "last_only").shape) == (2, 4)


def test_select_anchors_only_benign():
    x = torch.randn(5, 2, 3)
    y = torch.tensor([1, 0, 1, 0, 0])
    anchors = select_anchors(x, y, n_anchors=2, seed=0)
    assert anchors.shape == (2, 2, 3)
    benign = {tuple(row.flatten().tolist()) for row in x[y == 0]}
    assert all(tuple(row.flatten().tolist()) in benign for row in anchors)


def test_distance_features_shape_and_nonnegative():
    samples = torch.randn(4, 2, 3)
    anchors = torch.randn(3, 2, 3)
    feats = distance_features(samples, anchors, chunk_size=2)
    assert tuple(feats.shape) == (4, 4)
    assert torch.all(feats >= 0)


def test_diff_features_shape():
    samples = torch.randn(4, 2, 3)
    anchors = torch.randn(3, 2, 3)
    feats = diff_features(samples, anchors)
    assert tuple(feats.shape) == (4, 6)
