from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secmcp.activations.anchor import distance_features, flatten_activations, select_anchors
from secmcp.activations.drift_dataset import DriftActivationSplit, load_drift_split
from secmcp.config import OUTPUTS_DIR, load_training_cfg
from secmcp.detectors.evaluate import binary_metrics, predict_scores


@dataclass(frozen=True)
class DriftFeatureConfig:
    feature_mode: str = "concat"
    include_anchor_distances: bool = True
    aggregation: str = "max"


@dataclass(frozen=True)
class TrainedDriftDetector:
    model_name: str
    classifier: Any
    benign_incremental_anchors: Any
    benign_global_anchors: Any
    feature_config: DriftFeatureConfig
    threshold: float
    params: dict[str, Any]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


def _to_numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return tensor


def _layerwise_activations(activations, mode: str = "concat"):
    if mode == "concat":
        return activations.float()
    if mode == "last_only":
        return activations[:, -1:, :].float()
    raise ValueError(f"Unknown layer feature mode: {mode}")


def aggregate_step_scores(scores, aggregation: str = "max") -> float:
    if len(scores) == 0:
        return 0.0
    if aggregation == "max":
        return float(max(scores))
    if aggregation in {"top2_mean", "top-2 mean", "top_2_mean"}:
        top_scores = sorted((float(score) for score in scores), reverse=True)[:2]
        return float(sum(top_scores) / len(top_scores))
    raise ValueError(f"Unknown drift score aggregation: {aggregation}")


def aggregate_split_scores(split: DriftActivationSplit, scores, aggregation: str = "max"):
    import numpy as np

    grouped: dict[Any, dict[str, Any]] = {}
    labels = _to_numpy(split.labels)
    for idx, (label, score, meta) in enumerate(zip(labels, scores, split.metas)):
        key = meta.get("sample_index", idx)
        bucket = grouped.setdefault(key, {"label": int(label), "scores": []})
        if bucket["label"] != int(label):
            raise ValueError(f"Inconsistent labels for trajectory {key}")
        bucket["scores"].append(float(score))

    trajectory_labels = np.array([bucket["label"] for bucket in grouped.values()])
    trajectory_scores = np.array(
        [aggregate_step_scores(bucket["scores"], aggregation=aggregation) for bucket in grouped.values()]
    )
    return trajectory_labels, trajectory_scores


def drift_feature_matrix(
    split: DriftActivationSplit,
    *,
    benign_incremental_anchors=None,
    benign_global_anchors=None,
    feature_mode: str = "concat",
    include_anchor_distances: bool = True,
):
    import torch

    eps = 1e-6
    incremental = split.post - split.history
    global_drift = split.post - split.task
    history_from_task = split.history - split.task

    inc_flat = flatten_activations(incremental, mode=feature_mode)
    glob_flat = flatten_activations(global_drift, mode=feature_mode)
    inc_layers = _layerwise_activations(incremental, mode=feature_mode)
    glob_layers = _layerwise_activations(global_drift, mode=feature_mode)
    hist_layers = _layerwise_activations(history_from_task, mode=feature_mode)
    inc_norm = torch.linalg.vector_norm(inc_layers, dim=2)
    glob_norm = torch.linalg.vector_norm(glob_layers, dim=2)
    hist_norm = torch.linalg.vector_norm(hist_layers, dim=2)
    relative_norm = inc_norm / (hist_norm + eps)
    cosine = torch.nn.functional.cosine_similarity(inc_layers, glob_layers, dim=2)

    parts = [inc_flat, glob_flat, inc_norm, glob_norm, hist_norm, relative_norm, cosine]
    if include_anchor_distances:
        if benign_incremental_anchors is None or benign_global_anchors is None:
            raise ValueError("Anchor distance features require both anchor sets")
        parts.append(distance_features(incremental, benign_incremental_anchors, mode=feature_mode))
        parts.append(distance_features(global_drift, benign_global_anchors, mode=feature_mode))
    return torch.cat(parts, dim=1)


def _choose_threshold(labels, scores, target_tpr: float = 0.95) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return 0.5
    # Among thresholds reaching target TPR, choose the one with lowest FPR.
    best = valid[np.argmin(fpr[valid])]
    return float(thresholds[best])


def train_hist_gradient_drift(
    *,
    model_name: str,
    train: DriftActivationSplit,
    val: DriftActivationSplit,
    test: DriftActivationSplit | None = None,
    n_anchors: int = 1000,
    feature_mode: str = "concat",
    include_anchor_distances: bool = True,
    aggregation: str = "max",
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> TrainedDriftDetector:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if include_anchor_distances:
        incremental_train = train.post - train.history
        global_train = train.post - train.task
        benign_incremental_anchors = select_anchors(incremental_train, train.labels, n_anchors=n_anchors, seed=seed)
        benign_global_anchors = select_anchors(global_train, train.labels, n_anchors=n_anchors, seed=seed)
    else:
        benign_incremental_anchors = None
        benign_global_anchors = None

    x_train = drift_feature_matrix(
        train,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_mode=feature_mode,
        include_anchor_distances=include_anchor_distances,
    )
    x_val = drift_feature_matrix(
        val,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_mode=feature_mode,
        include_anchor_distances=include_anchor_distances,
    )

    clf_params = dict(params or {})
    clf_params.setdefault("random_state", seed)
    clf = HistGradientBoostingClassifier(**clf_params)
    clf.fit(_to_numpy(x_train), _to_numpy(train.labels))

    val_step_scores = predict_scores(clf, x_val)
    val_labels, val_scores = aggregate_split_scores(val, val_step_scores, aggregation=aggregation)
    threshold = _choose_threshold(val_labels, val_scores)
    val_metrics = binary_metrics(val_labels, val_scores, threshold=threshold).to_dict()

    test_metrics = None
    if test is not None:
        x_test = drift_feature_matrix(
            test,
            benign_incremental_anchors=benign_incremental_anchors,
            benign_global_anchors=benign_global_anchors,
            feature_mode=feature_mode,
            include_anchor_distances=include_anchor_distances,
        )
        test_step_scores = predict_scores(clf, x_test)
        test_labels, test_scores = aggregate_split_scores(test, test_step_scores, aggregation=aggregation)
        test_metrics = binary_metrics(test_labels, test_scores, threshold=threshold).to_dict()

    return TrainedDriftDetector(
        model_name=model_name,
        classifier=clf,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_config=DriftFeatureConfig(
            feature_mode=feature_mode,
            include_anchor_distances=include_anchor_distances,
            aggregation=aggregation,
        ),
        threshold=threshold,
        params={
            "n_anchors": int(benign_incremental_anchors.shape[0]) if benign_incremental_anchors is not None else 0,
            **clf_params,
        },
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def drift_detector_output_dir(model_name: str, output_root: Path | None = None) -> Path:
    return (output_root or OUTPUTS_DIR / "detectors") / model_name


def save_drift_detector(detector: TrainedDriftDetector, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "task_drift_best.pkl"
    metrics_path = output_dir / "task_drift_metrics.json"
    with model_path.open("wb") as f:
        pickle.dump(detector, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": detector.model_name,
                "detector_type": "task_drift",
                "feature_config": detector.feature_config.__dict__,
                "threshold": detector.threshold,
                "params": detector.params,
                "val_metrics": detector.val_metrics,
                "test_metrics": detector.test_metrics,
            },
            f,
            indent=2,
        )
    return model_path, metrics_path


def train_drift_detector_from_disk(
    *,
    model_name: str,
    drift_activation_root: Path | None = None,
    output_root: Path | None = None,
    n_anchors: int | None = None,
    include_test: bool = True,
) -> TrainedDriftDetector:
    cfg = load_training_cfg()
    train = load_drift_split(model_name, "train", drift_activation_root)
    val = load_drift_split(model_name, "val", drift_activation_root)
    test = load_drift_split(model_name, "test", drift_activation_root) if include_test else None
    seed = int(getattr(cfg.splits, "random_seed", 42)) if hasattr(cfg, "splits") else 42
    feature_mode = getattr(cfg.layers, "mode", "concat")
    anchor_n = int(n_anchors or cfg.anchor.default)
    task_drift_cfg = getattr(cfg, "task_drift", object())
    include_anchor_distances = bool(getattr(task_drift_cfg, "include_anchor_distances", True))
    aggregation = str(getattr(task_drift_cfg, "aggregation", "max"))
    params = {
        "max_iter": int(getattr(task_drift_cfg, "max_iter", 200)),
        "learning_rate": float(getattr(task_drift_cfg, "learning_rate", 0.05)),
        "max_leaf_nodes": int(getattr(task_drift_cfg, "max_leaf_nodes", 31)),
        "l2_regularization": float(getattr(task_drift_cfg, "l2_regularization", 0.0)),
    }
    detector = train_hist_gradient_drift(
        model_name=model_name,
        train=train,
        val=val,
        test=test,
        n_anchors=anchor_n,
        feature_mode=feature_mode,
        include_anchor_distances=include_anchor_distances,
        aggregation=aggregation,
        seed=seed,
        params=params,
    )
    save_drift_detector(detector, drift_detector_output_dir(model_name, output_root))
    return detector
