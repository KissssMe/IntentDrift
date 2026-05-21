from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secmcp.activations.anchor import diff_features, distance_features, select_anchors
from secmcp.activations.dataset import ActivationSplit, load_activation_split
from secmcp.config import OUTPUTS_DIR, load_training_cfg
from secmcp.detectors.evaluate import binary_metrics, predict_scores


@dataclass(frozen=True)
class TrainedDetector:
    model_name: str
    detector_type: str
    classifier: Any
    anchors: Any
    feature_type: str
    feature_mode: str
    params: dict[str, Any]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


def detector_output_dir(model_name: str, output_root: Path | None = None) -> Path:
    return (output_root or OUTPUTS_DIR / "detectors") / model_name


def _to_numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return tensor


def train_rf_anchor(
    *,
    model_name: str,
    train: ActivationSplit,
    val: ActivationSplit,
    test: ActivationSplit | None = None,
    n_anchors: int = 1000,
    feature_mode: str = "concat",
    seed: int = 42,
    rf_params: dict[str, Any] | None = None,
    show_progress: bool = False,
) -> TrainedDetector:
    from sklearn.ensemble import RandomForestClassifier

    if show_progress:
        print(f"[train] selecting anchors n={n_anchors}", file=sys.stderr, flush=True)
    anchors = select_anchors(train.activations, train.labels, n_anchors=n_anchors, seed=seed)
    if show_progress:
        print("[train] building train distance features", file=sys.stderr, flush=True)
    x_train = distance_features(train.activations, anchors, mode=feature_mode)
    if show_progress:
        print(f"[train] train features shape={tuple(x_train.shape)}", file=sys.stderr, flush=True)
        print("[train] building val distance features", file=sys.stderr, flush=True)
    x_val = distance_features(val.activations, anchors, mode=feature_mode)
    if show_progress:
        print(f"[train] val features shape={tuple(x_val.shape)}", file=sys.stderr, flush=True)

    params = dict(rf_params or {})
    params.setdefault("random_state", seed)
    if show_progress:
        print(f"[train] fitting RandomForestClassifier params={params}", file=sys.stderr, flush=True)
    clf = RandomForestClassifier(**params)
    clf.fit(_to_numpy(x_train), _to_numpy(train.labels))

    if show_progress:
        print("[train] scoring val", file=sys.stderr, flush=True)
    val_scores = predict_scores(clf, x_val)
    val_metrics = binary_metrics(_to_numpy(val.labels), val_scores).to_dict()
    if show_progress:
        print(f"[train] val metrics={val_metrics}", file=sys.stderr, flush=True)

    test_metrics = None
    if test is not None:
        if show_progress:
            print("[train] building test distance features", file=sys.stderr, flush=True)
        x_test = distance_features(test.activations, anchors, mode=feature_mode)
        if show_progress:
            print(f"[train] test features shape={tuple(x_test.shape)}", file=sys.stderr, flush=True)
            print("[train] scoring test", file=sys.stderr, flush=True)
        test_scores = predict_scores(clf, x_test)
        test_metrics = binary_metrics(_to_numpy(test.labels), test_scores).to_dict()
        if show_progress:
            print(f"[train] test metrics={test_metrics}", file=sys.stderr, flush=True)

    return TrainedDetector(
        model_name=model_name,
        detector_type="rf_anchor",
        classifier=clf,
        anchors=anchors,
        feature_type="distance",
        feature_mode=feature_mode,
        params={"n_anchors": int(anchors.shape[0]), **params},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def train_logistic_diff(
    *,
    model_name: str,
    train: ActivationSplit,
    val: ActivationSplit,
    test: ActivationSplit | None = None,
    n_anchors: int = 1000,
    feature_mode: str = "concat",
    seed: int = 42,
    logistic_params: dict[str, Any] | None = None,
    show_progress: bool = False,
) -> TrainedDetector:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if show_progress:
        print(f"[train] selecting anchors n={n_anchors}", file=sys.stderr, flush=True)
    anchors = select_anchors(train.activations, train.labels, n_anchors=n_anchors, seed=seed)
    if show_progress:
        print("[train] building train diff features", file=sys.stderr, flush=True)
    x_train = diff_features(train.activations, anchors, mode=feature_mode)
    if show_progress:
        print(f"[train] train features shape={tuple(x_train.shape)}", file=sys.stderr, flush=True)
        print("[train] building val diff features", file=sys.stderr, flush=True)
    x_val = diff_features(val.activations, anchors, mode=feature_mode)
    if show_progress:
        print(f"[train] val features shape={tuple(x_val.shape)}", file=sys.stderr, flush=True)

    params = dict(logistic_params or {})
    params.setdefault("random_state", seed)
    if show_progress:
        print(f"[train] fitting LogisticRegression params={params}", file=sys.stderr, flush=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
    clf.fit(_to_numpy(x_train), _to_numpy(train.labels))

    if show_progress:
        print("[train] scoring val", file=sys.stderr, flush=True)
    val_scores = predict_scores(clf, x_val)
    val_metrics = binary_metrics(_to_numpy(val.labels), val_scores).to_dict()
    if show_progress:
        print(f"[train] val metrics={val_metrics}", file=sys.stderr, flush=True)

    test_metrics = None
    if test is not None:
        if show_progress:
            print("[train] building test diff features", file=sys.stderr, flush=True)
        x_test = diff_features(test.activations, anchors, mode=feature_mode)
        if show_progress:
            print(f"[train] test features shape={tuple(x_test.shape)}", file=sys.stderr, flush=True)
            print("[train] scoring test", file=sys.stderr, flush=True)
        test_scores = predict_scores(clf, x_test)
        test_metrics = binary_metrics(_to_numpy(test.labels), test_scores).to_dict()
        if show_progress:
            print(f"[train] test metrics={test_metrics}", file=sys.stderr, flush=True)

    return TrainedDetector(
        model_name=model_name,
        detector_type="logistic_diff",
        classifier=clf,
        anchors=anchors,
        feature_type="diff",
        feature_mode=feature_mode,
        params={"n_anchors": int(anchors.shape[0]), **params},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def save_detector(detector: TrainedDetector, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{detector.detector_type}_best.pkl"
    metrics_path = output_dir / f"{detector.detector_type}_metrics.json"
    with model_path.open("wb") as f:
        pickle.dump(detector, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": detector.model_name,
                "detector_type": detector.detector_type,
                "feature_type": detector.feature_type,
                "feature_mode": detector.feature_mode,
                "params": detector.params,
                "val_metrics": detector.val_metrics,
                "test_metrics": detector.test_metrics,
            },
            f,
            indent=2,
        )
    return model_path, metrics_path


def train_detector_from_disk(
    *,
    model_name: str,
    detector_type: str,
    activation_root: Path | None = None,
    output_root: Path | None = None,
    n_anchors: int | None = None,
    include_test: bool = True,
    show_progress: bool = False,
) -> TrainedDetector:
    cfg = load_training_cfg()
    train = load_activation_split(model_name, "train", activation_root, show_progress=show_progress)
    val = load_activation_split(model_name, "val", activation_root, show_progress=show_progress)
    test = (
        load_activation_split(model_name, "test", activation_root, show_progress=show_progress)
        if include_test
        else None
    )
    seed = int(getattr(cfg.splits, "random_seed", 42)) if hasattr(cfg, "splits") else 42
    feature_mode = getattr(cfg.layers, "mode", "concat")
    anchor_n = int(n_anchors or cfg.anchor.default)

    if detector_type == "rf_anchor":
        detector = train_rf_anchor(
            model_name=model_name,
            train=train,
            val=val,
            test=test,
            n_anchors=anchor_n,
            feature_mode=feature_mode,
            seed=getattr(cfg.rf_anchor, "random_state", seed),
            rf_params=vars(cfg.rf_anchor),
            show_progress=show_progress,
        )
    elif detector_type == "logistic_diff":
        logistic_params = vars(cfg.logistic).copy()
        c_values = logistic_params.pop("C", [1.0])
        if isinstance(c_values, list):
            logistic_params["C"] = c_values[0]
        else:
            logistic_params["C"] = c_values
        detector = train_logistic_diff(
            model_name=model_name,
            train=train,
            val=val,
            test=test,
            n_anchors=anchor_n,
            feature_mode=feature_mode,
            seed=getattr(cfg.logistic, "random_state", seed),
            logistic_params=logistic_params,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")

    output_dir = detector_output_dir(model_name, output_root)
    if show_progress:
        print(f"[train] saving detector to {output_dir}", file=sys.stderr, flush=True)
    save_detector(detector, output_dir)
    return detector
