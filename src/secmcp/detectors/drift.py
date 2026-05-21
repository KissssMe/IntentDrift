from __future__ import annotations

import json
import pickle
import sys
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
    include_self_baseline: bool = True
    aggregation: str = "max"


@dataclass
class TrajectoryPriorNorms:
    """Per-trajectory running history of layer-wise drift norms.

    Used to z-score the current step's drift against the trajectory's own
    prior benign-or-not steps; this neutralizes domain/source-level
    differences in baseline drift magnitude (long technical tool outputs vs
    short structured ones) and surfaces only the *anomalous-for-this-trajectory*
    component, which the detector should care about.
    """

    inc_norm: list = None  # list of tensor [n_layers], ordered chronologically
    glob_norm: list = None
    hist_norm: list = None
    relative_norm: list = None

    def __post_init__(self) -> None:
        for name in ("inc_norm", "glob_norm", "hist_norm", "relative_norm"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, [])

    def append(self, inc, glob, hist, rel) -> None:
        self.inc_norm.append(inc)
        self.glob_norm.append(glob)
        self.hist_norm.append(hist)
        self.relative_norm.append(rel)


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
    diagnostics: dict[str, Any]
    global_baseline: dict | None = None


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


def _parse_aggregation_params(aggregation: str, prefix: str) -> dict[str, float]:
    """Parse trailing tokens like ``cusum_w5_s0.1`` → {window: 5, slack: 0.1}."""
    parts = aggregation.removeprefix(prefix).split("_")
    out: dict[str, float] = {}
    for part in parts:
        if not part:
            continue
        if part.startswith("w"):
            out["window"] = float(part[1:])
        elif part.startswith("s"):
            out["slack"] = float(part[1:])
        elif part.startswith("r"):
            out["reference"] = float(part[1:])
        else:
            try:
                out["window"] = float(part)
            except ValueError:
                pass
    return out


def aggregate_step_scores(scores, aggregation: str = "max") -> float:
    """Reduce per-step detector scores to one trajectory score.

    Supported strategies:

    - ``"max"`` — single highest step score. Structurally pessimistic: long
      benign trajectories suffer FPR ≈ 1 − (1 − step_fpr)^n.
    - ``"top2_mean"`` — mean of top-2 step scores.
    - ``"first_exceed_K"`` (K an int, e.g. ``"first_exceed_3"``) — max over
      the first K tool steps only. Bounds the detection budget to early
      tool turns where injection is most likely; later benign steps cannot
      trigger an abort.
    - ``"cusum"`` / ``"cusum_wN_sX_rY"`` — Page CUSUM with running sum
      ``S_t = max(0, S_{t-1} + score_t - reference - slack)``; returns
      ``max S_t``. Defaults: ``reference=0.5`` (sigmoid mid-point),
      ``slack=0.0``, ``window=None``. Sustained elevation above the
      reference triggers; isolated spikes decay.
    """
    if len(scores) == 0:
        return 0.0
    if aggregation == "max":
        return float(max(scores))
    if aggregation in {"top2_mean", "top-2 mean", "top_2_mean"}:
        top_scores = sorted((float(score) for score in scores), reverse=True)[:2]
        return float(sum(top_scores) / len(top_scores))
    if aggregation == "mean":
        return float(sum(float(s) for s in scores) / len(scores))
    if aggregation.startswith("first_exceed_"):
        try:
            k = int(aggregation.rsplit("_", 1)[1])
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"first_exceed aggregation needs trailing integer: {aggregation!r}") from exc
        if k <= 0:
            raise ValueError(f"first_exceed_K requires K > 0, got {k}")
        return float(max(float(s) for s in list(scores)[:k]))
    if aggregation == "cusum" or aggregation.startswith("cusum_"):
        params = _parse_aggregation_params(aggregation, "cusum_")
        reference = float(params.get("reference", 0.5))
        slack = float(params.get("slack", 0.0))
        window = int(params["window"]) if "window" in params else None
        items = list(scores) if window is None else list(scores)[: max(1, window)]
        s = 0.0
        s_max = 0.0
        for x in items:
            s = max(0.0, s + float(x) - reference - slack)
            if s > s_max:
                s_max = s
        return float(s_max)
    raise ValueError(f"Unknown drift score aggregation: {aggregation}")


def aggregate_split_scores(split: DriftActivationSplit, scores, aggregation: str = "max"):
    import numpy as np

    records = aggregate_split_records(split, scores, aggregation=aggregation)
    trajectory_labels = np.array([record["label"] for record in records])
    trajectory_scores = np.array([record["score"] for record in records])
    return trajectory_labels, trajectory_scores


def aggregate_split_records(split: DriftActivationSplit, scores, aggregation: str = "max") -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    labels = _to_numpy(split.labels)
    for idx, (label, score, meta) in enumerate(zip(labels, scores, split.metas)):
        key = meta.get("sample_index", idx)
        trajectory_label = int(meta.get("trajectory_label", int(label)))
        bucket = grouped.setdefault(
            key,
            {"label": trajectory_label, "step_labels": [], "scores": [], "meta": meta},
        )
        if bucket["label"] != trajectory_label:
            raise ValueError(f"Inconsistent trajectory labels for trajectory {key}")
        bucket["step_labels"].append(int(label))
        bucket["scores"].append(float(score))
    return [
        {
            "sample_index": key,
            "label": bucket["label"],
            "step_labels": bucket["step_labels"],
            "score": aggregate_step_scores(bucket["scores"], aggregation=aggregation),
            "meta": bucket["meta"],
        }
        for key, bucket in grouped.items()
    ]


_SELF_BASELINE_MIN_PRIOR = 2


def _z_score_against_prior(current, prior_list, global_stats: tuple | None = None):
    """Z-score ``current`` against the trajectory-local prior, falling back to
    ``global_stats`` (``(mean, std)`` over training benign steps) when fewer
    than ``_SELF_BASELINE_MIN_PRIOR`` prior values are available.

    Previously this returned zeros for early steps. That created a strong
    "feature block == 0" signal that always coincided with the first one or
    two tool steps — exactly the window where AgentDojo injections happen —
    so the classifier could shortcut on "early step ⇒ positive" instead of
    learning the actual drift pattern. Falling back to a population baseline
    keeps the feature on the same numeric scale as later steps and removes
    the shortcut.
    """
    import torch

    if len(prior_list) < _SELF_BASELINE_MIN_PRIOR:
        if global_stats is None:
            return torch.zeros_like(current)
        mean, std = global_stats
        return (current - mean) / (std + 1e-6)
    stacked = torch.stack(prior_list)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return (current - mean) / (std + 1e-6)


def _self_baseline_features(
    inc_norm,
    glob_norm,
    hist_norm,
    relative_norm,
    metas,
    prior_state: TrajectoryPriorNorms | None = None,
    global_baseline: dict | None = None,
):
    """For each row, z-score its layer-wise drift norms against the *prior*
    steps in the same trajectory.

    Two modes:

    - ``prior_state is None`` (batch / training): trajectories are reconstructed
      causally from ``metas[i]['sample_index']`` and ``['step_index']``; row i
      is z-scored against rows from the same trajectory whose step_index is
      strictly smaller. The first step of each trajectory yields zeros.
    - ``prior_state`` provided (runtime, single step): every row is z-scored
      against the same external prior state. Suitable when the caller maintains
      cross-call accumulator state.

    Output shape: ``[n_steps, 4 * n_layers]`` (concat of z-scored
    incremental-norm, global-norm, history-norm, relative-norm).
    """
    import torch
    from collections import defaultdict

    n, n_layers = inc_norm.shape
    z_inc = torch.zeros_like(inc_norm)
    z_glob = torch.zeros_like(glob_norm)
    z_hist = torch.zeros_like(hist_norm)
    z_rel = torch.zeros_like(relative_norm)

    gb = global_baseline or {}
    gb_inc = gb.get("inc_norm")
    gb_glob = gb.get("glob_norm")
    gb_hist = gb.get("hist_norm")
    gb_rel = gb.get("relative_norm")

    if prior_state is not None:
        # Single-step runtime: every row uses the same caller-supplied prior.
        for row in range(n):
            z_inc[row] = _z_score_against_prior(inc_norm[row], prior_state.inc_norm, gb_inc)
            z_glob[row] = _z_score_against_prior(glob_norm[row], prior_state.glob_norm, gb_glob)
            z_hist[row] = _z_score_against_prior(hist_norm[row], prior_state.hist_norm, gb_hist)
            z_rel[row] = _z_score_against_prior(relative_norm[row], prior_state.relative_norm, gb_rel)
        return torch.cat([z_inc, z_glob, z_hist, z_rel], dim=1)

    by_traj: dict = defaultdict(list)
    for row, meta in enumerate(metas):
        sample_idx = meta.get("sample_index", row)
        step_idx = meta.get("step_index", row)
        by_traj[sample_idx].append((step_idx, row))

    for trajectory_steps in by_traj.values():
        trajectory_steps.sort()
        prior_inc: list = []
        prior_glob: list = []
        prior_hist: list = []
        prior_rel: list = []
        for _, row in trajectory_steps:
            z_inc[row] = _z_score_against_prior(inc_norm[row], prior_inc, gb_inc)
            z_glob[row] = _z_score_against_prior(glob_norm[row], prior_glob, gb_glob)
            z_hist[row] = _z_score_against_prior(hist_norm[row], prior_hist, gb_hist)
            z_rel[row] = _z_score_against_prior(relative_norm[row], prior_rel, gb_rel)
            prior_inc.append(inc_norm[row])
            prior_glob.append(glob_norm[row])
            prior_hist.append(hist_norm[row])
            prior_rel.append(relative_norm[row])

    return torch.cat([z_inc, z_glob, z_hist, z_rel], dim=1)


def compute_benign_global_baseline(split: DriftActivationSplit, feature_mode: str = "concat") -> dict:
    """Compute (mean, std) over training benign tool steps for each layer-wise
    drift norm. Used as the early-step fallback in ``_self_baseline_features``
    instead of zero vectors.
    """
    import torch

    eps = 1e-6
    labels = _to_numpy(split.labels).astype(int)
    benign_mask = labels == 0
    if benign_mask.sum() < _SELF_BASELINE_MIN_PRIOR:
        return {}

    incremental = split.post - split.history
    global_drift = split.post - split.task
    history_from_task = split.history - split.task
    inc_layers = _layerwise_activations(incremental, mode=feature_mode)
    glob_layers = _layerwise_activations(global_drift, mode=feature_mode)
    hist_layers = _layerwise_activations(history_from_task, mode=feature_mode)
    inc_norm = torch.linalg.vector_norm(inc_layers, dim=2)
    glob_norm = torch.linalg.vector_norm(glob_layers, dim=2)
    hist_norm = torch.linalg.vector_norm(hist_layers, dim=2)
    relative_norm = inc_norm / (hist_norm + eps)

    mask = torch.from_numpy(benign_mask)
    out = {}
    for name, tensor in [
        ("inc_norm", inc_norm),
        ("glob_norm", glob_norm),
        ("hist_norm", hist_norm),
        ("relative_norm", relative_norm),
    ]:
        benign_rows = tensor[mask]
        out[name] = (benign_rows.mean(dim=0), benign_rows.std(dim=0, unbiased=False))
    return out


def drift_feature_matrix(
    split: DriftActivationSplit,
    *,
    benign_incremental_anchors=None,
    benign_global_anchors=None,
    feature_mode: str = "concat",
    include_anchor_distances: bool = True,
    include_self_baseline: bool = True,
    prior_state: TrajectoryPriorNorms | None = None,
    global_baseline: dict | None = None,
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
    if include_self_baseline:
        if prior_state is not None and inc_norm.shape[0] != 1:
            # The prior_state path is the runtime single-step contract; using
            # it on a multi-row split would silently apply one external prior
            # to every row instead of doing the causal per-trajectory scan.
            raise ValueError(
                "drift_feature_matrix(prior_state=...) is the runtime single-step path; "
                f"got a batch of {inc_norm.shape[0]} steps."
            )
        parts.append(
            _self_baseline_features(
                inc_norm,
                glob_norm,
                hist_norm,
                relative_norm,
                split.metas,
                prior_state=prior_state,
                global_baseline=global_baseline,
            )
        )
    return torch.cat(parts, dim=1)


def update_prior_norms_from_split(
    state: TrajectoryPriorNorms,
    split: DriftActivationSplit,
) -> None:
    """Append the layer-wise norms of the (typically single-step) split to the
    running prior state. Call after scoring at runtime so the next step's
    self-baseline reflects the just-observed step."""
    import torch

    eps = 1e-6
    incremental = split.post - split.history
    global_drift = split.post - split.task
    history_from_task = split.history - split.task
    inc_norm = torch.linalg.vector_norm(incremental, dim=2)
    glob_norm = torch.linalg.vector_norm(global_drift, dim=2)
    hist_norm = torch.linalg.vector_norm(history_from_task, dim=2)
    relative_norm = inc_norm / (hist_norm + eps)
    for row in range(inc_norm.shape[0]):
        state.append(inc_norm[row], glob_norm[row], hist_norm[row], relative_norm[row])


def confusion_at_threshold(labels, scores, threshold: float) -> dict[str, float | int]:
    import numpy as np

    labels = np.asarray(_to_numpy(labels))
    scores = np.asarray(_to_numpy(scores))
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    pos = tp + fn
    neg = fp + tn
    total = pos + neg
    return {
        "threshold": float(threshold),
        "total": int(total),
        "positive": int(pos),
        "negative": int(neg),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": float(tp / pos) if pos else 0.0,
        "fnr": float(fn / pos) if pos else 0.0,
        "fpr": float(fp / neg) if neg else 0.0,
        "tnr": float(tn / neg) if neg else 0.0,
        "accuracy": float((tp + tn) / total) if total else 0.0,
    }


def threshold_tradeoffs(labels, scores, target_fprs: list[float] | tuple[float, ...]) -> list[dict[str, float | int]]:
    import numpy as np

    labels = np.asarray(_to_numpy(labels))
    scores = np.asarray(_to_numpy(scores))
    thresholds = sorted({float(score) for score in scores}, reverse=True)
    results: list[dict[str, float | int]] = []
    for target_fpr in target_fprs:
        candidates = [confusion_at_threshold(labels, scores, threshold) for threshold in thresholds]
        valid = [row for row in candidates if float(row["fpr"]) <= float(target_fpr)]
        if valid:
            best = max(valid, key=lambda row: (float(row["tpr"]), -float(row["fpr"])))
        else:
            best = confusion_at_threshold(labels, scores, float("inf"))
        results.append({"target_fpr": float(target_fpr), **best})
    return results


def group_diagnostics(records: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, dict[str, float | int]]]:
    from collections import defaultdict

    def metadata_value(meta: dict[str, Any], field: str) -> str:
        if field.startswith("metadata."):
            value = (meta.get("metadata") or {}).get(field.split(".", 1)[1])
        else:
            value = meta.get(field)
        return str(value if value is not None else "unknown")

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("source", "sample_type", "metadata.suite_name"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            buckets[metadata_value(record.get("meta") or {}, field)].append(record)
        field_results: dict[str, dict[str, float | int]] = {}
        for value, rows in buckets.items():
            labels = [int(row["label"]) for row in rows]
            scores = [float(row["score"]) for row in rows]
            field_results[value] = confusion_at_threshold(labels, scores, threshold)
        results[field] = dict(sorted(field_results.items()))
    return results


def _choose_threshold(
    labels,
    scores,
    target_tpr: float = 0.95,
    max_fpr: float | None = None,
) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    if max_fpr is not None:
        valid = np.where(fpr <= max_fpr)[0]
        if len(valid) > 0:
            best_tpr = np.max(tpr[valid])
            best = valid[np.where(tpr[valid] == best_tpr)[0][-1]]
            return float(thresholds[best])
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return 0.5
    # Among thresholds reaching target TPR, choose the one with lowest FPR.
    best = valid[np.argmin(fpr[valid])]
    return float(thresholds[best])


def _balanced_sample_weights(labels):
    import numpy as np

    labels = _to_numpy(labels).astype(int)
    weights = np.ones(labels.shape[0], dtype=float)
    classes, counts = np.unique(labels, return_counts=True)
    total = float(labels.shape[0])
    for cls, count in zip(classes, counts):
        if count > 0:
            weights[labels == cls] = total / (len(classes) * float(count))
    return weights


def _filter_ignore_steps(features, labels):
    """Drop rows whose step label is ``-1`` (unlocatable-positive ignore mask).

    Returns ``(features_kept, labels_kept, kept_count, dropped_count)``.
    """
    import numpy as np

    label_arr = _to_numpy(labels).astype(int)
    keep_mask = label_arr != -1
    kept_count = int(keep_mask.sum())
    dropped_count = int((~keep_mask).sum())
    if dropped_count == 0:
        return features, labels, kept_count, dropped_count
    feats_np = _to_numpy(features)
    return feats_np[keep_mask], label_arr[keep_mask], kept_count, dropped_count


def train_hist_gradient_drift(
    *,
    model_name: str,
    train: DriftActivationSplit,
    val: DriftActivationSplit,
    test: DriftActivationSplit | None = None,
    n_anchors: int = 1000,
    feature_mode: str = "concat",
    include_anchor_distances: bool = True,
    include_self_baseline: bool = True,
    aggregation: str = "max",
    seed: int = 42,
    params: dict[str, Any] | None = None,
    threshold_target_tpr: float = 0.95,
    threshold_max_fpr: float | None = None,
    sample_weight: str | None = None,
    show_progress: bool = False,
) -> TrainedDriftDetector:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if include_anchor_distances:
        if show_progress:
            print(f"[train] selecting anchors n={n_anchors}", file=sys.stderr, flush=True)
        incremental_train = train.post - train.history
        global_train = train.post - train.task
        benign_incremental_anchors = select_anchors(incremental_train, train.labels, n_anchors=n_anchors, seed=seed)
        benign_global_anchors = select_anchors(global_train, train.labels, n_anchors=n_anchors, seed=seed)
    else:
        benign_incremental_anchors = None
        benign_global_anchors = None

    global_baseline = compute_benign_global_baseline(train, feature_mode=feature_mode) if include_self_baseline else None
    if show_progress and global_baseline:
        print(
            f"[train] computed global benign baseline keys={list(global_baseline.keys())}",
            file=sys.stderr,
            flush=True,
        )
    if show_progress:
        print("[train] building train feature matrix", file=sys.stderr, flush=True)
    x_train = drift_feature_matrix(
        train,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_mode=feature_mode,
        include_anchor_distances=include_anchor_distances,
        include_self_baseline=include_self_baseline,
        global_baseline=global_baseline,
    )
    if show_progress:
        print(f"[train] train features shape={tuple(x_train.shape)}", file=sys.stderr, flush=True)
        print("[train] building val feature matrix", file=sys.stderr, flush=True)
    x_val = drift_feature_matrix(
        val,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_mode=feature_mode,
        include_anchor_distances=include_anchor_distances,
        include_self_baseline=include_self_baseline,
        global_baseline=global_baseline,
    )
    if show_progress:
        print(f"[train] val features shape={tuple(x_val.shape)}", file=sys.stderr, flush=True)

    clf_params = dict(params or {})
    clf_params.setdefault("random_state", seed)
    x_train_fit, y_train_fit, kept_steps, dropped_steps = _filter_ignore_steps(x_train, train.labels)
    if show_progress and dropped_steps:
        print(
            f"[train] dropping {dropped_steps} ignore-labelled steps (kept={kept_steps})",
            file=sys.stderr,
            flush=True,
        )
    if show_progress:
        print(f"[train] fitting HistGradientBoostingClassifier params={clf_params}", file=sys.stderr, flush=True)
    clf = HistGradientBoostingClassifier(**clf_params)
    fit_kwargs = {}
    if sample_weight == "balanced":
        fit_kwargs["sample_weight"] = _balanced_sample_weights(y_train_fit)
    elif sample_weight not in {None, "none"}:
        raise ValueError(f"Unknown task-drift sample_weight mode: {sample_weight}")
    clf.fit(_to_numpy(x_train_fit), _to_numpy(y_train_fit), **fit_kwargs)

    if show_progress:
        print("[train] scoring val and choosing threshold", file=sys.stderr, flush=True)
    import numpy as np

    val_step_scores = predict_scores(clf, x_val)
    val_records = aggregate_split_records(val, val_step_scores, aggregation=aggregation)
    val_labels = np.array([record["label"] for record in val_records])
    val_scores = np.array([record["score"] for record in val_records])
    threshold = _choose_threshold(val_labels, val_scores, target_tpr=threshold_target_tpr, max_fpr=threshold_max_fpr)
    val_metrics = binary_metrics(val_labels, val_scores, threshold=threshold).to_dict()
    val_confusion = confusion_at_threshold(val_labels, val_scores, threshold)
    diagnostics: dict[str, Any] = {
        "threshold_target_tpr": float(threshold_target_tpr),
        "threshold_max_fpr": None if threshold_max_fpr is None else float(threshold_max_fpr),
        "sample_weight": sample_weight or "none",
        "aggregation": aggregation,
        # `val_confusion` is already trajectory-level (we aggregated step
        # scores via aggregation before thresholding). Surface the FPR under
        # an explicit name so it isn't confused with step-level metrics.
        "val_trajectory_abort_rate": val_confusion.get("fpr"),
        "val_confusion": val_confusion,
        "val_threshold_tradeoffs": threshold_tradeoffs(val_labels, val_scores, [0.05, 0.10, 0.20, 0.30, 0.50]),
        "val_group_diagnostics": group_diagnostics(val_records, threshold),
    }
    if show_progress:
        print(f"[train] val metrics={val_metrics}", file=sys.stderr, flush=True)

    test_metrics = None
    if test is not None:
        if show_progress:
            print("[train] building test feature matrix", file=sys.stderr, flush=True)
        x_test = drift_feature_matrix(
            test,
            benign_incremental_anchors=benign_incremental_anchors,
            benign_global_anchors=benign_global_anchors,
            feature_mode=feature_mode,
            include_anchor_distances=include_anchor_distances,
            include_self_baseline=include_self_baseline,
            global_baseline=global_baseline,
        )
        if show_progress:
            print(f"[train] test features shape={tuple(x_test.shape)}", file=sys.stderr, flush=True)
            print("[train] scoring test", file=sys.stderr, flush=True)
        test_step_scores = predict_scores(clf, x_test)
        test_records = aggregate_split_records(test, test_step_scores, aggregation=aggregation)
        test_labels = np.array([record["label"] for record in test_records])
        test_scores = np.array([record["score"] for record in test_records])
        test_metrics = binary_metrics(test_labels, test_scores, threshold=threshold).to_dict()
        test_confusion = confusion_at_threshold(test_labels, test_scores, threshold)
        diagnostics["test_trajectory_abort_rate"] = test_confusion.get("fpr")
        diagnostics["test_confusion"] = test_confusion
        diagnostics["test_threshold_tradeoffs"] = threshold_tradeoffs(
            test_labels, test_scores, [0.05, 0.10, 0.20, 0.30, 0.50]
        )
        diagnostics["test_group_diagnostics"] = group_diagnostics(test_records, threshold)
        if show_progress:
            print(f"[train] test metrics={test_metrics}", file=sys.stderr, flush=True)

    return TrainedDriftDetector(
        model_name=model_name,
        classifier=clf,
        benign_incremental_anchors=benign_incremental_anchors,
        benign_global_anchors=benign_global_anchors,
        feature_config=DriftFeatureConfig(
            feature_mode=feature_mode,
            include_anchor_distances=include_anchor_distances,
            include_self_baseline=include_self_baseline,
            aggregation=aggregation,
        ),
        threshold=threshold,
        params={
            "n_anchors": int(benign_incremental_anchors.shape[0]) if benign_incremental_anchors is not None else 0,
            "sample_weight": sample_weight or "none",
            "threshold_target_tpr": float(threshold_target_tpr),
            "threshold_max_fpr": None if threshold_max_fpr is None else float(threshold_max_fpr),
            **clf_params,
        },
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        diagnostics=diagnostics,
        global_baseline=global_baseline,
    )


def _minimal_norm_features(split: DriftActivationSplit) -> Any:
    """Two-scalar baseline: per-step ``inc_norm.mean()`` and ``glob_norm.mean()``.

    A sanity floor — if HGB+anchor+self-baseline features (~5k dims) can't beat
    this 2-dim Logistic Regression, the rich features are noise / overfitting,
    not signal.
    """
    import torch

    incremental = (split.post - split.history).float()
    global_drift = (split.post - split.task).float()
    inc_norm = torch.linalg.vector_norm(incremental, dim=2).mean(dim=1, keepdim=True)
    glob_norm = torch.linalg.vector_norm(global_drift, dim=2).mean(dim=1, keepdim=True)
    return torch.cat([inc_norm, glob_norm], dim=1)


def train_minimal_norm_baseline(
    *,
    model_name: str,
    train: DriftActivationSplit,
    val: DriftActivationSplit,
    test: DriftActivationSplit | None = None,
    aggregation: str = "max",
    seed: int = 42,
    show_progress: bool = False,
) -> TrainedDriftDetector:
    """Minimal interpretable baseline: 2-dim features + Logistic Regression.

    Trains on per-step (inc_norm.mean, glob_norm.mean); thresholds picked via
    the same trajectory-level FPR sweep used for ``task_drift``. Returned via
    the same ``TrainedDriftDetector`` shape so the rest of the reporting
    pipeline keeps working.
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    x_train = _minimal_norm_features(train)
    x_val = _minimal_norm_features(val)
    x_train_fit, y_train_fit, _, dropped_steps = _filter_ignore_steps(x_train, train.labels)
    if show_progress and dropped_steps:
        print(f"[minimal] dropping {dropped_steps} ignore-labelled steps", file=sys.stderr, flush=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(_to_numpy(x_train_fit), _to_numpy(y_train_fit))

    val_step_scores = predict_scores(clf, x_val)
    val_records = aggregate_split_records(val, val_step_scores, aggregation=aggregation)
    val_labels = np.array([record["label"] for record in val_records])
    val_scores = np.array([record["score"] for record in val_records])
    threshold = _choose_threshold(val_labels, val_scores, target_tpr=0.95, max_fpr=None)
    val_metrics = binary_metrics(val_labels, val_scores, threshold=threshold).to_dict()
    diagnostics: dict[str, Any] = {
        "aggregation": aggregation,
        "val_threshold_tradeoffs": threshold_tradeoffs(val_labels, val_scores, [0.05, 0.10, 0.20, 0.30, 0.50]),
        "val_confusion": confusion_at_threshold(val_labels, val_scores, threshold),
    }

    test_metrics = None
    if test is not None:
        x_test = _minimal_norm_features(test)
        test_step_scores = predict_scores(clf, x_test)
        test_records = aggregate_split_records(test, test_step_scores, aggregation=aggregation)
        test_labels = np.array([record["label"] for record in test_records])
        test_scores = np.array([record["score"] for record in test_records])
        test_metrics = binary_metrics(test_labels, test_scores, threshold=threshold).to_dict()
        diagnostics["test_threshold_tradeoffs"] = threshold_tradeoffs(
            test_labels, test_scores, [0.05, 0.10, 0.20, 0.30, 0.50]
        )
        diagnostics["test_confusion"] = confusion_at_threshold(test_labels, test_scores, threshold)

    return TrainedDriftDetector(
        model_name=model_name,
        classifier=clf,
        benign_incremental_anchors=None,
        benign_global_anchors=None,
        feature_config=DriftFeatureConfig(
            feature_mode="minimal_norm",
            include_anchor_distances=False,
            include_self_baseline=False,
            aggregation=aggregation,
        ),
        threshold=threshold,
        params={"detector": "minimal_norm", "max_iter": 1000, "class_weight": "balanced"},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        diagnostics=diagnostics,
        global_baseline=None,
    )


def save_minimal_norm_baseline(detector: TrainedDriftDetector, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "minimal_norm_best.pkl"
    metrics_path = output_dir / "minimal_norm_metrics.json"
    with model_path.open("wb") as f:
        pickle.dump(detector, f)
    val_tpr_at_fpr = _summarize_tpr_at_fpr(detector.diagnostics.get("val_threshold_tradeoffs", []))
    test_tpr_at_fpr = _summarize_tpr_at_fpr(detector.diagnostics.get("test_threshold_tradeoffs", []))
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": detector.model_name,
                "detector_type": "minimal_norm",
                "feature_config": detector.feature_config.__dict__,
                "threshold": detector.threshold,
                "params": detector.params,
                "val_metrics": detector.val_metrics,
                "test_metrics": detector.test_metrics,
                "val_tpr_at_fpr": val_tpr_at_fpr,
                "test_tpr_at_fpr": test_tpr_at_fpr,
                "accuracy_note": "informational_only_due_to_class_imbalance",
                "diagnostics": detector.diagnostics,
            },
            f,
            indent=2,
        )
    return model_path, metrics_path


def train_minimal_norm_from_disk(
    *,
    model_name: str,
    drift_activation_root: Path | None = None,
    output_root: Path | None = None,
    include_test: bool = True,
    show_progress: bool = False,
) -> TrainedDriftDetector:
    cfg = load_training_cfg()
    train = load_drift_split(model_name, "train", drift_activation_root, show_progress=show_progress)
    val = load_drift_split(model_name, "val", drift_activation_root, show_progress=show_progress)
    test = (
        load_drift_split(model_name, "test", drift_activation_root, show_progress=show_progress)
        if include_test
        else None
    )
    seed = int(getattr(cfg.splits, "random_seed", 42)) if hasattr(cfg, "splits") else 42
    aggregation = str(getattr(getattr(cfg, "task_drift", object()), "aggregation", "max"))
    detector = train_minimal_norm_baseline(
        model_name=model_name,
        train=train,
        val=val,
        test=test,
        aggregation=aggregation,
        seed=seed,
        show_progress=show_progress,
    )
    output_dir = drift_detector_output_dir(model_name, output_root)
    save_minimal_norm_baseline(detector, output_dir)
    return detector


def drift_detector_output_dir(model_name: str, output_root: Path | None = None) -> Path:
    return (output_root or OUTPUTS_DIR / "detectors") / model_name


def _summarize_tpr_at_fpr(tradeoffs: list[dict[str, Any]]) -> dict[str, float]:
    """Extract a flat ``tpr_at_fpr_{p}`` mapping from threshold tradeoff rows.

    Class imbalance (≈90% positives in this dataset) makes raw accuracy
    misleading, so we surface TPR at fixed FPR levels as the primary
    reporting metric.
    """
    out: dict[str, float] = {}
    for row in tradeoffs or []:
        target = row.get("target_fpr")
        if target is None:
            continue
        key = f"tpr_at_fpr_{int(round(float(target) * 100)):03d}"
        out[key] = float(row.get("tpr", 0.0))
    return out


def save_drift_detector(detector: TrainedDriftDetector, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "task_drift_best.pkl"
    metrics_path = output_dir / "task_drift_metrics.json"
    with model_path.open("wb") as f:
        pickle.dump(detector, f)
    val_tpr_at_fpr = _summarize_tpr_at_fpr(detector.diagnostics.get("val_threshold_tradeoffs", []))
    test_tpr_at_fpr = _summarize_tpr_at_fpr(detector.diagnostics.get("test_threshold_tradeoffs", []))
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
                "val_tpr_at_fpr": val_tpr_at_fpr,
                "test_tpr_at_fpr": test_tpr_at_fpr,
                "accuracy_note": "informational_only_due_to_class_imbalance",
                "diagnostics": detector.diagnostics,
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
    threshold_target_tpr: float | None = None,
    threshold_max_fpr: float | None = None,
    sample_weight: str | None = None,
    show_progress: bool = False,
) -> TrainedDriftDetector:
    cfg = load_training_cfg()
    train = load_drift_split(model_name, "train", drift_activation_root, show_progress=show_progress)
    val = load_drift_split(model_name, "val", drift_activation_root, show_progress=show_progress)
    test = (
        load_drift_split(model_name, "test", drift_activation_root, show_progress=show_progress)
        if include_test
        else None
    )
    seed = int(getattr(cfg.splits, "random_seed", 42)) if hasattr(cfg, "splits") else 42
    feature_mode = getattr(cfg.layers, "mode", "concat")
    anchor_n = int(n_anchors or cfg.anchor.default)
    task_drift_cfg = getattr(cfg, "task_drift", object())
    include_anchor_distances = bool(getattr(task_drift_cfg, "include_anchor_distances", True))
    include_self_baseline = bool(getattr(task_drift_cfg, "include_self_baseline", True))
    aggregation = str(getattr(task_drift_cfg, "aggregation", "max"))
    threshold_target = float(threshold_target_tpr or getattr(task_drift_cfg, "threshold_target_tpr", 0.95))
    configured_max_fpr = getattr(task_drift_cfg, "threshold_max_fpr", None)
    threshold_fpr = threshold_max_fpr if threshold_max_fpr is not None else configured_max_fpr
    threshold_fpr = None if threshold_fpr in {None, "null"} else float(threshold_fpr)
    weight_mode = sample_weight if sample_weight is not None else getattr(task_drift_cfg, "sample_weight", None)
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
        include_self_baseline=include_self_baseline,
        aggregation=aggregation,
        seed=seed,
        params=params,
        threshold_target_tpr=threshold_target,
        threshold_max_fpr=threshold_fpr,
        sample_weight=weight_mode,
        show_progress=show_progress,
    )
    output_dir = drift_detector_output_dir(model_name, output_root)
    if show_progress:
        print(f"[train] saving detector to {output_dir}", file=sys.stderr, flush=True)
    save_drift_detector(detector, output_dir)
    return detector
