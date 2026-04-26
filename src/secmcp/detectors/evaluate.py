from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Metrics:
    auroc: float
    auprc: float
    accuracy: float
    fpr_at_95tpr: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fpr_at_tpr(labels, scores, target_tpr: float = 0.95) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return 1.0
    return float(fpr[valid[0]])


def binary_metrics(labels, scores, threshold: float = 0.5) -> Metrics:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    preds = (scores >= threshold).astype(int)
    return Metrics(
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        accuracy=float(accuracy_score(labels, preds)),
        fpr_at_95tpr=fpr_at_tpr(labels, scores, 0.95),
    )


def predict_scores(clf, features):
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(features)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(features)
        import numpy as np

        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Classifier must expose predict_proba or decision_function")
