from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from secmcp.activations.drift import _task_prefix, _task_text
from secmcp.activations.drift_dataset import DriftActivationSplit
from secmcp.data.schema import normalize_messages_for_chat
from secmcp.detectors.drift import (
    TrajectoryPriorNorms,
    aggregate_step_scores,
    drift_feature_matrix,
    update_prior_norms_from_split,
)
from secmcp.models.hooks import task_anchored_hidden_states
from secmcp.models.loader import load_shared_model

try:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.logging import Logger as _AgentDojoLogger
except Exception:  # pragma: no cover - AgentDojo is optional outside integration runs.
    BasePipelineElement = object
    _AgentDojoLogger = None


class SecMCPTaskDriftDetector(BasePipelineElement):
    """AgentDojo pipeline element for task-drift detection.

    It is intended to be inserted as:
    ToolsExecutionLoop([ToolsExecutor(), SecMCPTaskDriftDetector(...), llm])
    so detection runs after tool results are appended and before the next LLM
    decision.
    """

    name = "secmcp_task_drift"

    def __init__(
        self,
        detector_path: str | Path,
        model_name: str,
        threshold: float | None = None,
        raise_on_detection: bool = False,
    ) -> None:
        with Path(detector_path).open("rb") as f:
            self.detector = pickle.load(f)
        if not hasattr(getattr(self.detector, "feature_config", None), "include_self_baseline"):
            raise ValueError(
                "Loaded task-drift detector artifact predates per-trajectory self-baseline support; "
                "retrain the detector after re-extracting task-anchor drift activations."
            )
        self.loaded = load_shared_model(model_name)
        self.threshold = float(threshold if threshold is not None else self.detector.threshold)
        self.raise_on_detection = raise_on_detection

    def _score(
        self,
        messages: Sequence[dict[str, Any]],
        prior_state: TrajectoryPriorNorms | None = None,
    ) -> tuple[float, DriftActivationSplit | None]:
        import torch

        normalized = normalize_messages_for_chat([dict(m) for m in messages])
        if not normalized or normalized[-1].get("role") != "tool":
            return 0.0, None

        task_prefix = _task_prefix(normalized)
        history_prefix = normalized[:-1]
        post_tool_prefix = normalized
        task_text = _task_text(normalized)
        layers = list(self.loaded.cfg.layers)
        task = task_anchored_hidden_states(
            self.loaded.model, self.loaded.tokenizer, task_prefix, task_text, layers, self.loaded.cfg
        )
        history = task_anchored_hidden_states(
            self.loaded.model, self.loaded.tokenizer, history_prefix, task_text, layers, self.loaded.cfg
        )
        post = task_anchored_hidden_states(
            self.loaded.model, self.loaded.tokenizer, post_tool_prefix, task_text, layers, self.loaded.cfg
        )

        split = DriftActivationSplit(
            task=task.unsqueeze(0),
            history=history.unsqueeze(0),
            post=post.unsqueeze(0),
            labels=torch.zeros(1, dtype=torch.long),
            metas=[{}],
        )
        features = drift_feature_matrix(
            split,
            benign_incremental_anchors=self.detector.benign_incremental_anchors,
            benign_global_anchors=self.detector.benign_global_anchors,
            feature_mode=self.detector.feature_config.feature_mode,
            include_anchor_distances=self.detector.feature_config.include_anchor_distances,
            include_self_baseline=self.detector.feature_config.include_self_baseline,
            prior_state=prior_state,
            global_baseline=getattr(self.detector, "global_baseline", None),
        )
        if hasattr(self.detector.classifier, "predict_proba"):
            score = float(self.detector.classifier.predict_proba(features.detach().cpu().numpy())[0, 1])
        else:
            raw = self.detector.classifier.decision_function(features.detach().cpu().numpy())[0]
            score = float(1.0 / (1.0 + torch.exp(torch.tensor(-raw))).item())
        return score, split

    def query(
        self,
        query: str,
        runtime,
        env=None,
        messages: Sequence[dict[str, Any]] = (),
        extra_args: dict | None = None,
    ):
        from agentdojo.agent_pipeline.errors import AbortAgentError

        if extra_args is None:
            extra_args = {}
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        prior_state = extra_args.get("secmcp_task_drift_prior_state") or TrajectoryPriorNorms()
        score, split = self._score(messages, prior_state=prior_state)
        if split is not None and self.detector.feature_config.include_self_baseline:
            update_prior_norms_from_split(prior_state, split)
        scores = list(extra_args.get("secmcp_task_drift_scores", []))
        scores.append(score)
        aggregation = self.detector.feature_config.aggregation
        trajectory_score = aggregate_step_scores(scores, aggregation=aggregation)
        is_detection = trajectory_score >= self.threshold
        extra_args = {
            **extra_args,
            "secmcp_task_drift_scores": scores,
            "secmcp_task_drift_last_score": score,
            "secmcp_task_drift_trajectory_score": trajectory_score,
            "secmcp_task_drift_aggregation": aggregation,
            "secmcp_task_drift_prior_state": prior_state,
        }
        if _AgentDojoLogger is not None:
            try:
                logger = _AgentDojoLogger.get()
                if hasattr(logger, "set_contextarg"):
                    logger.set_contextarg("secmcp_task_drift_scores", scores)
                    logger.set_contextarg("secmcp_task_drift_trajectory_score", trajectory_score)
                    logger.set_contextarg("secmcp_task_drift_aggregation", aggregation)
                    logger.set_contextarg("secmcp_task_drift_threshold", self.threshold)
                    logger.set_contextarg("secmcp_task_drift_triggered", bool(is_detection))
                    logger.set_contextarg("secmcp_task_drift_aborted", bool(self.raise_on_detection and is_detection))
                    if is_detection:
                        logger.set_contextarg("secmcp_task_drift_trigger_step", len(scores) - 1)
            except Exception:
                pass
        if self.raise_on_detection and is_detection:
            raise AbortAgentError(
                "aborting execution because SecMCP task drift was detected "
                f"(trajectory_score: {trajectory_score:.4f}, last_score: {score:.4f})",
                list(messages),
                env,
            )
        return query, runtime, env, messages, extra_args
