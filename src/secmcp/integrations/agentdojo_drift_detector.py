from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from secmcp.activations.drift import _task_prefix
from secmcp.activations.drift_dataset import DriftActivationSplit
from secmcp.data.schema import normalize_messages_for_chat
from secmcp.detectors.drift import aggregate_step_scores, drift_feature_matrix
from secmcp.models.hooks import last_token_hidden_states
from secmcp.models.loader import load_shared_model

try:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
except Exception:  # pragma: no cover - AgentDojo is optional outside integration runs.
    BasePipelineElement = object


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
        self.loaded = load_shared_model(model_name)
        self.threshold = float(threshold if threshold is not None else self.detector.threshold)
        self.raise_on_detection = raise_on_detection

    def _score(self, messages: Sequence[dict[str, Any]]) -> float:
        import torch

        normalized = normalize_messages_for_chat([dict(m) for m in messages])
        if not normalized or normalized[-1].get("role") != "tool":
            return 0.0

        task_prefix = _task_prefix(normalized)
        history_prefix = normalized[:-1]
        post_tool_prefix = normalized
        layers = list(self.loaded.cfg.layers)
        task = last_token_hidden_states(self.loaded.model, self.loaded.tokenizer, task_prefix, layers, self.loaded.cfg)
        history = last_token_hidden_states(
            self.loaded.model, self.loaded.tokenizer, history_prefix, layers, self.loaded.cfg
        )
        post = last_token_hidden_states(self.loaded.model, self.loaded.tokenizer, post_tool_prefix, layers, self.loaded.cfg)

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
        )
        if hasattr(self.detector.classifier, "predict_proba"):
            return float(self.detector.classifier.predict_proba(features.detach().cpu().numpy())[0, 1])
        score = self.detector.classifier.decision_function(features.detach().cpu().numpy())[0]
        return float(1.0 / (1.0 + torch.exp(torch.tensor(-score))).item())

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

        score = self._score(messages)
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
        }
        if self.raise_on_detection and is_detection:
            raise AbortAgentError(
                "aborting execution because SecMCP task drift was detected "
                f"(trajectory_score: {trajectory_score:.4f}, last_score: {score:.4f})",
                list(messages),
                env,
            )
        return query, runtime, env, messages, extra_args
