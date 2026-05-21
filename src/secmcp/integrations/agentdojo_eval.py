"""Aggregate AgentDojo per-task trace JSONs into Phase-5 metrics.

AgentDojo writes one JSON per (suite, user_task, attack, injection_task) under
``{logdir}/{pipeline_name}/{suite}/{user_task}/{attack}/{injection}.json``.
Each file contains the chat messages, utility, security, and any extra
contextargs set during the run. The SecMCP task-drift element persists its
trajectory scores via ``Logger.set_contextarg``, so they appear in the same
files as ``secmcp_task_drift_*`` keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def collect_traces(logdir: Path, pipeline_name: str, suites: list[str]) -> list[dict[str, Any]]:
    """Walk ``logdir`` and return one record per trace JSON found."""
    root = logdir / pipeline_name
    traces: list[dict[str, Any]] = []
    for suite in suites:
        suite_dir = root / suite
        if not suite_dir.exists():
            continue
        for trace_path in suite_dir.rglob("*.json"):
            data = _load_trace(trace_path)
            if data is None:
                continue
            traces.append(data)
    return traces


def _is_detector_abort(trace: dict[str, Any]) -> bool:
    if trace.get("secmcp_task_drift_aborted"):
        return True
    error = trace.get("error") or ""
    return "SecMCP task drift" in str(error)


def _is_regular_clean_task(trace: dict[str, Any]) -> bool:
    user_task_id = str(trace.get("user_task_id") or "")
    return not user_task_id.startswith("injection_task_")


def _attack_label(trace: dict[str, Any]) -> str:
    a = trace.get("attack_type")
    return a if a else "none"


def _group_by(traces: list[dict[str, Any]], key_fn) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for t in traces:
        out.setdefault(key_fn(t), []).append(t)
    return out


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Phase-5 metrics over a collected trace list."""

    def _aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"count": 0}

        clean = [t for t in group if not t.get("attack_type") or t.get("attack_type") == "none"]
        regular_clean = [t for t in clean if _is_regular_clean_task(t)]
        attacked = [t for t in group if t.get("attack_type") and t.get("attack_type") != "none"]

        def _frac(rows: list[dict[str, Any]], key: str) -> float | None:
            vals = [bool(r.get(key)) for r in rows if r.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None

        triggered = [t for t in group if t.get("secmcp_task_drift_triggered")]
        detector_seen = [t for t in group if "secmcp_task_drift_triggered" in t]
        attacked_task_success = _frac(attacked, "security")
        defense_success = (1 - attacked_task_success) if attacked_task_success is not None else None

        steps = [t.get("secmcp_task_drift_trigger_step") for t in triggered if t.get("secmcp_task_drift_trigger_step") is not None]

        return {
            "count": len(group),
            "clean_count": len(clean),
            "regular_clean_count": len(regular_clean),
            "attacked_count": len(attacked),
            "utility_clean": _frac(clean, "utility"),
            "utility_regular_clean": _frac(regular_clean, "utility"),
            "utility_attacked": _frac(attacked, "utility"),
            # AgentDojo's injection-task "security" check returns true when
            # the injected objective was achieved. Report both the raw value
            # and the defense-oriented complement to avoid reading it backward.
            "agentdojo_injection_task_success_rate": attacked_task_success,
            "attack_success_rate": attacked_task_success,
            "security_attacked": defense_success,
            "detector_trigger_rate_clean": _frac(clean, "secmcp_task_drift_triggered"),
            "detector_trigger_rate_regular_clean": _frac(regular_clean, "secmcp_task_drift_triggered"),
            "detector_trigger_rate_attacked": _frac(attacked, "secmcp_task_drift_triggered"),
            "benign_false_abort_rate": (
                sum(1 for t in clean if _is_detector_abort(t)) / len(clean) if clean else None
            ),
            "regular_clean_false_abort_rate": (
                sum(1 for t in regular_clean if _is_detector_abort(t)) / len(regular_clean) if regular_clean else None
            ),
            "detector_evaluated_count": len(detector_seen),
            "mean_trigger_step": (sum(steps) / len(steps)) if steps else None,
        }

    by_suite = _group_by(traces, lambda t: t.get("suite_name", "unknown"))
    by_attack = _group_by(traces, _attack_label)
    by_suite_attack = _group_by(traces, lambda t: f"{t.get('suite_name','unknown')}/{_attack_label(t)}")
    by_injection_task = _group_by(traces, lambda t: t.get("injection_task_id") or "none")
    by_user_task = _group_by(
        traces, lambda t: f"{t.get('suite_name','unknown')}/{t.get('user_task_id') or 'none'}"
    )

    def _grouped(d: dict[Any, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        return {str(k): _aggregate(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}

    return {
        "overall": _aggregate(traces),
        "by_suite": _grouped(by_suite),
        "by_attack_type": _grouped(by_attack),
        "by_suite_attack": _grouped(by_suite_attack),
        "by_injection_task": _grouped(by_injection_task),
        "by_user_task": _grouped(by_user_task),
    }


def write_metrics(
    out_dir: Path,
    pipeline_name: str,
    suites: list[str],
    logdir: Path,
) -> dict[str, Any]:
    traces = collect_traces(logdir, pipeline_name, suites)
    metrics = summarize_traces(traces)
    metrics["pipeline_name"] = pipeline_name
    metrics["suites"] = suites
    metrics["logdir"] = str(logdir)
    metrics["num_traces"] = len(traces)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "step_scores.jsonl").open("w") as f:
        for trace in traces:
            if "secmcp_task_drift_scores" not in trace:
                continue
            f.write(
                json.dumps(
                    {
                        "suite_name": trace.get("suite_name"),
                        "user_task_id": trace.get("user_task_id"),
                        "injection_task_id": trace.get("injection_task_id"),
                        "attack_type": trace.get("attack_type"),
                        "utility": trace.get("utility"),
                        "security": trace.get("security"),
                        "scores": trace.get("secmcp_task_drift_scores"),
                        "trajectory_score": trace.get("secmcp_task_drift_trajectory_score"),
                        "aggregation": trace.get("secmcp_task_drift_aggregation"),
                        "threshold": trace.get("secmcp_task_drift_threshold"),
                        "triggered": trace.get("secmcp_task_drift_triggered"),
                        "aborted": trace.get("secmcp_task_drift_aborted"),
                        "trigger_step": trace.get("secmcp_task_drift_trigger_step"),
                    }
                )
                + "\n"
            )

    return metrics
