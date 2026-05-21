from __future__ import annotations

import json
from pathlib import Path

from secmcp.integrations.agentdojo_eval import summarize_traces, write_metrics


def _trace(**overrides):
    base = {
        "suite_name": "workspace",
        "user_task_id": "user_task_0",
        "injection_task_id": None,
        "attack_type": None,
        "utility": True,
        "security": True,
        "error": None,
    }
    base.update(overrides)
    return base


def test_summarize_traces_clean_and_attacked():
    traces = [
        _trace(utility=True, attack_type=None),
        _trace(utility=False, attack_type=None),
        _trace(
            utility=False,
            security=False,
            attack_type="important_instructions",
            injection_task_id="injection_task_0",
        ),
        _trace(
            utility=False,
            security=True,
            attack_type="important_instructions",
            injection_task_id="injection_task_1",
            secmcp_task_drift_triggered=True,
            secmcp_task_drift_aborted=True,
            secmcp_task_drift_trigger_step=2,
            error="aborting execution because SecMCP task drift was detected",
        ),
    ]
    metrics = summarize_traces(traces)
    overall = metrics["overall"]
    assert overall["count"] == 4
    assert overall["clean_count"] == 2
    assert overall["regular_clean_count"] == 2
    assert overall["attacked_count"] == 2
    assert overall["utility_clean"] == 0.5
    assert overall["utility_regular_clean"] == 0.5
    assert overall["security_attacked"] == 0.5
    assert overall["attack_success_rate"] == 0.5
    assert overall["agentdojo_injection_task_success_rate"] == 0.5
    assert overall["detector_trigger_rate_attacked"] == 1.0  # only one had key set
    assert overall["benign_false_abort_rate"] == 0.0
    assert overall["mean_trigger_step"] == 2


def test_summarize_traces_separates_injection_tasks_used_as_clean():
    traces = [
        _trace(user_task_id="user_task_0", secmcp_task_drift_triggered=False),
        _trace(user_task_id="injection_task_0", utility=False, secmcp_task_drift_triggered=True),
    ]
    overall = summarize_traces(traces)["overall"]
    assert overall["clean_count"] == 2
    assert overall["regular_clean_count"] == 1
    assert overall["detector_trigger_rate_clean"] == 0.5
    assert overall["detector_trigger_rate_regular_clean"] == 0.0


def test_write_metrics_emits_files(tmp_path: Path):
    pipeline = "pipe"
    logdir = tmp_path / "runs"
    suite_dir = logdir / pipeline / "workspace" / "u0" / "none"
    suite_dir.mkdir(parents=True)
    (suite_dir / "none.json").write_text(json.dumps(_trace(utility=True)))

    out_dir = tmp_path / "out"
    metrics = write_metrics(out_dir, pipeline, ["workspace"], logdir)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "step_scores.jsonl").exists()
    assert metrics["num_traces"] == 1
