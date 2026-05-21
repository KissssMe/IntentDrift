#!/usr/bin/env python
"""Phase 5 — AgentDojo runtime evaluation for SecMCP task-drift detector.

Runs one or more AgentDojo task suites under three configurable defense modes:

- ``none``:    baseline pipeline (no detector).
- ``shadow``:  SecMCPTaskDriftDetector with ``raise_on_detection=False``; the
               detector observes every tool step and writes scores to the
               trace JSON without intervening. Used to calibrate thresholds
               and gather per-step score distributions.
- ``abort``:   SecMCPTaskDriftDetector with ``raise_on_detection=True``;
               AbortAgentError stops the agent before the next LLM decision.

For each (suite, attack) combo the script invokes
``benchmark_suite_with_injections`` (attacked) and
``benchmark_suite_without_injections`` (clean utility baseline). Traces land in
the standard AgentDojo logdir layout; the metrics aggregator then computes
utility / security / ASR / detector-trigger-rate / benign-false-abort-rate.

Pipeline construction is done manually rather than via
``AgentPipeline.from_config`` because that helper's ``defense`` enum does not
include SecMCP.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

from dotenv import load_dotenv

from secmcp.config import OUTPUTS_DIR, load_eval_cfg


def _build_llm(provider: str, model: str, model_id: str | None, tool_delimiter: str):
    """Mirror ``agent_pipeline.get_llm`` without requiring every SDK to be
    importable at module-import time."""
    import os

    if provider == "openai":
        import openai
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        return OpenAILLM(openai.OpenAI(), model)
    if provider == "anthropic":
        import anthropic
        from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM

        return AnthropicLLM(anthropic.Anthropic(), model)
    if provider == "local":
        import openai
        from agentdojo.agent_pipeline.llms.local_llm import LocalLLM

        port = os.getenv("LOCAL_LLM_PORT", 8000)
        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        return LocalLLM(client, model_id or model, tool_delimiter=tool_delimiter)
    raise ValueError(f"unsupported provider {provider!r}")


def _build_pipeline(
    *,
    llm_provider: str,
    llm_model: str,
    llm_model_id: str | None,
    tool_delimiter: str,
    system_message_name: str | None,
    defense_mode: str,
    detector_path: str | None,
    detector_model_name: str | None,
    detector_threshold: float | None,
    raise_on_detection: bool,
    pipeline_name_tag: str,
):
    """Manual pipeline assembly — equivalent to ``AgentPipeline.from_config``
    but supports the ``secmcp`` defense by inserting our pipeline element into
    the tool-execution loop."""

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

    llm = _build_llm(llm_provider, llm_model, llm_model_id, tool_delimiter)

    system_message = load_system_message(system_message_name)
    system_component = SystemMessage(system_message)
    init_query_component = InitQuery()

    if defense_mode == "none":
        loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    else:
        from secmcp.integrations.agentdojo_drift_detector import SecMCPTaskDriftDetector

        if detector_path is None or detector_model_name is None:
            raise ValueError("--detector-path and --model are required for defense != none")
        detector = SecMCPTaskDriftDetector(
            detector_path=detector_path,
            model_name=detector_model_name,
            threshold=detector_threshold,
            raise_on_detection=raise_on_detection,
        )
        loop = ToolsExecutionLoop([ToolsExecutor(), detector, llm])

    pipeline = AgentPipeline([system_component, init_query_component, llm, loop])
    pipeline.name = f"{llm_model}-{pipeline_name_tag}"
    return pipeline


def _sample_tasks(all_ids: list[str], fraction: float, seed: int) -> list[str]:
    if fraction >= 1.0:
        return list(all_ids)
    rng = random.Random(seed)
    n = max(1, int(round(len(all_ids) * fraction)))
    return sorted(rng.sample(list(all_ids), n))


def _sample_tag(args: argparse.Namespace, user_task_frac: float, injection_task_frac: float) -> str:
    user_part = f"u{args.max_user_tasks}" if args.max_user_tasks is not None else f"uf{user_task_frac:g}"
    inj_part = f"i{args.max_injection_tasks}" if args.max_injection_tasks is not None else f"if{injection_task_frac:g}"
    return f"{user_part}_{inj_part}_seed{args.seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistral_7b_v03", help="SecMCP detector backbone (matches configs/models.yaml).")
    parser.add_argument(
        "--detector-path",
        default=None,
        help="Pickled task-drift detector. Defaults to outputs/detectors/{model}/task_drift_best.pkl.",
    )
    parser.add_argument("--suite", "-s", dest="suites", nargs="+", default=None, help="Override eval.yaml agentdojo.suites.")
    parser.add_argument("--attack", default="important_instructions", help="AgentDojo attack name to run.")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "anthropic", "local"])
    parser.add_argument("--llm-model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--llm-model-id", default=None, help="Model id for local/vllm provider.")
    parser.add_argument("--tool-delimiter", default="tool")
    parser.add_argument("--system-message-name", default=None)
    parser.add_argument("--defense", choices=["none", "shadow", "abort"], default="abort")
    parser.add_argument("--threshold", type=float, default=None, help="Override detector threshold.")
    parser.add_argument("--user-task-frac", type=float, default=None)
    parser.add_argument("--injection-task-frac", type=float, default=None)
    parser.add_argument("--max-user-tasks", type=int, default=None, help="Hard cap on user tasks per suite (post-sampling).")
    parser.add_argument("--max-injection-tasks", type=int, default=None)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", "-f", action="store_true")
    parser.add_argument("--skip-clean", action="store_true", help="Skip clean (no-attack) utility runs.")
    parser.add_argument("--skip-attacked", action="store_true", help="Skip attacked runs.")
    parser.add_argument("--logdir", default=None, help="AgentDojo trace logdir. Default outputs/eval/agentdojo/runs.")
    parser.add_argument("--output-root", default=None, help="Where metrics.json/step_scores.jsonl go. Default outputs/eval/agentdojo.")
    parser.add_argument("--module-to-load", action="append", default=[], help="Importable modules to load (for custom suites/attacks).")
    parser.add_argument("--dry-run", action="store_true", help="List the planned suites/tasks and exit before building the pipeline or running benchmarks.")
    args = parser.parse_args()

    load_dotenv()

    import importlib
    for mod in args.module_to_load:
        importlib.import_module(mod)

    eval_cfg = load_eval_cfg()
    dojo_cfg = eval_cfg.agentdojo
    suites = args.suites or list(getattr(dojo_cfg, "suites"))
    user_task_frac = args.user_task_frac if args.user_task_frac is not None else float(dojo_cfg.user_task_sample_frac)
    injection_task_frac = args.injection_task_frac if args.injection_task_frac is not None else float(dojo_cfg.injection_task_sample_frac)
    configured_threshold = getattr(dojo_cfg, "detection_threshold", None)
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif configured_threshold in {None, "null"}:
        threshold = None
    else:
        threshold = float(configured_threshold)
    threshold_tag = "artifact" if threshold is None else f"{threshold:g}"

    if args.detector_path is None:
        args.detector_path = str(OUTPUTS_DIR / "detectors" / args.model / "task_drift_best.pkl")

    output_root = Path(args.output_root) if args.output_root else (OUTPUTS_DIR / "eval" / "agentdojo")
    logdir = Path(args.logdir) if args.logdir else (output_root / "runs")
    logdir.mkdir(parents=True, exist_ok=True)

    sample_tag = _sample_tag(args, user_task_frac, injection_task_frac)
    defense_to_kwargs = {
        "none": {"defense_mode": "none", "raise_on_detection": False, "pipeline_name_tag": f"no_defense_{sample_tag}"},
        "shadow": {
            "defense_mode": "secmcp",
            "raise_on_detection": False,
            "pipeline_name_tag": f"secmcp_shadow_{args.model}_thr{threshold_tag}_{sample_tag}",
        },
        "abort": {
            "defense_mode": "secmcp",
            "raise_on_detection": True,
            "pipeline_name_tag": f"secmcp_abort_{args.model}_thr{threshold_tag}_{sample_tag}",
        },
    }
    defense_kwargs = defense_to_kwargs[args.defense]

    if args.dry_run:
        from agentdojo.task_suite.load_suites import get_suite

        for suite_name in suites:
            suite = get_suite(args.benchmark_version, suite_name)
            all_user_ids = sorted(suite.user_tasks.keys())
            all_inj_ids = sorted(suite.injection_tasks.keys())
            user_ids = _sample_tasks(all_user_ids, user_task_frac, args.seed + hash(suite_name) % (2**31))
            inj_ids = _sample_tasks(all_inj_ids, injection_task_frac, args.seed + hash(suite_name + "inj") % (2**31))
            if args.max_user_tasks is not None:
                user_ids = user_ids[: args.max_user_tasks]
            if args.max_injection_tasks is not None:
                inj_ids = inj_ids[: args.max_injection_tasks]
            print(f"[dry-run] suite={suite_name} user_tasks({len(user_ids)})={user_ids} injection_tasks({len(inj_ids)})={inj_ids}")
        print(f"[dry-run] defense={args.defense} detector_path={args.detector_path} threshold={threshold_tag}")
        return

    pipeline = _build_pipeline(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_model_id=args.llm_model_id,
        tool_delimiter=args.tool_delimiter,
        system_message_name=args.system_message_name,
        detector_path=args.detector_path if defense_kwargs["defense_mode"] != "none" else None,
        detector_model_name=args.model if defense_kwargs["defense_mode"] != "none" else None,
        detector_threshold=threshold if defense_kwargs["defense_mode"] != "none" else None,
        **defense_kwargs,
    )
    pipeline_name = pipeline.name
    print(f"[secmcp eval] pipeline.name = {pipeline_name}")
    print(f"[secmcp eval] logdir = {logdir}")
    print(f"[secmcp eval] suites = {suites}")

    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite

    summary: dict[str, dict] = {}

    for suite_name in suites:
        suite = get_suite(args.benchmark_version, suite_name)
        all_user_ids = sorted(suite.user_tasks.keys())
        all_inj_ids = sorted(suite.injection_tasks.keys())
        user_ids = _sample_tasks(all_user_ids, user_task_frac, args.seed + hash(suite_name) % (2**31))
        inj_ids = _sample_tasks(all_inj_ids, injection_task_frac, args.seed + hash(suite_name + "inj") % (2**31))
        if args.max_user_tasks is not None:
            user_ids = user_ids[: args.max_user_tasks]
        if args.max_injection_tasks is not None:
            inj_ids = inj_ids[: args.max_injection_tasks]
        print(f"[secmcp eval] suite={suite_name} user_tasks={len(user_ids)}/{len(all_user_ids)} injection_tasks={len(inj_ids)}/{len(all_inj_ids)}")

        with OutputLogger(str(logdir), live=None):
            if not args.skip_clean:
                clean_results = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    user_tasks=user_ids,
                    logdir=logdir,
                    force_rerun=args.force_rerun,
                    benchmark_version=args.benchmark_version,
                )
            else:
                clean_results = None

            if not args.skip_attacked:
                try:
                    attacker = load_attack(args.attack, suite, pipeline)
                except KeyError as e:
                    raise SystemExit(f"unknown attack {args.attack!r}: {e}")
                attacked_results = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attacker,
                    user_tasks=user_ids,
                    injection_tasks=inj_ids,
                    logdir=logdir,
                    force_rerun=args.force_rerun,
                    benchmark_version=args.benchmark_version,
                )
            else:
                attacked_results = None

        summary[suite_name] = {
            "user_tasks": user_ids,
            "injection_tasks": inj_ids,
            "clean_done": clean_results is not None,
            "attacked_done": attacked_results is not None,
        }

    output_dir = output_root / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    from secmcp.integrations.agentdojo_eval import write_metrics

    metrics = write_metrics(output_dir, pipeline_name, suites, logdir)
    print(f"[secmcp eval] wrote {output_dir / 'metrics.json'}")
    overall = metrics.get("overall", {})
    print(
        "[secmcp eval] overall: "
        f"utility_clean={overall.get('utility_clean')} "
        f"utility_attacked={overall.get('utility_attacked')} "
        f"security_attacked={overall.get('security_attacked')} "
        f"ASR={overall.get('attack_success_rate')} "
        f"trigger_rate_clean={overall.get('detector_trigger_rate_clean')} "
        f"trigger_rate_attacked={overall.get('detector_trigger_rate_attacked')} "
        f"benign_false_abort={overall.get('benign_false_abort_rate')}"
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
