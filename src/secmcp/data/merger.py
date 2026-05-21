from __future__ import annotations

from pathlib import Path

from secmcp.config import DATA_DIR
from secmcp.data.counterfactual import build_counterfactual_pairs, collect_attack_payloads
from secmcp.data.loaders import agentdojo, agentharm, agenttraj_l, asb, injecagent
from secmcp.data.schema import UnifiedSample


def _has_step_locator(sample: UnifiedSample) -> bool:
    meta = sample.metadata or {}
    return bool(
        meta.get("malicious_tool_message_indices")
        or meta.get("injection_fragments")
        or meta.get("injection_text")
    )


def drop_unlocatable_positives(samples: list[UnifiedSample]) -> tuple[list[UnifiedSample], dict[str, int]]:
    """Remove positive samples for which no tool-step locator is available.

    Without a locator we cannot place the hijack point inside the trajectory;
    falling back to "all tool steps are positive" creates contradictory step
    supervision (pre-injection steps look identical to benign steps yet carry
    label 1). Counting per source to surface coverage regressions.
    """
    kept: list[UnifiedSample] = []
    dropped: dict[str, int] = {}
    for sample in samples:
        if sample.label == 1 and not _has_step_locator(sample):
            dropped[sample.source] = dropped.get(sample.source, 0) + 1
            continue
        kept.append(sample)
    return kept, dropped


def load_all_sources(
    data_dir: Path = DATA_DIR,
    agenttraj_frac: float = 0.25,
    seed: int = 42,
    max_per_source: int | None = None,
    agentdojo_pipeline: str | None = None,
    counterfactual_pairs: bool = False,
    counterfactual_attacks_per_benign: int = 1,
    drop_unlocatable: bool = True,
) -> list[UnifiedSample]:
    samples: list[UnifiedSample] = []
    samples.extend(injecagent.load(data_dir / "InjecAgent", max_samples=max_per_source))
    samples.extend(agentharm.load(data_dir / "AgentHarm", max_samples=max_per_source))
    samples.extend(asb.load(data_dir / "ASB", max_samples=max_per_source))
    benign_traj = agenttraj_l.load(
        data_dir / "AgentTraj-L",
        sample_frac=agenttraj_frac,
        seed=seed,
        max_samples=max_per_source,
    )
    if counterfactual_pairs:
        payloads = collect_attack_payloads(data_dir)
        benign_traj = build_counterfactual_pairs(
            benign_traj,
            payloads,
            seed=seed,
            n_attacked_per_benign=counterfactual_attacks_per_benign,
        )
    samples.extend(benign_traj)
    samples.extend(agentdojo.load(data_dir / "AgentDojo", max_samples=max_per_source, pipeline=agentdojo_pipeline))
    if drop_unlocatable:
        samples, dropped = drop_unlocatable_positives(samples)
        if dropped:
            print(f"[merger] dropped unlocatable positives by source: {dropped}")
    return samples


def source_label_counts(samples: list[UnifiedSample]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for sample in samples:
        key = (sample.source, sample.label)
        counts[key] = counts.get(key, 0) + 1
    return counts
