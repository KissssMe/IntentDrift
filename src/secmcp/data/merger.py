from __future__ import annotations

from pathlib import Path

from secmcp.config import DATA_DIR
from secmcp.data.loaders import agentdojo, agentharm, agenttraj_l, asb, injecagent
from secmcp.data.schema import UnifiedSample


def load_all_sources(
    data_dir: Path = DATA_DIR,
    agenttraj_frac: float = 0.25,
    seed: int = 42,
    max_per_source: int | None = None,
    agentdojo_pipeline: str | None = None,
) -> list[UnifiedSample]:
    samples: list[UnifiedSample] = []
    samples.extend(injecagent.load(data_dir / "InjecAgent", max_samples=max_per_source))
    samples.extend(agentharm.load(data_dir / "AgentHarm", max_samples=max_per_source))
    samples.extend(asb.load(data_dir / "ASB", max_samples=max_per_source))
    samples.extend(
        agenttraj_l.load(
            data_dir / "AgentTraj-L",
            sample_frac=agenttraj_frac,
            seed=seed,
            max_samples=max_per_source,
        )
    )
    samples.extend(agentdojo.load(data_dir / "AgentDojo", max_samples=max_per_source, pipeline=agentdojo_pipeline))
    return samples


def source_label_counts(samples: list[UnifiedSample]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for sample in samples:
        key = (sample.source, sample.label)
        counts[key] = counts.get(key, 0) + 1
    return counts
