#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from secmcp.config import OUTPUTS_DIR, load_data_cfg
from secmcp.data.io import write_samples_jsonl
from secmcp.data.merger import load_all_sources, source_label_counts
from secmcp.data.split import make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unify SecMCP training datasets into jsonl splits.")
    parser.add_argument("--max-per-source", type=int, default=None, help="Small debug cap per source.")
    parser.add_argument("--agentdojo-pipeline", default=None, help="Optional AgentDojo runs subdir, e.g. command-r.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "splits", help="Directory for split jsonl files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_cfg()
    samples = load_all_sources(
        agenttraj_frac=cfg.sources.agenttraj_l.sample_frac,
        seed=cfg.splits.random_seed,
        max_per_source=args.max_per_source,
        agentdojo_pipeline=args.agentdojo_pipeline,
    )
    splits = make_splits(
        samples,
        agentdojo_train_ratio=cfg.sources.agentdojo.train_frac,
        train_val_ratio=cfg.splits.train_val_ratio,
        seed=cfg.splits.random_seed,
    )
    for name, split_samples in splits.items():
        write_samples_jsonl(args.output_dir / f"{name}.jsonl", split_samples)

    print(f"loaded={len(samples)} counts={source_label_counts(samples)}")
    print({name: len(rows) for name, rows in splits.items()})


if __name__ == "__main__":
    main()
