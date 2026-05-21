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
    parser.add_argument(
        "--counterfactual-pairs",
        action="store_true",
        help=(
            "Replace AgentTraj-L benign trajectories with paired (clean, attacked) groups. "
            "Each injectable benign trajectory is emitted as a clean copy plus N attacked clones "
            "sharing one split_group, so 'same task & history, only the tool differs' is preserved on one side of the split."
        ),
    )
    parser.add_argument(
        "--counterfactual-attacks-per-benign",
        type=int,
        default=1,
        help="Number of attacked clones per benign trajectory when --counterfactual-pairs is set.",
    )
    parser.add_argument(
        "--keep-unlocatable-positives",
        action="store_true",
        help=(
            "Retain positive samples that have no injection locator (no malicious_tool_message_indices "
            "and no injection_text/fragments). By default these are dropped because their step-level "
            "supervision is contradictory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_cfg()
    samples = load_all_sources(
        agenttraj_frac=cfg.sources.agenttraj_l.sample_frac,
        seed=cfg.splits.random_seed,
        max_per_source=args.max_per_source,
        agentdojo_pipeline=args.agentdojo_pipeline,
        counterfactual_pairs=args.counterfactual_pairs,
        counterfactual_attacks_per_benign=args.counterfactual_attacks_per_benign,
        drop_unlocatable=not args.keep_unlocatable_positives,
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
