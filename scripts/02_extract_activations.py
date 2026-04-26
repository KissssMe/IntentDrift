#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from secmcp.activations.extract import extract_split_activations
from secmcp.activations.drift import extract_split_drift_activations
from secmcp.models.loader import load_shared_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract last-token activations for a SecMCP split.")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed shards.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--log-every", type=int, default=1, help="Print a heartbeat before every N samples.")
    parser.add_argument(
        "--mode",
        default="task_drift",
        choices=["whole_context", "task_drift"],
        help="Activation extraction target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[load] loading model={args.model}", file=sys.stderr, flush=True)
    loaded = load_shared_model(args.model)
    print(
        f"[load] loaded model={args.model} layers={list(loaded.cfg.layers)} "
        f"hidden_dim={loaded.cfg.hidden_dim}",
        file=sys.stderr,
        flush=True,
    )
    if args.mode == "task_drift":
        summary = extract_split_drift_activations(
            model_name=args.model,
            split=args.split,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            cfg=loaded.cfg,
            layers=list(loaded.cfg.layers),
            splits_dir=args.splits_dir,
            output_root=args.output_root,
            shard_size=args.shard_size,
            resume=not args.no_resume,
            show_progress=not args.no_progress,
        )
    else:
        summary = extract_split_activations(
            model_name=args.model,
            split=args.split,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            cfg=loaded.cfg,
            layers=list(loaded.cfg.layers),
            splits_dir=args.splits_dir,
            output_root=args.output_root,
            shard_size=args.shard_size,
            resume=not args.no_resume,
            show_progress=not args.no_progress,
            log_every=args.log_every,
        )
    print(
        f"model={summary.model_name} split={summary.split} total={summary.total_samples} "
        f"skipped={getattr(summary, 'skipped_samples', getattr(summary, 'skipped_steps', 0))} "
        f"extracted={getattr(summary, 'extracted_samples', getattr(summary, 'extracted_steps', 0))} "
        f"shards_written={summary.shards_written} output_dir={summary.output_dir}"
    )


if __name__ == "__main__":
    main()
