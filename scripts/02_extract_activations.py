#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from secmcp.activations.extract import extract_split_activations
from secmcp.activations.drift import extract_split_drift_activations
from secmcp.models.loader import load_shared_model


def distributed_rank() -> int:
    import os

    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def setup_tensor_parallel() -> None:
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("--tensor-parallel requires CUDA, but torch.cuda.is_available() is false")
    version_parts = torch.__version__.split("+", 1)[0].split(".")
    major_minor = tuple(int(part) for part in version_parts[:2])
    if major_minor < (2, 5):
        raise RuntimeError(
            f"--tensor-parallel requires torch>=2.5 for Transformers tp_plan='auto'; "
            f"current torch is {torch.__version__}"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def barrier_if_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
        "--batch-steps",
        type=int,
        default=1,
        help="Batch this many task-drift tool steps per forward group. Only applies to --mode task_drift.",
    )
    parser.add_argument(
        "--mode",
        default="task_drift",
        choices=["whole_context", "task_drift"],
        help="Activation extraction target.",
    )
    parser.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="Use Transformers tensor parallelism via tp_plan='auto'. Launch with torchrun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tensor_parallel:
        setup_tensor_parallel()
        if args.mode != "task_drift":
            raise ValueError("--tensor-parallel is currently supported only with --mode task_drift")
    main_process = is_main_process()
    if main_process:
        print(
            f"[load] loading model={args.model} tensor_parallel={args.tensor_parallel}",
            file=sys.stderr,
            flush=True,
        )
    loaded = load_shared_model(args.model, tensor_parallel=args.tensor_parallel)
    if main_process:
        print(
            f"[load] loaded model={args.model} layers={list(loaded.cfg.layers)} "
            f"hidden_dim={loaded.cfg.hidden_dim}",
            file=sys.stderr,
            flush=True,
        )
    barrier_if_distributed()
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
            show_progress=not args.no_progress and main_process,
            log_every=args.log_every,
            batch_steps=args.batch_steps,
            write_output=main_process,
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
    barrier_if_distributed()
    if main_process:
        print(
            f"model={summary.model_name} split={summary.split} total={summary.total_samples} "
            f"skipped={getattr(summary, 'skipped_samples', getattr(summary, 'skipped_steps', 0))} "
            f"extracted={getattr(summary, 'extracted_samples', getattr(summary, 'extracted_steps', 0))} "
            f"shards_written={summary.shards_written} output_dir={summary.output_dir}"
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
