#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from secmcp.detectors.train import train_detector_from_disk
from secmcp.detectors.drift import train_drift_detector_from_disk, train_minimal_norm_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SecMCP activation detector.")
    parser.add_argument("--model", required=True, help="Model key, e.g. mistral_7b_v03.")
    parser.add_argument(
        "--detector",
        default="task_drift",
        choices=["task_drift", "minimal_norm", "rf_anchor", "logistic_diff"],
    )
    parser.add_argument("--activation-root", type=Path, default=None)
    parser.add_argument("--drift-activation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-anchors", type=int, default=None)
    parser.add_argument("--threshold-target-tpr", type=float, default=None)
    parser.add_argument("--threshold-max-fpr", type=float, default=None)
    parser.add_argument("--sample-weight", choices=["none", "balanced"], default=None)
    parser.add_argument("--no-test", action="store_true", help="Skip held-out test evaluation.")
    parser.add_argument("--no-progress", action="store_true", help="Disable shard loading and training progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[train] model={args.model} detector={args.detector} "
        f"n_anchors={args.n_anchors or 'config'} include_test={not args.no_test}",
        file=sys.stderr,
        flush=True,
    )
    if args.detector == "task_drift":
        detector = train_drift_detector_from_disk(
            model_name=args.model,
            drift_activation_root=args.drift_activation_root or args.activation_root,
            output_root=args.output_root,
            n_anchors=args.n_anchors,
            include_test=not args.no_test,
            threshold_target_tpr=args.threshold_target_tpr,
            threshold_max_fpr=args.threshold_max_fpr,
            sample_weight=args.sample_weight,
            show_progress=not args.no_progress,
        )
    elif args.detector == "minimal_norm":
        detector = train_minimal_norm_from_disk(
            model_name=args.model,
            drift_activation_root=args.drift_activation_root or args.activation_root,
            output_root=args.output_root,
            include_test=not args.no_test,
            show_progress=not args.no_progress,
        )
    else:
        detector = train_detector_from_disk(
            model_name=args.model,
            detector_type=args.detector,
            activation_root=args.activation_root,
            output_root=args.output_root,
            n_anchors=args.n_anchors,
            include_test=not args.no_test,
            show_progress=not args.no_progress,
        )
    print("[train] finished", file=sys.stderr, flush=True)
    print(f"model={detector.model_name} detector={getattr(detector, 'detector_type', args.detector)}")
    print(f"val={detector.val_metrics}")
    if detector.test_metrics is not None:
        print(f"test={detector.test_metrics}")


if __name__ == "__main__":
    main()
