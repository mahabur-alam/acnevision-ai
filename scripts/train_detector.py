#!/usr/bin/env python3
"""Train (and optionally evaluate) the YOLO lesion detector on CPU.

Run from the repo root after building the dataset with
``scripts/prepare_yolo_dataset.py``::

    # Full recommended run (slow on CPU): YOLO11s @ 960
    python scripts/train_detector.py

    # Fast pipeline smoke-test: lighter model, tiny image size, few epochs
    python scripts/train_detector.py --model yolo11n.pt --imgsz 320 --epochs 2 --batch 2

    # Evaluate existing weights only
    python scripts/train_detector.py --eval-only \
        --weights models/trained/acne_yolo11s_960/weights/best.pt

Everything is forced onto the CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "ACNE04" / "yolo" / "data.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate the ACNE04 YOLO lesion detector (CPU only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Path to the YOLO data.yaml.")
    p.add_argument("--model", default="yolo11s.pt",
                   help="Pretrained weights to fine-tune (yolo11s.pt / yolo11n.pt).")
    p.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    p.add_argument("--batch", type=int, default=4, help="Batch size (keep small on CPU).")
    p.add_argument("--run-name", default="acne_yolo11s_960", help="Output run name.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate --weights.")
    p.add_argument("--weights", type=Path, default=None,
                   help="Weights to evaluate (defaults to the just-trained best.pt).")
    p.add_argument("--no-eval", action="store_true",
                   help="Train only; skip the validation pass.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )

    from acnevision.detection.train import DEFAULT_PROJECT, train_detector
    from acnevision.detection.evaluate import evaluate_detector

    weights = args.weights
    if not args.eval_only:
        train_detector(
            data_yaml=args.data, model_weights=args.model,
            run_name=args.run_name, imgsz=args.imgsz,
            epochs=args.epochs, batch=args.batch,
        )
        weights = PROJECT_ROOT / DEFAULT_PROJECT / args.run_name / "weights" / "best.pt"

    if not args.no_eval:
        if weights is None or not Path(weights).exists():
            print(f"No weights to evaluate at {weights}", file=sys.stderr)
            return 1
        summary = evaluate_detector(weights=weights, data_yaml=args.data,
                                    imgsz=args.imgsz, run_name=f"{args.run_name}_val")
        print("\n=== Detection validation metrics ===")
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
