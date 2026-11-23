#!/usr/bin/env python3
"""Build a YOLO-ready dataset from the processed ACNE04 data.

Runnable from the project root, no installation required, e.g.::

    python scripts/prepare_yolo_dataset.py                       # use 'split'
    python scripts/prepare_yolo_dataset.py --fold fold_2_split   # use CV fold 2
    python scripts/prepare_yolo_dataset.py --out-dir data/processed/ACNE04/yolo

It converts every ``lesion_boxes`` entry into normalized YOLO labels, symlinks the
images into ``images/{train,val}``, and writes ``data.yaml``. It never copies images
and never modifies the source annotations. Everything runs on CPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from acnevision.detection.prepare_yolo import prepare_yolo_dataset  # noqa: E402

DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "ACNE04" / "train"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "ACNE04" / "yolo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ACNE04 annotations into a YOLO dataset (labels + data.yaml).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED,
                        help="Folder with annotations.jsonl and images/.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Destination folder for the YOLO dataset layout.")
    parser.add_argument("--fold", default="split",
                        help="Split column: 'split' or fold_0_split..fold_4_split.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = prepare_yolo_dataset(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        fold_column=args.fold,
    )

    print("\n=== YOLO dataset summary ===")
    print(f"  fold column      : {stats.fold_column}")
    print(f"  images by split  : {stats.images_by_split}")
    print(f"  label files      : {stats.labels_written}")
    print(f"  total boxes      : {stats.total_boxes}")
    print(f"  skipped (unread) : {stats.skipped_unreadable}")
    print(f"  output dir       : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
