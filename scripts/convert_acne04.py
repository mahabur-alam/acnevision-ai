#!/usr/bin/env python3
"""Command-line entry point for the ACNE04 dataset pipeline.

Runnable from the project root, no installation required, e.g.::

    python scripts/convert_acne04.py                 # convert (idempotent)
    python scripts/convert_acne04.py --inspect        # print schema + sample
    python scripts/convert_acne04.py --validate       # check processed output
    python scripts/convert_acne04.py --force          # re-extract all images

By default it reads the Arrow shards from ``data/raw/ACNE04/train`` and writes
images + annotations to ``data/processed/ACNE04/train``.

Everything runs on CPU; no GPU is required anywhere in this project.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``src/`` importable when running straight from the repo (no install).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from acnevision.data.convert_arrow_to_jpg import (  # noqa: E402
    convert_all_arrow_to_jpg,
    write_summary,
)
from acnevision.data.inspect_dataset import inspect  # noqa: E402
from acnevision.data.validate_dataset import validate  # noqa: E402

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ACNE04" / "train"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ACNE04" / "train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ACNE04 Arrow shards to JPG + preserved annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Folder containing the data-*-of-*.arrow shards.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Destination folder for images/, annotations.jsonl, dataset_summary.json.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG quality (1-95) for newly written images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-decode and overwrite images that already exist.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print the dataset schema and a sample record, then exit.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the already-processed output, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.inspect:
        inspect(args.raw_dir)
        return 0

    if args.validate:
        report = validate(args.out_dir)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    stats = convert_all_arrow_to_jpg(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        quality=args.quality,
        force=args.force,
    )
    summary_path = write_summary(stats, args.out_dir)

    print("\n=== Conversion summary ===")
    print(f"  shards processed : {len(stats.shards)}")
    print(f"  total records    : {stats.total_records}")
    print(f"  extracted        : {stats.extracted}")
    print(f"  skipped (exists) : {stats.skipped}")
    print(f"  failed           : {stats.failed}")
    print(f"  summary file     : {summary_path}")

    # A non-zero exit on failures makes the script friendly to automation.
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
