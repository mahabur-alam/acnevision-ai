#!/usr/bin/env python3
"""Run the full AcneVision pipeline on an image: detect -> grade -> (optional) restore.

Run from the repo root with trained detector weights::

    python scripts/run_pipeline.py --image path/to/face.jpg \
        --weights models/trained/acne_yolo11s_960/weights/best.pt

    # Also produce a blemish-removed image
    python scripts/run_pipeline.py --image face.jpg --weights best.pt \
        --restore --out outputs/predictions/face_restored.jpg

Everything runs on CPU.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect acne lesions, grade severity, optionally remove blemishes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", type=Path, required=True, help="Input face image.")
    p.add_argument("--weights", type=Path, required=True,
                   help="Trained YOLO detector weights (best.pt).")
    p.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    p.add_argument("--restore", action="store_true",
                   help="Also inpaint the detected lesions.")
    p.add_argument("--method", default="telea", choices=["telea", "ns"],
                   help="Inpainting algorithm.")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "outputs" / "predictions" / "restored.jpg",
                   help="Where to save the restored image (with --restore).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )

    from acnevision.pipeline import AcnePipeline

    pipe = AcnePipeline(weights=args.weights, imgsz=args.imgsz, conf=args.conf)
    result = pipe.run(args.image, restore=args.restore, method=args.method)

    print("\n=== AcneVision result ===")
    print(json.dumps(result.summary(), indent=2))

    if args.restore and result.restored_image is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.restored_image.save(args.out, quality=92)
        print(f"  restored image  : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
