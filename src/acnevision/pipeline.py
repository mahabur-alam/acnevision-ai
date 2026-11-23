"""End-to-end AcneVision inference: detect -> count -> grade -> (optional) restore.

Ties the three pillars together behind one small class:

1. **Detect** lesions with the trained YOLO model (Phase B).
2. **Count** them and **grade** severity with the Hayashi thresholds (Phase C).
3. Optionally **restore** the face by inpainting the detected lesions (Phase D).

Everything runs on CPU. The detector weights are produced by
``scripts/train_detector.py``; until then you can still exercise grading/restoration
logic directly through the Phase C/D modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .classification.severity import grade_from_count, grade_to_level
from .preprocessing.transforms import DEFAULT_IMGSZ
from .restoration.inpaint import inpaint_lesions

logger = logging.getLogger(__name__)


@dataclass
class AcneResult:
    """Outcome of running the pipeline on one image."""

    image_path: str
    boxes: list[tuple[float, float, float, float, float]]  # x1,y1,x2,y2,conf
    lesion_count: int
    severity: str
    level: int
    restored_image: Image.Image | None = None

    def summary(self) -> dict[str, object]:
        """A JSON-friendly summary (without the restored image)."""
        return {
            "image_path": self.image_path,
            "lesion_count": self.lesion_count,
            "severity": self.severity,
            "level": self.level,
            "num_boxes": len(self.boxes),
        }


class AcnePipeline:
    """Load a trained detector once and run full inference per image (CPU)."""

    def __init__(self, weights: Path, imgsz: int = DEFAULT_IMGSZ, conf: float = 0.25):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(self.weights)
        logger.info("Loaded detector %s (imgsz=%d, conf=%.2f)",
                    self.weights, imgsz, conf)

    def detect(self, image_path: Path) -> list[tuple[float, float, float, float, float]]:
        """Return detected lesion boxes ``(x1, y1, x2, y2, conf)`` in original px."""
        results = self.model.predict(
            source=str(image_path), imgsz=self.imgsz, conf=self.conf,
            device="cpu", verbose=False,
        )
        boxes: list[tuple[float, float, float, float, float]] = []
        for res in results:
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes

    def run(self, image_path: Path, restore: bool = False,
            method: str = "telea") -> AcneResult:
        """Run detect -> grade -> (optional) restore on one image."""
        image_path = Path(image_path)
        boxes = self.detect(image_path)
        count = len(boxes)
        severity = grade_from_count(count)

        restored = None
        if restore:
            with Image.open(image_path) as im:
                restored = inpaint_lesions(
                    im, [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in boxes],
                    method=method,
                )

        return AcneResult(
            image_path=str(image_path), boxes=boxes, lesion_count=count,
            severity=severity, level=grade_to_level(severity),
            restored_image=restored,
        )
