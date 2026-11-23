"""Evaluate a trained YOLO lesion detector on the ACNE04 validation split (CPU).

Reports the standard detection metrics plus a reminder to look at small-object
recall specifically, since the EDA flagged that most lesions are small after
resizing (see docs/acne04_eda_report.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_detector(
    weights: Path,
    data_yaml: Path,
    imgsz: int = 960,
    run_name: str = "acne_yolo11s_960_val",
    project: Path = Path("outputs") / "detection_eval",
) -> dict[str, float]:
    """Run validation and return a small dict of headline metrics.

    Parameters
    ----------
    weights:
        Path to trained weights (e.g. ``models/trained/<run>/weights/best.pt``).
    data_yaml:
        The dataset ``data.yaml`` (its ``val`` split is evaluated).
    imgsz:
        Validation image size (match training; 960 by default).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        imgsz=imgsz,
        device="cpu",
        # Absolute path so results are not nested under runs/detect/.
        project=str(Path(project).resolve()),
        name=run_name,
        exist_ok=True,
    )

    box = metrics.box
    summary = {
        "map50_95": round(float(box.map), 4),
        "map50": round(float(box.map50), 4),
        "map75": round(float(box.map75), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
    }
    logger.info("Validation metrics: %s", summary)
    logger.info("Also inspect per-size recall; acne lesions skew 'small' after resize.")
    return summary
