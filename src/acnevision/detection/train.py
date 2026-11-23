"""Train a YOLO lesion detector on the prepared ACNE04 dataset (CPU only).

Thin, documented wrapper around Ultralytics ``YOLO.train`` that applies the
EDA-driven defaults from :mod:`acnevision.preprocessing.transforms`
(``imgsz = 960``, ``device = 'cpu'``, conservative augmentation).

The recommended model is **YOLO11s**; drop to **YOLO11n** on slow machines by
passing ``model_weights="yolo11n.pt"``. Weights download automatically on first use.

CPU note: YOLO11s @ 960 px is slow on CPU. For a quick pipeline check, lower
``epochs`` / ``imgsz`` via overrides -- the function passes any extra keyword
straight through to Ultralytics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..preprocessing.transforms import DEFAULT_TRAIN_ARGS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_PROJECT = Path("models") / "trained"


def train_detector(
    data_yaml: Path,
    model_weights: str = DEFAULT_MODEL,
    project: Path = DEFAULT_PROJECT,
    run_name: str = "acne_yolo11s_960",
    **overrides: Any,
) -> Any:
    """Train a YOLO detector and return the Ultralytics results object.

    Parameters
    ----------
    data_yaml:
        Path to the ``data.yaml`` written by
        :func:`acnevision.detection.prepare_yolo.prepare_yolo_dataset`.
    model_weights:
        Pretrained checkpoint to fine-tune (``yolo11s.pt`` default,
        ``yolo11n.pt`` for a lighter/faster CPU run).
    project, run_name:
        Output goes to ``<project>/<run_name>/`` (weights, curves, args).
    **overrides:
        Any Ultralytics training argument (e.g. ``epochs=5``, ``imgsz=640``)
        overrides the CPU defaults.
    """
    # Import here so the heavy dependency is only needed when actually training.
    from ultralytics import YOLO

    args = {**DEFAULT_TRAIN_ARGS, **overrides}
    args["device"] = "cpu"  # hard constraint: never use GPU in this project
    logger.info("Training %s on %s with args: %s", model_weights, data_yaml, args)

    model = YOLO(model_weights)
    results = model.train(
        data=str(Path(data_yaml).resolve()),
        # Absolute project path so Ultralytics does NOT nest it under runs/detect/.
        project=str(Path(project).resolve()),
        name=run_name,
        exist_ok=True,
        **args,
    )
    logger.info("Training finished. Best weights under %s/%s/weights/best.pt",
                project, run_name)
    return results
