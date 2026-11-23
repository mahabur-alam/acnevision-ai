"""Image preprocessing shared by training and inference.

The ACNE04 images vary enormously in size (143-5184 px wide, see the EDA report),
so every image must be brought to a fixed square input before it reaches the model.
We use **letterboxing** -- scale by the longer side and pad the rest -- which is
exactly what Ultralytics YOLO does internally. Keeping the same function here means
our own inference/visualisation matches training.

Also defines the CPU-friendly training hyper-parameters chosen from the EDA
(``imgsz = 960``; see docs/acne04_eda_report.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Input resolution chosen by the EDA resize simulation (960 roughly halves the
# share of COCO-'small' lesions vs 640, for modest extra CPU cost).
DEFAULT_IMGSZ = 960


@dataclass(frozen=True)
class LetterboxResult:
    """A letterboxed image plus the geometry needed to map boxes back."""

    image: Image.Image
    scale: float          # factor the ORIGINAL was multiplied by
    pad_left: int
    pad_top: int
    new_size: int         # final square side (== imgsz)


def letterbox(image: Image.Image, imgsz: int = DEFAULT_IMGSZ,
              fill: tuple[int, int, int] = (114, 114, 114)) -> LetterboxResult:
    """Resize ``image`` to an ``imgsz`` square, preserving aspect with padding.

    Returns the padded image and the ``scale``/``pad`` values so a prediction made
    in letterboxed space can be mapped back to original pixels via
    :func:`undo_letterbox_box`.
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = imgsz / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (imgsz, imgsz), fill)
    pad_left = (imgsz - new_w) // 2
    pad_top = (imgsz - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return LetterboxResult(canvas, scale, pad_left, pad_top, imgsz)


def undo_letterbox_box(box: tuple[float, float, float, float],
                       lb: LetterboxResult) -> tuple[float, float, float, float]:
    """Map a ``(x1, y1, x2, y2)`` box from letterboxed space to original pixels."""
    x1, y1, x2, y2 = box
    x1 = (x1 - lb.pad_left) / lb.scale
    x2 = (x2 - lb.pad_left) / lb.scale
    y1 = (y1 - lb.pad_top) / lb.scale
    y2 = (y2 - lb.pad_top) / lb.scale
    return (x1, y1, x2, y2)


def to_array(image: Image.Image) -> np.ndarray:
    """Return an ``HxWx3`` uint8 RGB array (handy for OpenCV / model input)."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


# CPU-friendly YOLO training hyper-parameters (passed to ``YOLO.train``).
# Mild geometric aug + HSV; mosaic helps small-object recall. Kept conservative so
# lesions are never distorted off real skin.
DEFAULT_TRAIN_ARGS: dict[str, object] = {
    "imgsz": DEFAULT_IMGSZ,
    "device": "cpu",
    "epochs": 50,
    "batch": 4,          # small, so 960-px images fit in CPU RAM
    "patience": 15,
    "mosaic": 1.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,       # faces have a clear up/down orientation
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
}
