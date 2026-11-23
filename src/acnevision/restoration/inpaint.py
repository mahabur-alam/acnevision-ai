"""Blemish removal by inpainting the detected lesion regions (CPU only).

Given the lesion boxes from the detector (Phase B), we build a binary mask over
those regions and let OpenCV *inpaint* them -- i.e. fill each lesion with plausible
surrounding skin texture. This is a classical, CPU-friendly approach (no GPU, no
training) and a solid baseline for the "facial blemish removal" goal; the interface
leaves room to swap in a learned inpainting model later.

Two OpenCV algorithms are supported:

* ``telea`` -- Fast Marching Method (Telea 2004); good default for small blemishes.
* ``ns``    -- Navier-Stokes based; sometimes smoother on larger regions.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_METHODS = ("telea", "ns")


def boxes_to_mask(
    image_size: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
    dilate_px: int = 4,
) -> np.ndarray:
    """Build a uint8 mask (255 inside lesion boxes) for an ``(width, height)`` image.

    Each box is expanded by ``dilate_px`` on every side so the inpainter also
    rebuilds the skin right at the lesion's edge. Boxes are clipped to the image.
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        xa = max(0, int(round(min(x1, x2))) - dilate_px)
        ya = max(0, int(round(min(y1, y2))) - dilate_px)
        xb = min(width, int(round(max(x1, x2))) + dilate_px)
        yb = min(height, int(round(max(y1, y2))) + dilate_px)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 255
    return mask


def inpaint_lesions(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    method: str = "telea",
    dilate_px: int = 4,
    radius: int = 5,
) -> Image.Image:
    """Return a copy of ``image`` with the given lesion boxes inpainted away.

    Parameters
    ----------
    image:
        The original face image (any mode; converted to RGB).
    boxes:
        Lesion boxes as ``(x1, y1, x2, y2)`` in original pixel coordinates.
    method:
        ``'telea'`` (default) or ``'ns'``.
    dilate_px:
        How far to grow each box before filling (blends the edges).
    radius:
        OpenCV inpainting neighbourhood radius (px).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    import cv2  # imported here so restoration is optional at import time

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not boxes:
        logger.info("No lesion boxes given; returning the original image unchanged.")
        return Image.fromarray(rgb)

    mask = boxes_to_mask((rgb.shape[1], rgb.shape[0]), boxes, dilate_px=dilate_px)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    # OpenCV works in BGR; convert in and back out.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restored_bgr = cv2.inpaint(bgr, mask, inpaintRadius=radius, flags=flag)
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    logger.info("Inpainted %d lesion region(s) with method '%s'", len(boxes), method)
    return Image.fromarray(restored_rgb)
