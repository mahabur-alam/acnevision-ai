"""Bounding-box analysis for ACNE04 EDA.

ACNE04 stores each lesion as a pixel-space box (``xmin``/``ymin``/``xmax``/``ymax``)
on the *original* image. These functions turn those corners into the measurements a
detector cares about -- width, height, area, aspect ratio and **image-normalized**
sizes -- and simulate what happens to lesion size when images are resized (with
letterboxing) to the candidate YOLO input resolutions 640 / 960 / 1024.

The resize simulation is the core evidence for the question *"is 640 pixels enough
to detect acne lesions?"*: a lesion only a few pixels wide after downscaling is
effectively invisible to the network.

Size buckets follow the COCO convention on pixel **area**:

* small  : area < 32**2  (1024 px^2)
* medium : 32**2 <= area < 96**2
* large  : area >= 96**2 (9216 px^2)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# COCO area thresholds (in pixels^2).
SMALL_MAX = 32 ** 2       # 1024
MEDIUM_MAX = 96 ** 2      # 9216

# Candidate YOLO input resolutions to simulate.
DEFAULT_TARGETS = (640, 960, 1024)

# A lesion narrower than this many pixels (on its shorter side) is treated as
# "too small to detect reliably" -- a practical, conservative floor for YOLO.
MIN_DETECTABLE_PX = 8


def box_metrics(boxes_df: pd.DataFrame, sizes_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-box width/height/area/aspect plus image-normalized sizes.

    Parameters
    ----------
    boxes_df:
        One row per lesion box (from
        :func:`acnevision.data.eda_data.explode_boxes`).
    sizes_df:
        Per-image DataFrame carrying ``image_file``, ``width``, ``height`` (from
        :func:`acnevision.data.eda_data.attach_image_sizes`), used to normalize
        and to drive the resize simulation.

    Returns
    -------
    pandas.DataFrame
        ``boxes_df`` plus: ``img_width``, ``img_height``, ``box_width``,
        ``box_height``, ``box_area``, ``box_aspect`` (w/h), and the normalized
        ``norm_width``, ``norm_height``, ``norm_area`` (fractions of the image).
    """
    sizes = sizes_df[["image_file", "width", "height"]].rename(
        columns={"width": "img_width", "height": "img_height"}
    )
    out = boxes_df.merge(sizes, on="image_file", how="left")

    out["box_width"] = (out["xmax"] - out["xmin"]).clip(lower=0)
    out["box_height"] = (out["ymax"] - out["ymin"]).clip(lower=0)
    out["box_area"] = out["box_width"] * out["box_height"]
    out["box_aspect"] = out["box_width"] / out["box_height"].replace(0, np.nan)

    out["norm_width"] = out["box_width"] / out["img_width"]
    out["norm_height"] = out["box_height"] / out["img_height"]
    out["norm_area"] = out["box_area"] / (out["img_width"] * out["img_height"])
    return out


def _bucket(area: pd.Series) -> pd.Series:
    """Classify a pixel-area series into small/medium/large (COCO)."""
    return pd.cut(
        area,
        bins=[-1, SMALL_MAX, MEDIUM_MAX, np.inf],
        labels=["small", "medium", "large"],
    )


def size_buckets(metrics_df: pd.DataFrame, area_col: str = "box_area") -> pd.DataFrame:
    """Return small/medium/large counts and percentages for an area column."""
    buckets = _bucket(metrics_df[area_col])
    counts = buckets.value_counts().reindex(["small", "medium", "large"], fill_value=0)
    table = counts.rename_axis("size").reset_index(name="count")
    total = int(counts.sum())
    table["percentage"] = (table["count"] / total * 100).round(2) if total else 0.0
    return table


def box_metric_summary(metrics_df: pd.DataFrame) -> dict[str, object]:
    """Headline statistics over the raw (original-resolution) box metrics."""
    def _stats(series: pd.Series) -> dict[str, float]:
        series = series.dropna()
        return {
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "mean": round(float(series.mean()), 4),
            "max": round(float(series.max()), 4),
        }

    return {
        "num_boxes": int(len(metrics_df)),
        "box_width_px": _stats(metrics_df["box_width"]),
        "box_height_px": _stats(metrics_df["box_height"]),
        "box_area_px2": _stats(metrics_df["box_area"]),
        "box_aspect": _stats(metrics_df["box_aspect"]),
        "norm_width": _stats(metrics_df["norm_width"]),
        "norm_height": _stats(metrics_df["norm_height"]),
    }


def simulate_resize(
    metrics_df: pd.DataFrame,
    targets: tuple[int, ...] = DEFAULT_TARGETS,
    min_detectable_px: int = MIN_DETECTABLE_PX,
) -> pd.DataFrame:
    """Simulate lesion size after letterbox-resizing images to each target.

    Letterboxing scales an image by ``target / max(img_width, img_height)`` so the
    longest side becomes ``target`` and the aspect ratio is preserved (exactly what
    Ultralytics YOLO does). Every box is scaled by the same factor.

    Returns one row per target with: median box side (px), median box area (px^2),
    the share of lesions that become "tiny" (shorter side < ``min_detectable_px``),
    and the small/medium/large split at that resolution.
    """
    rows: list[dict[str, object]] = []
    valid = metrics_df.dropna(subset=["img_width", "img_height"])

    for target in targets:
        scale = target / valid[["img_width", "img_height"]].max(axis=1)
        eff_w = valid["box_width"] * scale
        eff_h = valid["box_height"] * scale
        eff_area = valid["box_area"] * (scale ** 2)
        eff_short = np.minimum(eff_w, eff_h)

        buckets = _bucket(eff_area)
        bucket_counts = buckets.value_counts()
        total = int(len(eff_area))

        rows.append(
            {
                "target_px": target,
                "median_box_side_px": round(float(np.minimum(eff_w, eff_h).median()), 2),
                "median_box_area_px2": round(float(eff_area.median()), 2),
                "pct_tiny_lt_%dpx" % min_detectable_px:
                    round(float((eff_short < min_detectable_px).mean() * 100), 2),
                "pct_small": round(bucket_counts.get("small", 0) / total * 100, 2),
                "pct_medium": round(bucket_counts.get("medium", 0) / total * 100, 2),
                "pct_large": round(bucket_counts.get("large", 0) / total * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def invalid_boxes(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return boxes that are degenerate or fall outside the image bounds.

    Flags zero/negative width or height and any corner outside ``[0, img_dim]``.
    An empty result means every box is geometrically valid.
    """
    cond_zero = (metrics_df["box_width"] <= 0) | (metrics_df["box_height"] <= 0)
    cond_oob = (
        (metrics_df["xmin"] < 0)
        | (metrics_df["ymin"] < 0)
        | (metrics_df["xmax"] > metrics_df["img_width"])
        | (metrics_df["ymax"] > metrics_df["img_height"])
    )
    bad = metrics_df[cond_zero | cond_oob]
    logger.info("invalid/out-of-bounds boxes: %d of %d", len(bad), len(metrics_df))
    cols = ["image_file", "xmin", "ymin", "xmax", "ymax",
            "box_width", "box_height", "img_width", "img_height"]
    return bad[cols].reset_index(drop=True)
