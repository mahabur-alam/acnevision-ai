"""Image-level analysis for ACNE04 EDA (counts, sizes, formats, quality).

Works on the DataFrame produced by
:func:`acnevision.data.eda_data.attach_image_sizes`, which already carries the
``width``/``height``/``aspect_ratio``/``mode``/``img_format`` columns and the
``missing``/``corrupted`` flags. Every function only *reads* and returns tidy
numbers or small DataFrames for charting.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def summarize_images(df: pd.DataFrame) -> dict[str, object]:
    """Return a compact dictionary of headline image statistics.

    Expects the size/quality columns added by ``attach_image_sizes``. Width and
    height stats are computed only over readable images.
    """
    ok = df[(~df["missing"]) & (~df["corrupted"])]
    widths = ok["width"].dropna()
    heights = ok["height"].dropna()
    ratios = ok["aspect_ratio"].dropna()

    def _stats(series: pd.Series) -> dict[str, float]:
        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 2),
            "median": float(series.median()),
        }

    return {
        "total_images": int(len(df)),
        "readable_images": int(len(ok)),
        "missing_images": int(df["missing"].sum()),
        "corrupted_images": int(df["corrupted"].sum()),
        "width": _stats(widths) if len(widths) else None,
        "height": _stats(heights) if len(heights) else None,
        "aspect_ratio": _stats(ratios) if len(ratios) else None,
    }


def format_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count images per file format (e.g. JPEG) -- see ``img_format`` column."""
    counts = df["img_format"].value_counts(dropna=False).rename_axis("img_format")
    return counts.reset_index(name="count")


def channel_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count images per colour mode (``RGB``, ``L`` grayscale, ...).

    Useful before training: a stray grayscale (``L``) or ``RGBA`` image needs
    converting to 3-channel RGB. ACNE04 JPGs were saved as RGB by the converter.
    """
    counts = df["mode"].value_counts(dropna=False).rename_axis("mode")
    out = counts.reset_index(name="count")
    out["channels"] = out["mode"].map(
        {"RGB": 3, "L": 1, "RGBA": 4, "CMYK": 4, "P": 1}
    )
    return out


def resolution_table(df: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Bucket images by megapixel size to see the spread of resolutions.

    Returns a DataFrame with the megapixel range and how many images fall in it.
    """
    ok = df[(~df["missing"]) & (~df["corrupted"])].copy()
    ok["megapixels"] = (ok["width"] * ok["height"]) / 1_000_000
    buckets = pd.cut(ok["megapixels"], bins=bins)
    table = buckets.value_counts().sort_index().rename_axis("megapixel_range")
    return table.reset_index(name="count")
