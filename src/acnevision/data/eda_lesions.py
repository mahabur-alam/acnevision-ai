"""Lesion-count analysis for ACNE04 EDA.

How many annotated lesions does each face carry? These functions summarise the
per-image lesion counts and cross-check the three count fields that ACNE04 stores
(``num_boxes`` = boxes actually present, ``num_lesions_annotated`` and
``lesion_count_label``) so any disagreement is surfaced as a data-quality issue
rather than silently trusted.

Expects the per-image DataFrame from
:func:`acnevision.data.eda_data.load_dataframe` (which adds ``num_boxes``).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def lesion_count_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return headline lesion-count statistics across all images."""
    counts = df["num_boxes"]
    return {
        "total_lesions": int(counts.sum()),
        "num_images": int(len(df)),
        "mean_per_image": round(float(counts.mean()), 2),
        "median_per_image": float(counts.median()),
        "min_per_image": int(counts.min()),
        "max_per_image": int(counts.max()),
        "images_with_zero_boxes": int((counts == 0).sum()),
    }


def lesions_per_image_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frequency table: how many images have N lesions."""
    dist = df["num_boxes"].value_counts().sort_index().rename_axis("num_lesions")
    return dist.reset_index(name="num_images")


def lesions_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / max lesion count grouped by severity class."""
    grouped = (
        df.groupby("severity")["num_boxes"]
        .agg(["count", "mean", "median", "min", "max", "sum"])
        .round(2)
    )
    return grouped.reset_index()


def top_images_by_lesions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the ``n`` images with the most annotated lesions."""
    cols = ["image_file", "severity", "num_boxes", "num_lesions_annotated"]
    present = [c for c in cols if c in df.columns]
    return df.sort_values("num_boxes", ascending=False).head(n)[present].reset_index(drop=True)


def count_field_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where the three lesion-count fields disagree.

    Compares the number of boxes actually stored (``num_boxes``) against
    ``num_lesions_annotated`` and ``lesion_count_label``. Returns only the
    disagreeing rows (empty means all three agree everywhere).
    """
    mask = pd.Series(False, index=df.index)
    if "num_lesions_annotated" in df.columns:
        mask |= df["num_boxes"] != df["num_lesions_annotated"]
    if "lesion_count_label" in df.columns:
        mask |= df["num_boxes"] != df["lesion_count_label"]

    cols = [c for c in ["image_file", "severity", "num_boxes",
                        "num_lesions_annotated", "lesion_count_label"]
            if c in df.columns]
    mismatched = df.loc[mask, cols]
    logger.info("count-field mismatches: %d of %d rows", len(mismatched), len(df))
    return mismatched.reset_index(drop=True)
