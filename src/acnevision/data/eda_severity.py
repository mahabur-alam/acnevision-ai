"""Severity-class analysis for ACNE04 EDA (distribution + imbalance).

ACNE04 grades each face into four ordered severity classes, derived from the
Hayashi lesion-count criteria:

====  ===========  ==================================
level  severity     typical meaning
====  ===========  ==================================
0      mild         few lesions
1      moderate     more lesions
2      severe       many lesions
3      very_severe  the most lesions / nodular acne
====  ===========  ==================================

These functions quantify how (im)balanced the four classes are, which directly
affects how a YOLO/classifier should be trained (class weights, augmentation).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical order so tables/charts always read mild -> very_severe.
SEVERITY_ORDER = ["mild", "moderate", "severe", "very_severe"]
LEVEL_TO_SEVERITY = {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"}


def severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-class table: count, percentage and imbalance ratio.

    ``imbalance_ratio`` is *majority count / this class count* -- so the largest
    class is 1.0 and a value of 5.0 means "5x rarer than the biggest class".
    """
    counts = (
        df["severity"].value_counts()
        .reindex(SEVERITY_ORDER, fill_value=0)
        .rename_axis("severity")
    )
    total = int(counts.sum())
    majority = int(counts.max()) if total else 0

    table = counts.reset_index(name="count")
    table["percentage"] = (table["count"] / total * 100).round(2) if total else 0.0
    table["imbalance_ratio"] = table["count"].apply(
        lambda c: round(majority / c, 2) if c else float("inf")
    )
    return table


def level_severity_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-check that the numeric ``level`` matches the text ``severity``.

    Returns the rows where ``level`` does not map to the expected ``severity``
    (an empty frame means the two columns are perfectly consistent).
    """
    expected = df["level"].map(LEVEL_TO_SEVERITY)
    mismatched = df[expected != df["severity"]]
    logger.info("level<->severity mismatches: %d of %d rows",
                len(mismatched), len(df))
    return mismatched[["image_file", "level", "severity"]]


def imbalance_summary(table: pd.DataFrame) -> dict[str, object]:
    """Summarise imbalance from a :func:`severity_table` result."""
    nonzero = table[table["count"] > 0]
    return {
        "num_classes": int(len(nonzero)),
        "largest_class": str(table.loc[table["count"].idxmax(), "severity"]),
        "smallest_class": str(nonzero.loc[nonzero["count"].idxmin(), "severity"]),
        "max_imbalance_ratio": float(nonzero["imbalance_ratio"].max()),
        "minority_percentage": float(nonzero["percentage"].min()),
    }
