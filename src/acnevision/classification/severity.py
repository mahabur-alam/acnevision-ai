"""Acne severity grading from a lesion count.

ACNE04's four classes follow the **Hayashi global acne grading** thresholds, which
are defined on the number of lesions:

====  ===========  ================
level  severity     lesion count
====  ===========  ================
0      mild         <= 5
1      moderate     6 - 20
2      severe       21 - 50
3      very_severe  >= 51
====  ===========  ================

This matches the EDA (mild 1-5, moderate 6-20, severe 21-42, very_severe 51-65):
the empty 43-50 band simply never occurred in the data but belongs to *severe*.

So once the detector (Phase B) has counted lesions on a face, severity grading is a
simple, interpretable, training-free mapping. :func:`grade_from_count` does exactly
that; :func:`evaluate_threshold_grading` checks the mapping reproduces the dataset's
own severity labels from the annotated counts.

A learned ``YOLO11-cls`` classifier is a possible alternative (handle the ~4.9x
class imbalance with class weights), but the count mapping is the recommended
baseline because it is exact on the annotated data and needs no CPU training.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Upper bound (inclusive) of each class on lesion count; ordered mild -> severe.
# very_severe is everything above the 'severe' bound.
HAYASHI_THRESHOLDS = [
    (5, "mild"),
    (20, "moderate"),
    (50, "severe"),
]
VERY_SEVERE = "very_severe"
SEVERITY_TO_LEVEL = {"mild": 0, "moderate": 1, "severe": 2, "very_severe": 3}


def grade_from_count(lesion_count: int) -> str:
    """Return the severity label for a given number of lesions.

    >>> grade_from_count(3)
    'mild'
    >>> grade_from_count(25)
    'severe'
    >>> grade_from_count(60)
    'very_severe'
    """
    if lesion_count < 0:
        raise ValueError("lesion_count cannot be negative")
    for upper, label in HAYASHI_THRESHOLDS:
        if lesion_count <= upper:
            return label
    return VERY_SEVERE


def grade_to_level(severity: str) -> int:
    """Map a severity label back to its numeric level (0-3)."""
    return SEVERITY_TO_LEVEL[severity]


def evaluate_threshold_grading(counts: list[int], truth: list[str]) -> dict[str, object]:
    """Check how well :func:`grade_from_count` reproduces known severity labels.

    Parameters
    ----------
    counts:
        Annotated lesion counts per image (e.g. ``num_lesions_annotated``).
    truth:
        The dataset's severity label for each image.

    Returns
    -------
    dict
        Accuracy and a small confusion table (truth -> predicted -> count).
    """
    if len(counts) != len(truth):
        raise ValueError("counts and truth must be the same length")

    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    for count, true_label in zip(counts, truth):
        pred = grade_from_count(count)
        if pred == true_label:
            correct += 1
        confusion.setdefault(true_label, {}).setdefault(pred, 0)
        confusion[true_label][pred] += 1

    n = len(counts)
    accuracy = round(correct / n, 4) if n else 0.0
    logger.info("Threshold grading accuracy: %.4f over %d images", accuracy, n)
    return {"accuracy": accuracy, "n": n, "confusion": confusion}
