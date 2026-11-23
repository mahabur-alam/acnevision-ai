"""Tests for severity grading (Phase C) and inpainting masks (Phase D)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acnevision.classification.severity import (  # noqa: E402
    evaluate_threshold_grading,
    grade_from_count,
    grade_to_level,
)
from acnevision.restoration.inpaint import boxes_to_mask, inpaint_lesions  # noqa: E402


def test_grade_from_count_boundaries():
    assert grade_from_count(0) == "mild"
    assert grade_from_count(5) == "mild"
    assert grade_from_count(6) == "moderate"
    assert grade_from_count(20) == "moderate"
    assert grade_from_count(21) == "severe"
    assert grade_from_count(50) == "severe"      # fills the 43-50 gap
    assert grade_from_count(51) == "very_severe"
    assert grade_from_count(65) == "very_severe"


def test_grade_to_level():
    assert grade_to_level("mild") == 0
    assert grade_to_level("very_severe") == 3


def test_evaluate_threshold_grading_perfect():
    counts = [3, 10, 25, 60]
    truth = ["mild", "moderate", "severe", "very_severe"]
    res = evaluate_threshold_grading(counts, truth)
    assert res["accuracy"] == 1.0
    assert res["n"] == 4


def test_boxes_to_mask_marks_boxes_with_dilation():
    mask = boxes_to_mask((100, 100), [(40, 40, 60, 60)], dilate_px=5)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Center is inside the box -> marked.
    assert mask[50, 50] == 255
    # Dilation grows the 40-60 box by 5 -> 35..65 marked, 30 not.
    assert mask[50, 36] == 255
    assert mask[50, 30] == 0


def test_inpaint_lesions_returns_same_size_image():
    img = Image.new("RGB", (64, 64), (200, 180, 170))
    out = inpaint_lesions(img, [(20, 20, 30, 30)], method="telea")
    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)


def test_inpaint_no_boxes_is_noop():
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    out = inpaint_lesions(img, [])
    assert np.array_equal(np.asarray(out), np.asarray(img))
