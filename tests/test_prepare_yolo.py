"""Tests for the YOLO label conversion (Phase A), using tiny synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acnevision.detection.prepare_yolo import (  # noqa: E402
    box_to_yolo,
    prepare_yolo_dataset,
)


def test_box_to_yolo_centers_and_normalizes():
    # 100x100 image, box (10,20)-(30,60) -> center (20,40), size (20,40)
    assert box_to_yolo(10, 20, 30, 60, 100, 100) == (0.2, 0.4, 0.2, 0.4)


def test_box_to_yolo_clamps_out_of_bounds():
    xc, yc, w, h = box_to_yolo(-10, -10, 120, 120, 100, 100)
    assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
    assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0


def _make_dataset(tmp_path: Path) -> Path:
    """Create a 2-image processed dataset (one train, one test)."""
    processed = tmp_path / "train"
    images = processed / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (200, 180, 170)).save(images / "acne_00000.jpg")
    Image.new("RGB", (200, 100), (200, 180, 170)).save(images / "acne_00001.jpg")

    records = [
        {"image_file": "acne_00000.jpg", "severity": "mild", "level": 0,
         "num_lesions_annotated": 1, "lesion_count_label": 1, "split": "train",
         "lesion_boxes": {"xmin": [10], "ymin": [20], "xmax": [30], "ymax": [60]}},
        {"image_file": "acne_00001.jpg", "severity": "moderate", "level": 1,
         "num_lesions_annotated": 2, "lesion_count_label": 2, "split": "test",
         "lesion_boxes": {"xmin": [0, 100], "ymin": [0, 0],
                          "xmax": [50, 200], "ymax": [50, 100]}},
    ]
    with (processed / "annotations.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return processed


def test_prepare_yolo_dataset_layout_and_counts(tmp_path):
    processed = _make_dataset(tmp_path)
    out = tmp_path / "yolo"
    stats = prepare_yolo_dataset(processed_dir=processed, out_dir=out,
                                 fold_column="split")

    assert stats.images_by_split == {"train": 1, "val": 1}
    assert stats.labels_written == 2
    assert stats.total_boxes == 3  # 1 + 2

    # data.yaml and the four leaf folders exist.
    assert (out / "data.yaml").exists()
    assert (out / "images" / "train" / "acne_00000.jpg").exists()
    assert (out / "images" / "val" / "acne_00001.jpg").exists()

    # train label matches the hand-computed normalized box.
    line = (out / "labels" / "train" / "acne_00000.txt").read_text().strip()
    assert line == "0 0.200000 0.400000 0.200000 0.400000"

    # val image has two boxes.
    val_lines = (out / "labels" / "val" / "acne_00001.txt").read_text().strip().split("\n")
    assert len(val_lines) == 2
    assert all(l.startswith("0 ") for l in val_lines)
