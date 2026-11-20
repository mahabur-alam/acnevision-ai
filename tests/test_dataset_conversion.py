"""Tests for the dataset conversion and annotation-mapping logic.

These tests use tiny synthetic data and do NOT touch the real ~1 GB Arrow
shards, so they run fast on CPU. Run from the project root with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

# Make ``src/`` importable without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from acnevision.data.convert_arrow_to_jpg import (  # noqa: E402
    build_annotation,
    image_filename,
)
from acnevision.data.validate_dataset import load_annotations, validate  # noqa: E402


# --------------------------------------------------------------------------- #
# Filename generation
# --------------------------------------------------------------------------- #
def test_image_filename_is_zero_padded():
    assert image_filename(0) == "acne_00000.jpg"
    assert image_filename(42) == "acne_00042.jpg"
    assert image_filename(1456) == "acne_01456.jpg"


def test_image_filenames_unique_across_simulated_shards():
    """Sequential global indices across shards must never collide."""
    shard_sizes = [486, 486, 485]  # mirrors the real ACNE04 shards
    names = []
    index = 0
    for size in shard_sizes:
        for _ in range(size):
            names.append(image_filename(index))
            index += 1
    assert len(names) == 1457
    assert len(set(names)) == 1457  # all unique, no collisions


# --------------------------------------------------------------------------- #
# Annotation mapping
# --------------------------------------------------------------------------- #
def test_build_annotation_preserves_all_fields_and_nested_boxes():
    record = {
        "file_name": "levle0_0.jpg",
        "level": 0,
        "severity": "mild",
        "lesion_count_label": 3,
        "num_lesions_annotated": 3,
        "lesion_boxes": {
            "xmin": [223, 217, 186],
            "ymin": [290, 250, 514],
            "xmax": [251, 239, 209],
            "ymax": [306, 276, 536],
        },
        "split": "train",
        "fold_0_split": "train",
        "fold_3_split": "test",
    }
    ann = build_annotation(record, "acne_00000.jpg")

    # The generated image filename is added...
    assert ann["image_file"] == "acne_00000.jpg"
    # ...and every original field is preserved verbatim.
    for key, value in record.items():
        assert ann[key] == value
    # Nested bounding-box structure survives untouched.
    assert ann["lesion_boxes"]["xmin"] == [223, 217, 186]
    assert ann["lesion_boxes"]["ymax"][-1] == 536


def test_annotation_is_json_round_trippable():
    record = {
        "file_name": "x.jpg",
        "level": 2,
        "lesion_boxes": {"xmin": [1], "ymin": [2], "xmax": [3], "ymax": [4]},
    }
    ann = build_annotation(record, "acne_00001.jpg")
    restored = json.loads(json.dumps(ann))
    assert restored == ann


# --------------------------------------------------------------------------- #
# JSONL round-trip + validation (end to end on a tiny synthetic dataset)
# --------------------------------------------------------------------------- #
def _make_jpg(path: Path, size=(8, 8)) -> None:
    Image.new("RGB", size, color=(128, 64, 32)).save(path, format="JPEG")


def test_jsonl_round_trip(tmp_path: Path):
    annotations_path = tmp_path / "annotations.jsonl"
    records = [
        build_annotation({"level": 0, "severity": "mild"}, image_filename(0)),
        build_annotation({"level": 1, "severity": "moderate"}, image_filename(1)),
    ]
    with annotations_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    loaded = load_annotations(annotations_path)
    assert loaded == records


def test_validate_passes_on_consistent_dataset(tmp_path: Path):
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    records = []
    for i in range(3):
        name = image_filename(i)
        _make_jpg(images / name)
        records.append(build_annotation({"level": i}, name))
    with (processed / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    report = validate(processed)
    assert report["ok"] is True
    assert report["num_annotations"] == 3
    assert report["num_images_on_disk"] == 3


def test_validate_detects_missing_image(tmp_path: Path):
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    # Annotation references acne_00001.jpg but we never create that file.
    _make_jpg(images / image_filename(0))
    records = [
        build_annotation({"level": 0}, image_filename(0)),
        build_annotation({"level": 1}, image_filename(1)),  # missing on disk
    ]
    with (processed / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    report = validate(processed)
    assert report["ok"] is False
    assert report["missing_images"] == 1


def test_validate_detects_orphan_image(tmp_path: Path):
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    # Two images on disk but only one annotation -> one orphan.
    _make_jpg(images / image_filename(0))
    _make_jpg(images / image_filename(1))
    with (processed / "annotations.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(build_annotation({"level": 0}, image_filename(0))) + "\n")

    report = validate(processed)
    assert report["ok"] is False
    assert report["orphan_images"] == 1
