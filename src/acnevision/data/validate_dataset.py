"""Validate the processed dataset against the project's acceptance criteria.

Checks that:
* ``annotations.jsonl`` exists and every record has an ``image_file``.
* Every referenced image file actually exists on disk.
* Image filenames are unique (no collisions).
* Every image on disk is referenced by an annotation (no orphans).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_annotations(annotations_path: Path) -> list[dict[str, Any]]:
    """Load an ``annotations.jsonl`` file into a list of dicts."""
    annotations_path = Path(annotations_path)
    records: list[dict[str, Any]] = []
    with annotations_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_no} of {annotations_path}: {exc}"
                ) from exc
    return records


def validate(processed_dir: Path) -> dict[str, Any]:
    """Validate the processed dataset folder and return a report.

    ``report["ok"]`` is ``True`` only when every check passes.
    """
    processed_dir = Path(processed_dir)
    images_dir = processed_dir / "images"
    annotations_path = processed_dir / "annotations.jsonl"

    problems: list[str] = []

    if not annotations_path.exists():
        return {"ok": False, "problems": [f"Missing {annotations_path}"]}
    if not images_dir.is_dir():
        problems.append(f"Missing images directory {images_dir}")

    records = load_annotations(annotations_path)

    filenames: list[str] = []
    missing_images: list[str] = []
    missing_field: int = 0
    for rec in records:
        name = rec.get("image_file")
        if not name:
            missing_field += 1
            continue
        filenames.append(name)
        if not (images_dir / name).exists():
            missing_images.append(name)

    unique_names = set(filenames)
    duplicates = len(filenames) - len(unique_names)

    image_files_on_disk = (
        {p.name for p in images_dir.glob("*.jpg")} if images_dir.is_dir() else set()
    )
    orphan_images = sorted(image_files_on_disk - unique_names)

    if missing_field:
        problems.append(f"{missing_field} annotation(s) missing 'image_file'")
    if missing_images:
        problems.append(
            f"{len(missing_images)} annotation(s) reference a missing image "
            f"(e.g. {missing_images[:3]})"
        )
    if duplicates:
        problems.append(f"{duplicates} duplicate image filename(s) in annotations")
    if orphan_images:
        problems.append(
            f"{len(orphan_images)} image file(s) have no annotation "
            f"(e.g. {orphan_images[:3]})"
        )

    report: dict[str, Any] = {
        "ok": not problems,
        "num_annotations": len(records),
        "num_images_on_disk": len(image_files_on_disk),
        "unique_filenames": len(unique_names),
        "missing_images": len(missing_images),
        "duplicate_filenames": duplicates,
        "orphan_images": len(orphan_images),
        "problems": problems,
    }

    if report["ok"]:
        logger.info(
            "Validation PASSED: %d annotations, %d images, all consistent.",
            report["num_annotations"],
            report["num_images_on_disk"],
        )
    else:
        logger.warning("Validation FAILED:")
        for problem in problems:
            logger.warning("  - %s", problem)

    return report
