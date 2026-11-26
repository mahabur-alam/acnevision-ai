"""Turn the processed ACNE04 annotations into a YOLO-ready dataset.

ACNE04 stores each lesion as a pixel-space box (``xmin``/``ymin``/``xmax``/``ymax``)
on the original image. Ultralytics YOLO instead wants, per image, a ``.txt`` label
file with one line per box::

    <class_id> <x_center> <y_center> <width> <height>

where every value is **normalized** to ``[0, 1]`` by the image's own width/height.
This module does that conversion and lays out the standard Ultralytics folder
structure, **without copying the images** (it symlinks them, so the ~1.1 GB of JPGs
is never duplicated)::

    <out_dir>/
        images/train/acne_00000.jpg -> ../../../images/acne_00000.jpg (symlink)
        images/val/...
        labels/train/acne_00000.txt
        labels/val/...
        data.yaml

Design choices (see docs/acne04_eda_report.md):
* Single detection class ``lesion`` (``nc = 1``) -- severity is a separate task.
* Boxes come from ``lesion_boxes`` only; the ``lesion_count_label`` field is ignored
  because the EDA showed it disagrees with the drawn boxes on 21 very_severe images.
* The split is taken from an existing column (default ``split`` == ``fold_0``);
  ``train`` stays train and ``test`` becomes YOLO's ``val``. Assignments are read,
  never changed.

Everything here is read-only with respect to the source dataset: it only writes into
``out_dir``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..data.eda_data import attach_image_sizes, default_processed_dir, load_dataframe

logger = logging.getLogger(__name__)

# ACNE04 'test' rows become YOLO's validation set.
SPLIT_TO_YOLO = {"train": "train", "test": "val"}
LESION_CLASS_ID = 0
CLASS_NAMES = ["lesion"]


def _clamp01(value: float) -> float:
    """Clamp a normalized coordinate into the valid ``[0, 1]`` range."""
    return min(max(value, 0.0), 1.0)


def box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert one pixel box to normalized YOLO ``(xc, yc, w, h)``.

    >>> box_to_yolo(10, 20, 30, 60, 100, 100)
    (0.2, 0.4, 0.2, 0.4)
    """
    if not img_w or not img_h:
        raise ValueError("image width/height must be positive and known")
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return (_clamp01(xc), _clamp01(yc), _clamp01(w), _clamp01(h))


def yolo_label_lines(record: pd.Series) -> list[str]:
    """Build the YOLO label lines for one image record.

    ``record`` must carry ``lesion_boxes`` and the real ``width``/``height``
    (from :func:`acnevision.data.eda_data.attach_image_sizes`). Degenerate boxes
    (zero/negative width or height after clamping) are skipped with a warning.
    """
    boxes = record["lesion_boxes"] or {}
    img_w, img_h = record["width"], record["height"]
    lines: list[str] = []
    coords = zip(
        boxes.get("xmin", []), boxes.get("ymin", []),
        boxes.get("xmax", []), boxes.get("ymax", []),
    )
    for xmin, ymin, xmax, ymax in coords:
        xc, yc, w, h = box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if w <= 0 or h <= 0:
            logger.warning("Skipping degenerate box in %s", record["image_file"])
            continue
        lines.append(f"{LESION_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def _symlink(src: Path, dst: Path) -> None:
    """Create/refresh a relative symlink ``dst`` -> ``src`` (copy as fallback)."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError as exc:  # e.g. filesystem without symlink support
        logger.warning("Symlink failed (%s); copying instead: %s", exc, dst.name)
        dst.write_bytes(src.read_bytes())


@dataclass
class PrepareStats:
    """Counts describing a YOLO dataset build."""

    fold_column: str = "split"
    images_by_split: dict[str, int] = field(default_factory=dict)
    labels_written: int = 0
    total_boxes: int = 0
    skipped_unknown_split: int = 0
    skipped_unreadable: int = 0


def prepare_yolo_dataset(
    processed_dir: Path | None = None,
    out_dir: Path | None = None,
    fold_column: str = "split",
) -> PrepareStats:
    """Build an Ultralytics YOLO dataset from the processed ACNE04 data.

    Parameters
    ----------
    processed_dir:
        Folder with ``annotations.jsonl`` and ``images/``. Defaults to
        ``data/processed/ACNE04/train``.
    out_dir:
        Where the YOLO layout is written. Defaults to
        ``<processed_dir>/../yolo`` (i.e. ``data/processed/ACNE04/yolo``).
    fold_column:
        Which split column to use (``split`` or ``fold_0_split`` ...
        ``fold_4_split``). ``train`` -> YOLO train, ``test`` -> YOLO val.

    Returns
    -------
    PrepareStats
        Per-split image counts, labels written and total boxes.
    """
    processed_dir = default_processed_dir(processed_dir)
    images_dir = processed_dir / "images"
    out_dir = Path(out_dir) if out_dir else processed_dir.parent / "yolo"

    df = attach_image_sizes(load_dataframe(processed_dir))
    if fold_column not in df.columns:
        raise ValueError(
            f"fold_column '{fold_column}' not found. Available: "
            f"{[c for c in df.columns if c == 'split' or c.startswith('fold_')]}"
        )

    # Create the four leaf folders up front.
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)

    stats = PrepareStats(fold_column=fold_column)

    for _, rec in df.iterrows():
        if rec["missing"] or rec["corrupted"]:
            stats.skipped_unreadable += 1
            continue
        yolo_split = SPLIT_TO_YOLO.get(rec[fold_column])
        if yolo_split is None:
            stats.skipped_unknown_split += 1
            continue

        name = rec["image_file"]
        stem = Path(name).stem

        _symlink(images_dir / name, out_dir / "images" / yolo_split / name)
        lines = yolo_label_lines(rec)
        label_path = out_dir / "labels" / yolo_split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")

        stats.labels_written += 1
        stats.total_boxes += len(lines)
        stats.images_by_split[yolo_split] = stats.images_by_split.get(yolo_split, 0) + 1

    write_data_yaml(out_dir)
    logger.info("YOLO dataset ready at %s (%s)", out_dir, stats.images_by_split)
    return stats


def write_data_yaml(out_dir: Path) -> Path:
    """Write the Ultralytics ``data.yaml`` describing the dataset.

    The ``path:`` key is intentionally omitted. Ultralytics resolves a missing
    ``path`` to the yaml file's own parent directory (``check_det_dataset`` ->
    ``path = ... or Path(data["yaml_file"]).parent``), so ``train``/``val``
    always resolve relative to wherever ``data.yaml`` physically lives. Writing
    an absolute ``path`` instead goes stale the moment the repo is moved to a
    different mount, which previously broke training.
    """
    out_dir = Path(out_dir)
    yaml_path = out_dir / "data.yaml"
    content = (
        f"# Auto-generated by acnevision.detection.prepare_yolo -- do not hand-edit.\n"
        f"# 'path' is omitted on purpose: Ultralytics then anchors train/val to\n"
        f"# this file's directory, so the dataset survives being moved/remounted.\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", yaml_path)
    return yaml_path
