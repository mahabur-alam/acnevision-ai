"""Load and reshape the processed ACNE04 dataset for Exploratory Data Analysis.

This module is the *central entry point* for every EDA notebook: it turns the
``annotations.jsonl`` file into tidy pandas DataFrames that the other ``eda_*``
modules analyse. Nothing here modifies the dataset on disk -- it only reads.

Three shapes of data are produced:

``load_dataframe``
    One row per **image** (1457 rows), with all annotation fields plus a derived
    ``num_boxes`` and an absolute ``image_path``.
``explode_boxes``
    One row per **lesion bounding box**, carrying its parent image's filename and
    severity. Images with zero boxes contribute no rows.
``attach_image_sizes``
    Adds the real ``width``/``height``/``mode``/``img_format`` of each JPG (read
    from the file header, so it is cheap) and flags ``missing`` / ``corrupted``
    images.

All functions are small, typed and documented so they are easy to follow while
learning. They reuse :func:`load_annotations` from :mod:`validate_dataset` instead
of re-parsing the JSONL by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PIL import Image

from .validate_dataset import load_annotations

logger = logging.getLogger(__name__)

# The four parallel coordinate lists stored inside each ``lesion_boxes`` struct.
BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")


def default_processed_dir(start: Path | None = None) -> Path:
    """Return the usual processed-data folder ``data/processed/ACNE04/train``.

    ``start`` lets a notebook pass its own location; by default we resolve the
    path relative to the repository root (two levels above ``src/acnevision``).
    """
    if start is not None:
        return Path(start)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "processed" / "ACNE04" / "train"


def _count_boxes(lesion_boxes: object) -> int:
    """Return how many boxes a ``lesion_boxes`` value holds (0 if empty/missing)."""
    if not isinstance(lesion_boxes, dict):
        return 0
    xmin = lesion_boxes.get("xmin")
    return len(xmin) if isinstance(xmin, list) else 0


def load_dataframe(processed_dir: Path | None = None) -> pd.DataFrame:
    """Load ``annotations.jsonl`` into one-row-per-image DataFrame.

    Parameters
    ----------
    processed_dir:
        Folder containing ``annotations.jsonl`` and ``images/``. Defaults to
        ``data/processed/ACNE04/train``.

    Returns
    -------
    pandas.DataFrame
        All original annotation columns, plus:
        ``num_boxes`` -- number of bounding boxes actually stored for the image,
        ``image_path`` -- absolute path to the JPG on disk.
    """
    processed_dir = default_processed_dir(processed_dir)
    annotations_path = processed_dir / "annotations.jsonl"
    images_dir = processed_dir / "images"

    records = load_annotations(annotations_path)
    df = pd.DataFrame(records)
    logger.info("Loaded %d annotation records from %s", len(df), annotations_path)

    # Derived helper columns (do not touch the original fields).
    df["num_boxes"] = df["lesion_boxes"].apply(_count_boxes)
    df["image_path"] = df["image_file"].apply(lambda name: str(images_dir / name))
    return df


def explode_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-lesion-box DataFrame from a per-image ``df``.

    Each output row carries the box corners (``xmin``/``ymin``/``xmax``/``ymax``)
    together with its parent ``image_file`` and ``severity`` so box statistics can
    be grouped by class. Images with no boxes simply contribute no rows.
    """
    rows: list[dict[str, object]] = []
    for _, rec in df.iterrows():
        boxes = rec["lesion_boxes"]
        if not isinstance(boxes, dict):
            continue
        xmins = boxes.get("xmin") or []
        ymins = boxes.get("ymin") or []
        xmaxs = boxes.get("xmax") or []
        ymaxs = boxes.get("ymax") or []
        # The four lists are parallel; zip stops at the shortest for safety.
        for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
            rows.append(
                {
                    "image_file": rec["image_file"],
                    "severity": rec["severity"],
                    "level": rec["level"],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

    boxes_df = pd.DataFrame(rows, columns=["image_file", "severity", "level",
                                           "xmin", "ymin", "xmax", "ymax"])
    logger.info("Exploded %d images into %d individual lesion boxes",
                len(df), len(boxes_df))
    return boxes_df


def _read_header(path: Path) -> dict[str, object]:
    """Read width/height/mode/format from a JPG header, flagging problems.

    Only the header is read (no full pixel decode), so this stays fast even for
    the very large ACNE04 images. Missing or unreadable files are reported via
    the ``missing`` / ``corrupted`` flags instead of raising.
    """
    if not path.exists():
        return {"width": None, "height": None, "mode": None,
                "img_format": None, "missing": True, "corrupted": False}
    try:
        with Image.open(path) as im:
            return {"width": im.width, "height": im.height, "mode": im.mode,
                    "img_format": im.format, "missing": False, "corrupted": False}
    except Exception as exc:  # noqa: BLE001 - never let one bad file stop EDA
        logger.warning("Could not read image header for %s: %s", path.name, exc)
        return {"width": None, "height": None, "mode": None,
                "img_format": None, "missing": False, "corrupted": True}


def attach_image_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with real image dimensions and quality flags.

    Adds columns: ``width``, ``height``, ``mode`` (e.g. ``RGB``), ``img_format``
    (e.g. ``JPEG``), ``missing`` (file absent) and ``corrupted`` (file present but
    unreadable). Also adds ``aspect_ratio`` = width / height for readable images.
    """
    headers = [_read_header(Path(p)) for p in df["image_path"]]
    header_df = pd.DataFrame(headers, index=df.index)
    out = pd.concat([df, header_df], axis=1)

    # Aspect ratio only where both dimensions are known and non-zero.
    out["aspect_ratio"] = out.apply(
        lambda r: (r["width"] / r["height"])
        if r["width"] and r["height"] else None,
        axis=1,
    )
    n_missing = int(out["missing"].sum())
    n_corrupt = int(out["corrupted"].sum())
    logger.info("Image headers read: %d missing, %d corrupted, %d ok",
                n_missing, n_corrupt, len(out) - n_missing - n_corrupt)
    return out


def split_columns(df: pd.DataFrame) -> list[str]:
    """Return the split-defining columns present: ``split`` + any ``fold_*``."""
    cols = [c for c in df.columns if c == "split" or c.startswith("fold_")]
    return sorted(cols)


def split_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return the train/test counts and percentages for one split column.

    The original assignments are only *read* here -- never reassigned.
    """
    counts = df[column].value_counts().rename_axis(column)
    total = int(counts.sum())
    table = counts.reset_index(name="count")
    table["percentage"] = (table["count"] / total * 100).round(2) if total else 0.0
    return table


def all_split_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide table of train/test counts for every split column.

    One row per split column (``split``, ``fold_0_split`` ...), with the count for
    each label value (typically ``train`` / ``test``) as columns.
    """
    rows: list[dict[str, object]] = []
    for col in split_columns(df):
        counts = df[col].value_counts().to_dict()
        row: dict[str, object] = {"split_column": col}
        row.update({str(k): int(v) for k, v in counts.items()})
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)
