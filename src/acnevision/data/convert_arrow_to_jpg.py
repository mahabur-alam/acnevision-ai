"""Convert ACNE04 Arrow shards into JPG images + preserved annotations.

This is the refactored replacement for the original ``extract_images.py``.

What it does differently from the original script
-------------------------------------------------
* Processes **one shard at a time** instead of concatenating every shard into
  memory at once (much lower peak memory).
* **Preserves every annotation field** from the dataset (not only the image):
  the original filename, severity, level, lesion counts, lesion bounding boxes,
  the train/test split and all 5 cross-validation fold assignments.
* Is **idempotent**: images that already exist are skipped and are *not*
  re-decoded from Arrow (pass ``force=True`` to re-extract them).
* Reports progress and a clear summary of extracted / skipped / failed records.
* Handles missing or corrupted images gracefully instead of crashing.

Outputs (under ``out_dir``, default ``data/processed/ACNE04/train``)
    images/acne_00000.jpg, acne_00001.jpg, ...   one JPG per record
    annotations.jsonl                            one JSON record per image
    dataset_summary.json                         aggregate statistics

The filename for record *i* (counting across all sorted shards, in order) is
``acne_{i:05d}.jpg``. Because the shards are always processed in sorted order
and rows in their natural order, these indices are stable and reproduce the
filenames produced by the original script.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# The column that holds the PIL image; everything else is an annotation field.
IMAGE_COLUMN = "image"

# Shard files look like ``data-00000-of-00003.arrow``.
SHARD_GLOB = "data-*-of-*.arrow"


@dataclass
class ConversionStats:
    """Aggregate numbers collected while converting the dataset."""

    shards: list[str] = field(default_factory=list)
    total_records: int = 0
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    # image_file -> reason, for records whose image could not be written.
    failures: dict[str, str] = field(default_factory=dict)
    severity_distribution: Counter = field(default_factory=Counter)
    split_distribution: Counter = field(default_factory=Counter)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


def image_filename(index: int) -> str:
    """Return the deterministic JPG filename for a global record index.

    >>> image_filename(0)
    'acne_00000.jpg'
    >>> image_filename(1456)
    'acne_01456.jpg'
    """
    return f"acne_{index:05d}.jpg"


def find_shards(raw_dir: Path) -> list[Path]:
    """Find all ``.arrow`` shards in ``raw_dir``, sorted for a stable order.

    Sorting guarantees the global record order is deterministic across runs,
    which is what keeps the generated image filenames stable.
    """
    raw_dir = Path(raw_dir)
    shards = sorted(raw_dir.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"No Arrow shards matching '{SHARD_GLOB}' found in {raw_dir}. "
            "Did you download the dataset and point --raw-dir at the right folder?"
        )
    return shards


def _metadata_view(ds: Dataset) -> Dataset:
    """Return a view of ``ds`` without the image column.

    Reading rows from this view does not decode any image, so building
    annotations and checking which files already exist stays cheap.
    """
    if IMAGE_COLUMN in ds.column_names:
        return ds.remove_columns([IMAGE_COLUMN])
    return ds


def build_annotation(record: dict[str, Any], image_file: str) -> dict[str, Any]:
    """Build one annotation record: the generated filename + all dataset fields.

    ``record`` is a single row with the image column already removed, so every
    remaining key is a genuine annotation field and is preserved verbatim
    (including the nested ``lesion_boxes`` structure).
    """
    annotation: dict[str, Any] = {"image_file": image_file}
    annotation.update(record)
    return annotation


def _json_default(obj: Any) -> Any:
    """Fallback serializer for numpy scalars/arrays that ``json`` can't handle."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` of a JPG by reading only its header."""
    try:
        with Image.open(path) as im:
            return im.size  # (width, height); header-only, no full decode
    except Exception as exc:  # noqa: BLE001 - never let stats-gathering crash the run
        logger.warning("Could not read image size for %s: %s", path.name, exc)
        return None


def convert_all_arrow_to_jpg(
    raw_dir: Path,
    out_dir: Path,
    quality: int = 95,
    force: bool = False,
) -> ConversionStats:
    """Convert every Arrow shard in ``raw_dir`` into JPGs + ``annotations.jsonl``.

    Parameters
    ----------
    raw_dir:
        Folder holding the ``data-*-of-*.arrow`` shards.
    out_dir:
        Destination folder; ``images/`` and ``annotations.jsonl`` are written
        inside it.
    quality:
        JPG quality (1-95) for newly written images.
    force:
        Re-decode and overwrite images that already exist. Default ``False``
        skips existing images (idempotent, fast re-runs).

    Returns
    -------
    ConversionStats
        Counts and distributions describing the run.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.jsonl"

    shards = find_shards(raw_dir)
    stats = ConversionStats(shards=[p.name for p in shards])

    logger.info("Found %d shard(s) in %s", len(shards), raw_dir)

    global_index = 0
    # Rewrite the annotations file from scratch so it always matches the images.
    with annotations_path.open("w", encoding="utf-8") as ann_file:
        for shard in shards:
            logger.info("Loading shard %s ...", shard.name)
            ds = Dataset.from_file(str(shard))
            if not stats.columns:
                stats.columns = list(ds.column_names)
            meta = _metadata_view(ds)
            n = meta.num_rows
            logger.info("  %s: %d record(s)", shard.name, n)

            for local_i in range(n):
                record = meta[local_i]  # dict without the image -> cheap
                image_file = image_filename(global_index)
                out_path = images_dir / image_file

                size: tuple[int, int] | None = None
                if out_path.exists() and not force:
                    stats.skipped += 1
                    size = _image_size(out_path)
                else:
                    try:
                        image = ds[local_i][IMAGE_COLUMN]  # decodes this row only
                        if image is None:
                            raise ValueError("image field is empty/null")
                        rgb = image.convert("RGB")
                        rgb.save(out_path, format="JPEG", quality=quality)
                        size = rgb.size
                        stats.extracted += 1
                    except Exception as exc:  # noqa: BLE001 - keep going on bad rows
                        stats.failed += 1
                        stats.failures[image_file] = str(exc)
                        logger.warning(
                            "  FAILED to extract record %d (%s): %s",
                            global_index,
                            image_file,
                            exc,
                        )

                # Always record the annotation so every image maps to one record.
                annotation = build_annotation(record, image_file)
                ann_file.write(
                    json.dumps(annotation, ensure_ascii=False, default=_json_default)
                    + "\n"
                )

                # Distributions / dimension stats (only when values are present).
                severity = record.get("severity")
                if severity is not None:
                    stats.severity_distribution[severity] += 1
                split = record.get("split")
                if split is not None:
                    stats.split_distribution[split] += 1
                if size is not None:
                    stats.widths.append(size[0])
                    stats.heights.append(size[1])

                global_index += 1
                if global_index % 200 == 0:
                    logger.info(
                        "  progress: %d processed (%d extracted, %d skipped, %d failed)",
                        global_index,
                        stats.extracted,
                        stats.skipped,
                        stats.failed,
                    )

            # Free the shard before loading the next one.
            del ds, meta

    stats.total_records = global_index
    logger.info(
        "Done: %d records -> %d extracted, %d skipped, %d failed",
        stats.total_records,
        stats.extracted,
        stats.skipped,
        stats.failed,
    )
    logger.info("Annotations written to %s", annotations_path)
    return stats


def _dimension_stats(values: list[int]) -> dict[str, float] | None:
    """Return min/max/mean for a list of pixel sizes, or ``None`` if empty."""
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "count": len(values),
    }


def write_summary(stats: ConversionStats, out_dir: Path) -> Path:
    """Write ``dataset_summary.json`` from the collected stats.

    Only values that were actually observed are written; nothing is invented.
    ``missing_annotations`` counts annotation records whose image file could not
    be written (extraction failures).
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / "dataset_summary.json"

    summary: dict[str, Any] = {
        "dataset": "ACNE04",
        "split": "train",
        "num_shards_processed": len(stats.shards),
        "shards": stats.shards,
        "total_records": stats.total_records,
        "images_extracted": stats.extracted,
        "images_skipped_existing": stats.skipped,
        "images_failed": stats.failed,
        "missing_annotations": stats.failed,
        "available_columns": stats.columns,
        "severity_distribution": dict(sorted(stats.severity_distribution.items())),
        "split_distribution": dict(sorted(stats.split_distribution.items())),
        "image_width_stats": _dimension_stats(stats.widths),
        "image_height_stats": _dimension_stats(stats.heights),
    }
    if stats.failures:
        summary["failures"] = stats.failures

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Summary written to %s", summary_path)
    return summary_path
