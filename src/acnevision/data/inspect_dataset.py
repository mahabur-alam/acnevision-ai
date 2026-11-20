"""Inspect the raw ACNE04 Arrow shards: schema, columns and a sample record.

Use this to understand the dataset *before* converting it, so you never have to
guess column names or annotation structures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from .convert_arrow_to_jpg import IMAGE_COLUMN, find_shards

logger = logging.getLogger(__name__)


def inspect(raw_dir: Path) -> dict[str, Any]:
    """Return (and log) a small report describing the raw dataset.

    The report contains the feature schema, column names, the number of shards,
    the total number of records and a single sample record with the image
    field removed (so nothing huge is printed).
    """
    raw_dir = Path(raw_dir)
    shards = find_shards(raw_dir)

    first = Dataset.from_file(str(shards[0]))
    features = {name: str(feat) for name, feat in first.features.items()}
    columns = list(first.column_names)

    total_records = 0
    for shard in shards:
        total_records += Dataset.from_file(str(shard)).num_rows

    sample_view = (
        first.remove_columns([IMAGE_COLUMN])
        if IMAGE_COLUMN in columns
        else first
    )
    sample = sample_view[0] if sample_view.num_rows else {}

    report: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "num_shards": len(shards),
        "shards": [p.name for p in shards],
        "total_records": total_records,
        "columns": columns,
        "features": features,
        "sample_record": sample,
    }

    logger.info("Dataset: %s", raw_dir)
    logger.info("Shards (%d): %s", len(shards), ", ".join(report["shards"]))
    logger.info("Total records: %d", total_records)
    logger.info("Columns: %s", ", ".join(columns))
    logger.info("Features:")
    for name, feat in features.items():
        logger.info("  %-22s %s", name, feat)
    logger.info("Sample record (image field omitted):")
    logger.info(json.dumps(sample, indent=2, ensure_ascii=False, default=str))

    return report
