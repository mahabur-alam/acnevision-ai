"""Dataset preparation for AcneVision AI.

Modules
-------
convert_arrow_to_jpg
    Convert the ACNE04 ``.arrow`` shards into JPG images plus an
    ``annotations.jsonl`` file and a ``dataset_summary.json`` report.
inspect_dataset
    Print the schema, columns and a sample record of the raw Arrow shards.
validate_dataset
    Check that every annotation points at an existing, uniquely named image.
"""
