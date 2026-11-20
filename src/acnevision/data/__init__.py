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

Exploratory Data Analysis (read-only) helpers
----------------------------------------------
eda_data
    Load ``annotations.jsonl`` into tidy per-image / per-box DataFrames and attach
    real image sizes; split/fold tables.
eda_images
    Image counts, resolution, aspect-ratio, format and colour-channel analysis.
eda_severity
    Severity-class distribution and imbalance analysis.
eda_lesions
    Lesion-count statistics and count-field consistency checks.
eda_boxes
    Bounding-box width/height/area/aspect, normalized sizes, COCO size buckets and
    the 640/960/1024 resize simulation used to pick the YOLO input resolution.
eda_viz
    Matplotlib/seaborn charts and bounding-box overlay images.
"""
