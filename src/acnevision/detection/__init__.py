"""AcneVision AI — lesion detection.

Modules
-------
prepare_yolo
    Convert ACNE04 ``lesion_boxes`` into a YOLO dataset (labels + data.yaml).
train
    Train a YOLO11 lesion detector on CPU (EDA defaults: imgsz=960).
evaluate
    Validate trained weights and report detection metrics.
"""
