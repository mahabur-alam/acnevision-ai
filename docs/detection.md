# AcneVision — Detection, Grading & Restoration Pipeline

This document describes the end-to-end pipeline added on top of the ACNE04 dataset
prep + EDA. Everything runs on **CPU only**. Design decisions trace back to
[acne04_eda_report.md](acne04_eda_report.md).

## Overview

```
face image ──▶ [Detect lesions]  ──▶ boxes ──▶ [Count] ──▶ [Grade severity]
   (YOLO11, Phase B)                              (Phase C, Hayashi thresholds)
        │
        └──▶ [Restore] inpaint lesions (Phase D, optional)  ──▶ clean-skin image
```

## Phase A — Build the YOLO dataset

`src/acnevision/detection/prepare_yolo.py` (CLI: `scripts/prepare_yolo_dataset.py`)

- Converts each `lesion_boxes` entry (pixel `xmin/ymin/xmax/ymax`) to normalized
  YOLO labels `0 xc yc w h`, using **each image's own width/height**.
- Lays out the Ultralytics structure under `data/processed/ACNE04/yolo/`
  (`images/{train,val}` via **symlinks** — images are never duplicated —
  `labels/{train,val}`, and `data.yaml` with `nc: 1`, `names: [lesion]`).
- Split comes from an existing column (`--fold split` by default; `fold_0_split`…
  `fold_4_split` for cross-validation). `test` rows become YOLO's `val`.
- Uses `lesion_boxes` / `num_lesions_annotated` only — **not** `lesion_count_label`
  (the EDA found the latter disagrees on 21 very_severe images).

Verified: 1457 label files, 18,983 boxes, train 1165 / val 292, all coords in [0,1].

## Phase B — Train the detector

`src/acnevision/detection/{train,evaluate}.py` (CLI: `scripts/train_detector.py`)

- **Model: YOLO11s** (`yolo11s.pt`), the EDA recommendation; fall back to
  **YOLO11n** on slow machines (`--model yolo11n.pt`).
- **`imgsz = 960`** — the key EDA finding: at 640 px ~82% of lesions fall in the
  COCO-*small* regime; 960 roughly halves that. `device = cpu` is forced.
- Single `lesion` class; conservative augmentation (mosaic, HSV, small flips/rotates)
  from `acnevision.preprocessing.transforms.DEFAULT_TRAIN_ARGS`.
- Evaluation reports mAP@50, mAP@50-95, precision, recall. **Watch small-object
  recall** specifically.

CPU note: a full `YOLO11s @ 960` run is slow. For a quick wiring check use e.g.
`--model yolo11n.pt --imgsz 160 --epochs 2 --batch 2`. For final metrics, optionally
repeat across the 5 folds.

## Phase C — Severity grading

`src/acnevision/classification/severity.py`

Training-free **Hayashi count thresholds**: `≤5 mild · 6–20 moderate · 21–50 severe
· ≥51 very_severe`. This reproduces ACNE04's own severity labels from the annotated
counts with **100% accuracy** (1457/1457), so it is the recommended baseline. The
empty 43–50 band in the data belongs to *severe*. A learned `YOLO11-cls` classifier
(with class weights for the ~4.9× imbalance) is an optional alternative.

## Phase D — Blemish removal (restoration)

`src/acnevision/restoration/inpaint.py`

Builds a mask over the detected lesion boxes (slightly dilated) and fills them with
OpenCV `cv2.inpaint` (`telea` default, or `ns`). Classical, CPU-friendly, no
training. The interface leaves room to swap in a learned inpainter later.

## Phase E — End-to-end

`src/acnevision/pipeline.py` (CLI: `scripts/run_pipeline.py`)

`AcnePipeline(weights).run(image, restore=True)` → `AcneResult` with the boxes,
lesion count, severity + level, and an optional restored image.

```bash
python scripts/run_pipeline.py --image face.jpg \
    --weights models/trained/acne_yolo11s_960/weights/best.pt --restore \
    --out outputs/predictions/face_restored.jpg
```

## Guardrails
- CPU only — install the CPU torch wheel, never CUDA.
- The source dataset/annotations are read-only; labels are derived, never
  hand-edited, and the original split/fold assignments are reused unchanged.
