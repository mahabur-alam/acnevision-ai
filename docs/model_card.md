# Model Card — AcneVision Lesion Detector

## Model details
- **Task:** facial acne lesion detection (single class `lesion`), with downstream
  severity grading and optional blemish removal.
- **Architecture:** Ultralytics **YOLO11s** (fallback: YOLO11n), fine-tuned from
  COCO-pretrained weights.
- **Input:** RGB image, letterboxed to **960×960** (see EDA resolution analysis).
- **Compute:** **CPU only** (PyTorch CPU build). No GPU is used anywhere.
- **Code:** `src/acnevision/detection/`, `scripts/train_detector.py`.

## Training data
- **ACNE04** (processed `train` split): 1,457 face images, 18,983 lesion boxes.
- Labels derived from `lesion_boxes`; split from the dataset's own `split` column
  (train 1165 / val 292), with 5 CV folds available.
- **License: academic / research use only — not for commercial use.**

## Intended use
- Research and education on acne detection/grading; demos; a baseline to improve on.
- **Not** a medical device; not for diagnosis or treatment decisions.

## Severity grading
- Rule-based **Hayashi thresholds** on detected lesion count
  (`≤5 mild · 6–20 moderate · 21–50 severe · ≥51 very_severe`). Reproduces the
  dataset's labels with 100% accuracy on annotated counts.

## Limitations & risks
- **Small objects:** most lesions are small after resizing; recall on faint/small
  lesions is the main risk (mitigated by `imgsz=960`, possibly a P2 head / 1024).
- **Class imbalance:** severe (12%) and very_severe (9%) are under-represented;
  handled with class weights / augmentation, not synthetic labels.
- **Population bias:** ACNE04's demographics may not generalize to all skin tones /
  lighting; evaluate before any real-world use.
- **Detection count ≠ clinical count** in very dense faces (annotations cap near 65).
- **Restoration** is cosmetic inpainting, not dermatological correction.

## Evaluation
- Metrics: mAP@50, mAP@50-95, precision, recall (via `scripts/train_detector.py`),
  plus per-size recall inspection. Use the provided folds for cross-validation.

## Ethical notes
- Faces are personal data; keep usage within the research-only license and avoid
  identifying or profiling individuals.
