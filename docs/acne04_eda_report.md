# ACNE04 — Exploratory Data Analysis (EDA) Report

**Project:** AcneVision AI · **Dataset:** ACNE04 (processed `train` split)
**Date:** 2026-11-20 · **Scope:** read-only EDA before YOLO training (no training, no synthetic annotations)

This report is generated from the reusable helpers in
[`src/acnevision/data/eda_*.py`](../src/acnevision/data) and the notebooks in
[`notebooks/`](../notebooks). All numbers below come from the live dataset
(`data/processed/ACNE04/train`). Figures are written to
[`outputs/figures/`](../outputs/figures) (that folder is git-ignored, so regenerate
them by running the notebooks or `src/acnevision/data/eda_viz.py` helpers).

---

## 1. Dataset overview

| Item | Value |
|---|---|
| Total images | **1457** (JPEG, RGB) |
| Total annotations | 1457 (1 record per image, 1:1 mapping verified) |
| Severity classes | 4 — mild, moderate, severe, very_severe |
| Total annotated lesions | **18,983** bounding boxes |
| Default split | train 1165 / test 292 (≈80/20) |
| Cross-validation | 5 folds (`fold_0_split` … `fold_4_split`), each 1165/292 |
| Box format | pixel `xmin/ymin/xmax/ymax` on the **original** image |

Per-record fields: `image_file`, `file_name`, `level` (0–3), `severity`,
`lesion_count_label`, `num_lesions_annotated`, `lesion_boxes` (parallel coord
lists), `split`, `fold_0_split`…`fold_4_split`.

**Integrity:** `validate()` passes — every annotation maps to an existing, uniquely
named image; no duplicates, no orphan images.

---

## 2. Image quality findings

| Metric | Result |
|---|---|
| Missing images | **0** |
| Corrupted images | **0** |
| File format | 100% JPEG |
| Colour mode | 100% RGB (3 channels) |
| Width (px) | min 143 · median 3112 · mean 2918 · max 5184 |
| Height (px) | min 155 · median 3456 · mean 3028 · max 3456 |
| Aspect ratio (w/h) | min 0.56 · median 0.90 · mean 0.97 · max 1.78 |

Images are clean (no decode failures, no channel surprises) but **highly
heterogeneous in size** — from tiny (143 px) to very large (5184 px) — and mostly
**portrait** (median aspect 0.90). Figures:
`image_resolution_scatter.png`, `image_aspect_ratio_hist.png`.

**Implication:** every image must be resized/letterboxed to a fixed input size for
training; the wide size range makes that choice (Section 6) consequential.

---

## 3. Severity distribution

| Severity | Count | % | Imbalance ratio* |
|---|---:|---:|---:|
| mild | 513 | 35.2% | 1.23× |
| moderate | 633 | 43.5% | 1.00× (majority) |
| severe | 182 | 12.5% | 3.48× |
| very_severe | 129 | 8.9% | 4.91× |

\* majority count ÷ class count. Figure: `severity_distribution_bar.png`.

- **Moderate + mild make up ~79%**; severe and very_severe together only ~21%.
- Max imbalance ≈ **4.9×** (very_severe vs moderate).
- `level` ↔ `severity` are **perfectly consistent** (0 mismatches).

**Implication:** the two minority classes need class weighting or targeted
augmentation; do not let the model collapse toward mild/moderate.

---

## 4. Lesion distribution

| Metric | Value |
|---|---:|
| Total lesions | 18,983 |
| Mean per image | 13.0 |
| Median per image | 7 |
| Min / Max per image | 1 / 65 |
| Images with 0 lesions | 0 |

Lesion count rises cleanly with severity (it is essentially the grading criterion):

| Severity | mean | median | min–max |
|---|---:|---:|---:|
| mild | 2.1 | 2 | 1–5 |
| moderate | 9.0 | 8 | 6–20 |
| severe | 26.1 | 25 | 21–42 |
| very_severe | 58.0 | 58 | 51–65 |

Figure: `lesions_per_image_hist.png`. The most crowded faces (all very_severe) carry
up to 65 boxes.

**Count-field consistency:** `num_boxes` (stored boxes) equals
`num_lesions_annotated` for **every** image (0 mismatches) — the detection labels
are internally consistent. In **21** very_severe images the separate
`lesion_count_label` differs by a few units; this is ACNE04's label-distribution
*counting* target, not the box count, and should **not** be used as the number of
detection boxes. See Section 8.

---

## 5. Bounding-box size analysis

At **original** resolution:

| Metric | min | median | mean | max |
|---|---:|---:|---:|---:|
| Box width (px) | 4 | 93 | 93.2 | 514 |
| Box height (px) | 4 | 99 | 93.1 | 491 |
| Box area (px²) | 16 | 10,332 | 12,294 | 241,081 |
| Box aspect (w/h) | 0.22 | 1.00 | 1.05 | 4.69 |
| Normalized width | 0.002 | **0.044** | 0.047 | 0.188 |
| Normalized height | 0.003 | **0.042** | 0.044 | 0.150 |

COCO size buckets on original pixel area: **small 27.1% · medium 20.8% · large
52.1%**. Boxes are roughly square (median aspect 1.0) and **small relative to the
image** — a typical lesion spans only ~4.4% of image width. No invalid or
out-of-bounds boxes were found. Figure: `box_size_hist.png`.

The absolute pixel sizes look large only because the source images are huge; what
matters for a detector is size **after** resizing (Section 6).

---

## 6. Resolution recommendation (is 640 enough?)

Letterbox-resizing every image (scale = `target / max(width, height)`) and scaling
the boxes gives:

| Input px | Median lesion side (px) | COCO *small* % | *medium* % | % tiny (<8px) |
|---:|---:|---:|---:|---:|
| **640** | 22.5 | **82.5%** | 17.5% | 0.84% |
| **960** | 33.8 | 29.8% | 70.2% | 0.14% |
| **1024** | 36.0 | 22.6% | 77.3% | 0.13% |

Figure: `resize_simulation_tiny.png`.

- At **640**, lesions remain *visible* (median ~22 px, <1% drop below 8 px) but
  **~82% land in the COCO *small* regime**, where detector recall is weakest —
  especially for mild faces with only 1–5 small lesions.
- At **960**, the small-object share falls to **~30%** and the median lesion moves
  into the *medium* range — a large gain for modest extra CPU cost.
- **1024** improves only marginally over 960.

**Recommendation: train at `imgsz = 960`.** 640 is workable for a fast baseline but
will under-detect small lesions; 1024 is a fallback only if minority-class / small
lesion recall is still short. On CPU, 960 is the best accuracy/latency balance.

---

## 7. Dataset split information

| Split column | train | test |
|---|---:|---:|
| `split` (default) | 1165 | 292 |
| `fold_0_split` | 1165 | 292 |
| `fold_1_split` | 1165 | 292 |
| `fold_2_split` | 1165 | 292 |
| `fold_3_split` | 1165 | 292 |
| `fold_4_split` | 1165 | 292 |

Every column is a clean ~80/20 partition, and the severity mix is broadly preserved
across train/test. The five folds support cross-validation. **These assignments are
reported only — never modified.** Use `split` (= `fold_0_split`) for a single
train/test run, or the five folds for CV.

---

## 8. Potential data-quality problems

1. **Severity class imbalance** (very_severe 8.9%, severe 12.5%) — bias risk toward
   mild/moderate. *Mitigation:* class weights, focal loss, or minority augmentation.
2. **Extreme image-size variance** (143 → 5184 px) — forces a fixed resize; some
   tiny source images will be upscaled. *Mitigation:* letterbox to a fixed `imgsz`.
3. **Small objects after downscaling** — ~82% of lesions become COCO-*small* at 640.
   *Mitigation:* higher `imgsz` (Section 6), possibly a P2 detection head.
4. **`lesion_count_label` ≠ box count in 21 very_severe images** — the counting
   target differs from the number of drawn boxes. *Mitigation:* for **detection**,
   use `lesion_boxes` / `num_lesions_annotated` (always consistent); keep
   `lesion_count_label` only for a separate counting/grading task.
5. **Very crowded faces (up to 65 lesions)** — many overlapping small boxes stress
   NMS and small-object recall. *Mitigation:* higher `imgsz`, tuned NMS IoU,
   mosaic/copy-paste augmentation.

No corrupted images, no missing files, no invalid/out-of-bounds boxes were found.

---

## 9. Recommendations before YOLO training

1. **Input size:** start at **`imgsz = 960`** (letterboxed). Keep 640 only for a
   quick smoke-test baseline; try 1024 only if small-lesion recall lags.
2. **Handle imbalance** via class weights / focal loss and stronger augmentation for
   severe + very_severe, not by inventing labels.
3. **Build YOLO labels from `lesion_boxes`** (convert `xmin/ymin/xmax/ymax` →
   normalized `xc/yc/w/h`), using `num_lesions_annotated` as the per-image count.
   Ignore `lesion_count_label` for detection.
4. **Single detection class** (`lesion`). Severity grading is better framed as a
   separate per-image task (classification or count-threshold), since very_severe
   boxes are capped near 65.
5. **Validate with the provided folds** — report 5-fold CV, don't invent new splits.
6. **Augmentation:** mosaic, HSV jitter, small rotations/flips; avoid heavy geometric
   distortion that would move lesions off real skin.

---

## 10. Recommended YOLO model

Constraints: **CPU-only**, tiny/numerous lesions, class imbalance, 1457 images,
beginner-friendly.

- **Framework — Ultralytics YOLO11.** Current, best accuracy-per-parameter in the
  family, simplest beginner API/docs, and a clean CPU/ONNX export path. (YOLOv8 is a
  fine fallback with the same API.)
- **Size — start with `YOLO11s` (≈9–10M params).** Best accuracy/speed balance that
  is still realistic on CPU. **Drop to `YOLO11n`** if training or inference is too
  slow. **Avoid m/l/x** — impractical on CPU for this dataset.
- **Input resolution — `imgsz = 960`** (from Section 6). This is the single most
  important setting for acne lesions; 640 under-detects small ones.
- **Small-object head (optional):** a **P2 head** variant (adds a higher-resolution
  detection layer) can help the smallest lesions, at extra CPU cost — try it only if
  960 still misses small lesions.
- **Task framing:** per-lesion *detection* is the expensive CPU case. If the
  near-term goal is **severity grading**, a **`YOLO11-cls`** classifier (or mapping
  lesion count → severity with the known thresholds in Section 4) is far cheaper and
  more robust. Consider shipping the classifier first, detection second.

**Bottom line:** **YOLO11s at `imgsz=960`, single `lesion` class, trained with the
provided 5 folds and class-imbalance handling** — fall back to YOLO11n if CPU speed
demands it, and reach for a P2 head or 1024 only if small-lesion recall is short.
