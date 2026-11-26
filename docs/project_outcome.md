# AcneVision AI — Expected Final Outcome

A CPU-only computer-vision system that, given a face photo, **detects the face,
finds acne lesions, grades severity, removes the acne, and shows a before/after
comparison** — with honest, reproducible evaluation.

## What "done" looks like

1. **Face detection** — locate the face in the image, flag/reject non-face inputs,
   and focus acne detection on the skin region (lesion boxes are mapped back to
   original-image coordinates).
2. **Acne detection** — a YOLO11 detector trained on ACNE04 returns lesion
   bounding boxes.
3. **Severity grading** — classify into *mild / moderate / severe / very_severe*
   using the Hayashi lesion-count thresholds, plus a raw lesion count and a 0–100
   severity index.
4. **Acne removal** — inpaint the detected lesion regions to produce a
   clean-skin image.
5. **Before/after comparison** — the original image is kept untouched, the cleaned
   copy is saved as a new file, and a side-by-side
   `original | annotated | restored` image is generated with a severity caption.

## Deliverables

- Trained detector weights in `models/trained/` with a **results table**
  (mAP@50, mAP@50-95, per-severity accuracy) reported across the 5 cross-validation
  folds (mean ± std).
- `scripts/run_pipeline.py` producing `<name>_restored.jpg` and
  `<name>_compare.jpg` in `outputs/predictions/`.
- A small **Gradio/Streamlit demo** (upload a face → see detected acne, severity,
  and a before/after slider).
- A results notebook (`notebooks/05_results.ipynb`) rendering the metrics table,
  a severity confusion matrix, a predicted-vs-true count-calibration plot, and a
  grid of before/after examples.
- Updated `README.md` (hero before/after image + results table) and
  `docs/model_card.md`.
- A passing `pytest` suite with CI, and a `pyproject.toml` for clean installs
  (`pip install -e .`).

## Quality & integrity

- Cross-validated metrics (not a single split), a count-calibration analysis, and
  an informal skin-tone fairness check on a sample of faces.
- Deterministic seeds and logged run commands/args for reproducibility.
- A clear limitations and ethics note: this is a **research / education tool, not a
  medical device**, and ACNE04 is for **academic / non-commercial use only**.

## Non-goals

- Not a diagnostic or medical product.
- Not real-time or mobile-optimized.
- Not for commercial use (dataset license).
