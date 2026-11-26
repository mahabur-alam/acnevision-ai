# Pre-YOLO Dataset Readiness

**Project:** AcneVision AI · **Dataset:** ACNE04 (processed `train` split)
**Date:** 2026-11-26 · **Scope:** confirm the dataset steps required *before*
implementing/training the YOLO model.

## Summary

The pre-YOLO pipeline is organized into three phases: **(1) Dataset
Understanding, (2) Data Quality & Annotation, (3) Dataset Preparation.** All
three are **complete and already executed** in this repository. The only work
remaining before training is a one-line stale-path fix in the generated
`data.yaml` (see [Blocker](#blocker-before-training)); the model itself is not
trained yet (`models/trained/` is empty).

---

## Phase 1 — Dataset Understanding ✅ complete

| Step | Implemented by |
|---|---|
| Load ACNE04 | `scripts/convert_acne04.py` → `src/acnevision/data/convert_arrow_to_jpg.py` (Arrow shards → 1457 JPGs) |
| Understand all columns | `src/acnevision/data/inspect_dataset.py` (`--inspect`); schema in `docs/dataset_documentation.md` |
| Basic EDA | `src/acnevision/data/eda_*.py`, notebooks `01`–`03`; full write-up in `docs/acne04_eda_report.md` |
| Visualize sample images | `notebooks/04_dataset_visualization.ipynb`, `src/acnevision/data/eda_viz.py`, figures in `outputs/figures/` |

---

## Phase 2 — Data Quality & Annotation ✅ complete

| Step | Result |
|---|---|
| Image quality inspection | `eda_images.py` — 0 missing, 0 corrupted, 100% RGB JPEG (EDA report §2) |
| Duplicate detection | `src/acnevision/data/validate_dataset.py` (`--validate`) — no duplicates, no orphans, verified 1:1 image↔annotation map |
| Bounding box validation | `eda_boxes.py` — no invalid or out-of-bounds boxes (EDA report §5) |
| Box-size analysis | `eda_boxes.py` — COCO size buckets + 640-vs-960 resize simulation (EDA report §5–6) |

---

## Phase 3 — Dataset Preparation ✅ complete

| Step | Output |
|---|---|
| Official split preservation | split/fold columns kept read-only; `test` rows → YOLO `val` (`src/acnevision/detection/prepare_yolo.py`) |
| Train/validation split | 1165 train / 292 val (default `split`; `fold_0_split`…`fold_4_split` for 5-fold CV) |
| YOLO label conversion | `lesion_boxes` (`xmin/ymin/xmax/ymax`) → normalized `0 xc yc w h`; 1165 + 292 `.txt` label files, 18,983 boxes total |
| Metadata & YAML creation | `data/processed/ACNE04/yolo/data.yaml` (`nc: 1`, `names: ['lesion']`), plus `annotations.jsonl` and `dataset_summary.json` |

---

## Blocker before training

The Phase 3 output is built, but the generated
`data/processed/ACNE04/yolo/data.yaml` has a **stale absolute `path`**:

```
path: /media/mahabur/...c6e01/projects/acnevision-ai/data/processed/ACNE04/yolo
                        ^^^^^ wrong mount
```

The project now lives on mount **`…c6e02`**, not `…c6e01`. The `…c6e01` mount
exists but does not contain the project, so the path does not resolve — and the
`images/{train,val}` symlinks point at the same dead location. Ultralytics will
fail with a "dataset images not found / val set missing" error the moment
training starts.

**Root cause:** `prepare_yolo.py` writes an absolute `path:` into `data.yaml`,
which went stale after the repo was moved to a different partition.

**Fix:** regenerate from the current location (idempotent — rebuilds `data.yaml`
and the symlinks):

```bash
python scripts/prepare_yolo_dataset.py
```

Optionally, change `prepare_yolo.py` to emit a `path` relative to the yaml file
so future moves don't break it.

---

## Verification

```bash
cd /media/mahabur/af33425f-321f-4960-a4d5-f823d206c6e02/projects/acnevision-ai

# 1. Regenerate and confirm the path is correct
python scripts/prepare_yolo_dataset.py
grep '^path:' data/processed/ACNE04/yolo/data.yaml        # must show ...c6e02

# 2. Confirm images resolve (not broken symlinks)
ls -lL data/processed/ACNE04/yolo/images/val | head

# 3. Re-validate integrity (read-only)
python scripts/convert_acne04.py --validate

# 4. Smoke-test training wiring (fast; proves the dataset loads)
python scripts/train_detector.py --model yolo11n.pt --imgsz 160 --epochs 2 --batch 2
```

---

## Bottom line

- **Phases 1, 2, 3 are all complete** — your pre-YOLO checklist is fully
  implemented and executed.
- Before training, **fix one thing:** regenerate `data.yaml` so its `path`
  points at the current mount (`…c6e02`), then smoke-test.
- The real next step after that is **YOLO training itself** (`YOLO11s @ imgsz
  960`, per `docs/acne04_eda_report.md` §6/§10) — `models/trained/` is still empty.
