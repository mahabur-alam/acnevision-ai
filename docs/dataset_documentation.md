# ACNE04 Dataset Documentation

This document describes the dataset used by AcneVision AI, how it is stored in
this project, and its usage restrictions. **Read the license section before using
the data for anything.**

## Source

- **Hugging Face repackaging:** [`Layered-Labs/ACNE04`](https://huggingface.co/datasets/Layered-Labs/ACNE04)
- **Original dataset:** ACNE04, released by Xiaoping Wu et al. via the
  [`xpwu95/LDL`](https://github.com/xpwu95/LDL) repository.
- **Original paper (please cite):**
  Wu, Xiaoping; Wen, Ni; Liang, Jie; Lai, Yu-Kun; She, Dongyu; Cheng, Ming-Ming; Yang, Jufeng.
  *"Joint Acne Image Grading and Counting via Label Distribution Learning."*
  Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 10642–10651.

The Hugging Face card notes the images and labels come straight from the official
release and that nothing was re-annotated. When you publish work, cite both the
original paper and the Hugging Face repackaging (Ridwan, Hossain, Noor, 2026).

## License & usage restrictions

> **Academic / research use only — not for commercial use.**

The dataset's declared license is the custom identifier
**`acne04-academic-use-only`**. According to the dataset card:

- The underlying ACNE04 dataset is released by the original authors for
  **academic/research use only**.
- For any other use, you must contact the original author (Xiaoping Wu) per the
  terms in the official repository.
- The repackaging carries the same restriction: **do not use this dataset for
  commercial purposes**.
- **Intended use:** academic research and model benchmarking. It is **not**
  intended for clinical deployment or diagnostic use.

This project does **not** add, remove, or change any licensing terms. These
restrictions are reproduced here for convenience only; the dataset card and the
original repository are the authoritative sources.

## What the data contains

- **1,457** facial acne photographs (one `train` split).
- **18,983** lesion-level bounding-box annotations across all images.
- **4-level Hayashi severity grading**, with this distribution (matches our
  generated `dataset_summary.json`):

  | `level` | `severity`    | count |
  |--------:|---------------|------:|
  | 0       | `mild`        | 513   |
  | 1       | `moderate`    | 633   |
  | 2       | `severe`      | 182   |
  | 3       | `very_severe` | 129   |

- **Canonical split** (`split` column) = fold 0: **1,165 train / 292 test**.
- **Image sizes vary widely** (observed width 143–5184 px, height 155–3456 px);
  they are routine clinical photos, not standardized captures.

### Schema (verified from the Arrow shards)

| Column | Type | Meaning |
|--------|------|---------|
| `image` | image | The facial photo (removed from annotations; extracted to a JPG). |
| `file_name` | string | Original filename in the source release (e.g. `levle0_0.jpg`). |
| `level` | int32 | Severity grade 0–3. |
| `severity` | string | Severity label (`mild`/`moderate`/`severe`/`very_severe`). |
| `lesion_count_label` | int32 | Raw bucket label from the paper's files (not fully documented upstream). |
| `num_lesions_annotated` | int32 | Number of annotated lesion boxes for the image. |
| `lesion_boxes` | struct of lists | Parallel lists `xmin`, `ymin`, `xmax`, `ymax`; the i-th entry of each is one box (Pascal VOC pixel coords). |
| `split` | string | Canonical train/test assignment (= fold 0). |
| `fold_0_split` … `fold_4_split` | string | 5-fold cross-validation assignments (`train`/`test`). |

> **Caveat from the card:** the five test folds are *close to* but not a perfectly
> disjoint partition. Keep this in mind if you do cross-validation.

## How it is stored in this project

```
data/
├── raw/ACNE04/train/                 # the original .arrow shards (never edited)
│   ├── data-00000-of-00003.arrow
│   ├── data-00001-of-00003.arrow
│   └── data-00002-of-00003.arrow
└── processed/ACNE04/train/
    ├── images/                       # acne_00000.jpg … acne_01456.jpg
    ├── annotations.jsonl             # one JSON record per image (all fields preserved)
    └── dataset_summary.json          # aggregate statistics
```

Each `annotations.jsonl` record adds an `image_file` field (the generated JPG
name) and preserves **every** original annotation field, including the nested
`lesion_boxes` structure and all five CV folds. Record *i* corresponds to
`images/acne_{i:05d}.jpg`.

The whole `data/` folder is git-ignored — the dataset is never committed. Download
it from Hugging Face and regenerate the processed outputs locally (see the README).
