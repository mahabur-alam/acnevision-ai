# AcneVision AI

A computer-vision project built on the **ACNE04** facial-acne dataset. This stage
of the project establishes a clean, reliable **dataset-preparation pipeline** and a
scalable project structure for future work. Everything runs on **CPU — no GPU is
required**.

> **Status:** dataset preparation only. Detection, classification, and image
> restoration are **not implemented yet** — they are planned (see the roadmap).

## Objectives

- Convert the raw ACNE04 Arrow shards into plain JPG images.
- **Preserve every annotation** (severity, lesion counts, lesion bounding boxes,
  train/test split, and 5-fold cross-validation assignments) in an easy-to-use
  `annotations.jsonl` file.
- Produce a reproducible dataset summary.
- Provide a clean, scalable project layout for future CV development.

## Dataset

**ACNE04** — 1,457 facial acne photographs with 4-level severity grades and
lesion-level bounding boxes.

- Source: [`Layered-Labs/ACNE04`](https://huggingface.co/datasets/Layered-Labs/ACNE04)
  (repackaging of the original ACNE04 by Wu et al., ICCV 2019).
- **License: academic / research use only — not for commercial use.**

See [docs/dataset_documentation.md](docs/dataset_documentation.md) for the full
schema, severity distribution, citation, and usage restrictions.

## Project structure

```text
acnevision-ai/
├── data/
│   ├── raw/ACNE04/train/              # original .arrow shards (git-ignored)
│   └── processed/ACNE04/train/        # generated (git-ignored)
│       ├── images/                    # acne_00000.jpg … acne_01456.jpg
│       ├── annotations.jsonl          # one record per image, all fields preserved
│       └── dataset_summary.json
├── notebooks/                         # exploration / visualization / analysis
├── src/acnevision/
│   ├── data/                          # convert / inspect / validate the dataset
│   ├── preprocessing/ detection/ classification/ restoration/   # placeholders
├── scripts/convert_acne04.py          # command-line entry point
├── tests/                             # pytest unit tests
├── docs/                              # documentation
├── models/                            # checkpoints/ trained/ (git-ignored contents)
├── outputs/                           # figures/ predictions/ logs/ (git-ignored contents)
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (CPU-only)
pip install -r requirements.txt
# For tests + notebooks as well:
pip install -r requirements-dev.txt
```

## Getting the dataset

The dataset is **not** committed to git. Download it from Hugging Face:

```bash
python download_dataset.py
```

Then place the three `data-*-of-*.arrow` shards under
`data/raw/ACNE04/train/` (this is where the pipeline looks by default). If your
shards live elsewhere, pass `--raw-dir` to the conversion script.

## Converting the dataset

Run from the project root:

```bash
python scripts/convert_acne04.py
```

This reads the Arrow shards from `data/raw/ACNE04/train/` and writes:

- `data/processed/ACNE04/train/images/acne_XXXXX.jpg`
- `data/processed/ACNE04/train/annotations.jsonl`
- `data/processed/ACNE04/train/dataset_summary.json`

The conversion is **idempotent**: images that already exist are skipped (not
re-decoded), while the annotations and summary are always rebuilt. Useful flags:

| Flag | Effect |
|------|--------|
| `--force` | Re-decode and overwrite all images. |
| `--raw-dir PATH` | Where the `.arrow` shards live. |
| `--out-dir PATH` | Where to write images/annotations/summary. |
| `--quality N` | JPG quality (1–95) for newly written images. |
| `--inspect` | Print the schema and a sample record, then exit. |
| `--validate` | Check the processed output for consistency, then exit. |

## Inspecting the dataset

```bash
# Schema, columns, shard count, total records, and a sample record
python scripts/convert_acne04.py --inspect

# Verify every annotation maps to an existing, uniquely named image
python scripts/convert_acne04.py --validate
```

The notebooks in [notebooks/](notebooks/) go further: `01_dataset_exploration`,
`02_data_visualization` (images with lesion boxes), and `03_annotation_analysis`
(severity / fold distributions).

## Running the tests

```bash
python -m pytest tests/ -q
```

The tests use tiny synthetic data, so they run fast and do not need the full
dataset.

## Roadmap (future work — not yet implemented)

- **Preprocessing:** resizing/normalization utilities for model input.
- **Detection:** lesion detection from the `lesion_boxes` annotations.
- **Classification:** acne-severity grading (`level` / `severity`).
- **Restoration:** experimental acne removal / skin restoration.

These correspond to the placeholder packages under `src/acnevision/` and have no
implementation yet.

## License & data use

The **code** in this repository is yours to license as you wish. The **ACNE04
dataset is academic / research use only and must not be used commercially** — this
project does not change those terms. See
[docs/dataset_documentation.md](docs/dataset_documentation.md).
