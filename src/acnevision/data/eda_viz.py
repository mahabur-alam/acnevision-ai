"""Plotting helpers and bounding-box overlays for ACNE04 EDA.

All charts use matplotlib + seaborn and are written to disk with
:func:`save_fig`, so the notebooks stay short and every figure lands under
``outputs/figures/``. Nothing here needs a GPU.

The overlay helpers draw the stored ``xmin``/``ymin``/``xmax``/``ymax`` boxes on a
*copy* of each original image, which both visualises lesions and verifies that the
coordinates map correctly onto the pictures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Use a non-interactive backend so notebooks/scripts render headless (CPU-only).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .eda_severity import SEVERITY_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

# A readable colour per severity class, reused by charts and overlays.
SEVERITY_COLORS = {
    "mild": "#4C9F70",
    "moderate": "#E1A730",
    "severe": "#D9603B",
    "very_severe": "#B23A48",
}


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 120) -> Path:
    """Save ``fig`` to ``out_path`` (creating parent dirs) and close it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)
    return out_path


def plot_resolution_scatter(sizes_df: pd.DataFrame, out_path: Path) -> Path:
    """Scatter of image width vs height to show the spread of resolutions."""
    ok = sizes_df.dropna(subset=["width", "height"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ok["width"], ok["height"], s=12, alpha=0.4, color="#3E6D9C")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Image height (px)")
    ax.set_title("ACNE04 image resolution distribution")
    return save_fig(fig, out_path)


def plot_aspect_ratio_hist(sizes_df: pd.DataFrame, out_path: Path) -> Path:
    """Histogram of width/height aspect ratios."""
    ok = sizes_df.dropna(subset=["aspect_ratio"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(ok["aspect_ratio"], bins=30, ax=ax, color="#3E6D9C")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="square (1.0)")
    ax.set_xlabel("Aspect ratio (width / height)")
    ax.set_title("ACNE04 aspect-ratio distribution")
    ax.legend()
    return save_fig(fig, out_path)


def plot_severity_bar(table: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of per-class image counts from a severity table."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [SEVERITY_COLORS.get(s, "#888888") for s in table["severity"]]
    ax.bar(table["severity"], table["count"], color=colors)
    for x, (count, pct) in enumerate(zip(table["count"], table["percentage"])):
        ax.text(x, count, f"{int(count)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title("ACNE04 severity-class distribution")
    ax.margins(y=0.15)
    return save_fig(fig, out_path)


def plot_lesion_count_hist(df: pd.DataFrame, out_path: Path, clip: int = 80) -> Path:
    """Histogram of lesions-per-image (clipped for readability)."""
    counts = df["num_boxes"].clip(upper=clip)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(counts, bins=40, ax=ax, color="#7D5BA6")
    ax.set_xlabel(f"Lesions per image (clipped at {clip})")
    ax.set_ylabel("Number of images")
    ax.set_title("ACNE04 lesions-per-image distribution")
    return save_fig(fig, out_path)


def plot_box_size_hist(metrics_df: pd.DataFrame, out_path: Path) -> Path:
    """Histograms of raw box width and height (original pixels)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(metrics_df["box_width"], bins=40, ax=axes[0], color="#3E6D9C")
    axes[0].set_title("Lesion box width (px, original)")
    axes[0].set_xlabel("width (px)")
    sns.histplot(metrics_df["box_height"], bins=40, ax=axes[1], color="#D9603B")
    axes[1].set_title("Lesion box height (px, original)")
    axes[1].set_xlabel("height (px)")
    return save_fig(fig, out_path)


def plot_resize_simulation(sim_df: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of the 'tiny lesion' share at each candidate resolution."""
    tiny_col = next((c for c in sim_df.columns if c.startswith("pct_tiny")), None)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sim_df["target_px"].astype(str), sim_df[tiny_col], color="#B23A48")
    for x, v in enumerate(sim_df[tiny_col]):
        ax.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("YOLO input resolution (px)")
    ax.set_ylabel("% lesions too tiny to detect")
    ax.set_title("Share of lesions that become sub-detectable after resize")
    ax.margins(y=0.15)
    return save_fig(fig, out_path)


def _norm_color(hex_color: str) -> tuple[int, int, int]:
    """Convert a '#RRGGBB' string to an (R, G, B) 0-255 tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_boxes_on_image(
    image_path: Path,
    boxes: list[tuple[int, int, int, int]],
    out_path: Path,
    color: str = "#00FF88",
    width_frac: float = 0.004,
) -> Path | None:
    """Draw lesion boxes on a copy of one original image and save it.

    ``boxes`` is a list of ``(xmin, ymin, xmax, ymax)`` tuples in original pixel
    coordinates. Line thickness scales with image size so boxes stay visible on
    very large pictures. Returns ``None`` if the image cannot be opened.
    """
    image_path = Path(image_path)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot open %s for overlay: %s", image_path.name, exc)
        return None

    draw = ImageDraw.Draw(img)
    line_w = max(2, int(min(img.size) * width_frac))
    rgb = _norm_color(color)
    for xmin, ymin, xmax, ymax in boxes:
        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb, width=line_w)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)
    return out_path


def save_box_samples(
    df: pd.DataFrame,
    out_dir: Path,
    per_class: int = 5,
    seed: int = 42,
) -> list[Path]:
    """Save box-overlay samples spanning all four severity classes.

    Picks ``per_class`` images (preferring ones that actually have lesions) from
    each severity class, draws their boxes, and writes them to ``out_dir``.
    With the default ``per_class=5`` this yields >=20 images across the 4 classes.
    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for severity in SEVERITY_ORDER:
        subset = df[df["severity"] == severity]
        with_boxes = subset[subset["num_boxes"] > 0]
        pool = with_boxes if len(with_boxes) >= per_class else subset
        if pool.empty:
            continue
        chosen = pool.sample(n=min(per_class, len(pool)), random_state=seed)

        for _, rec in chosen.iterrows():
            lesion_boxes = rec["lesion_boxes"] or {}
            boxes = list(zip(
                lesion_boxes.get("xmin", []),
                lesion_boxes.get("ymin", []),
                lesion_boxes.get("xmax", []),
                lesion_boxes.get("ymax", []),
            ))
            out_path = out_dir / f"{severity}__{rec['image_file']}"
            saved = draw_boxes_on_image(rec["image_path"], boxes, out_path)
            if saved is not None:
                written.append(saved)

    logger.info("Saved %d bounding-box sample images to %s", len(written), out_dir)
    return written
