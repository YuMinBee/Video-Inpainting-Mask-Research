#!/usr/bin/env python3
"""Build the README's factual same-area mask-comparison figure.

The figure uses only binary experiment masks, not redistributable source RGB.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(
    "D:/DCG-TR_experiment_results/experiments/15_learned_budget_density_cv"
)
DEFAULT_CLIP = "synthetic_002_I-211109_I07021_W06"
DEFAULT_FRAME = "frame_0008"
DEFAULT_SETTING = "sam_jitter_10"

BACKGROUND = (248, 250, 252)
RAW_CORRECT = (71, 85, 105)
RECOVERED = (16, 185, 129)
FALSE_COVERED = (245, 158, 11)
STILL_MISSING = (220, 38, 38)
GT_COLOR = (37, 99, 235)
INK = (15, 23, 42)
MUTED = (71, 85, 105)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--setting", default=DEFAULT_SETTING)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/figures/mask_correction_comparison.png"),
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 0


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    selected_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=selected_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=selected_font,
        fill=fill,
    )


def metric_rows(root: Path, clip: str, frame: str, setting: str) -> dict[str, dict]:
    path = root / "evaluation" / "sealed_test" / "frame_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["clip"] == clip
            and row["frame"] == frame
            and row["setting"] == setting
        ]
    result = {row["method"]: row for row in rows}
    required = {"hybrid_frontier", "exact_area_distance"}
    if not required <= result.keys():
        raise RuntimeError(f"Missing registered frame metrics: {required - result.keys()}")
    return result


def crop_box(*masks: np.ndarray, padding: int = 25) -> tuple[int, int, int, int]:
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    if not len(xs):
        raise RuntimeError("Cannot crop empty masks")
    height, width = union.shape
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )


def categorical_image(
    gt: np.ndarray, raw: np.ndarray, corrected: np.ndarray | None
) -> np.ndarray:
    image = np.full((*gt.shape, 3), BACKGROUND, dtype=np.uint8)
    image[raw & gt] = RAW_CORRECT
    image[raw & ~gt] = FALSE_COVERED
    if corrected is None:
        image[gt & ~raw] = STILL_MISSING
        return image
    added = corrected & ~raw
    image[added & gt] = RECOVERED
    image[added & ~gt] = FALSE_COVERED
    image[gt & ~corrected] = STILL_MISSING
    return image


def fit_panel(image: np.ndarray, width: int, height: int) -> Image.Image:
    source = Image.fromarray(image)
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.NEAREST,
    )
    panel = Image.new("RGB", (width, height), BACKGROUND)
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return panel


def legend_item(
    draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], label: str
) -> int:
    draw.rounded_rectangle((x, y, x + 22, y + 22), radius=4, fill=color)
    draw.text((x + 31, y - 1), label, font=font(18), fill=MUTED)
    return x + 31 + int(draw.textlength(label, font=font(18))) + 34


def main() -> None:
    args = parse_args()
    root = args.experiment_root
    clip = args.clip
    frame_name = f"{args.frame}.png"
    gt = read_mask(root / "data" / "sealed_test" / "probe" / clip / "masks" / frame_name)
    raw = read_mask(
        root
        / "data"
        / "sealed_test"
        / "sam_raw"
        / args.setting
        / clip
        / "masks"
        / frame_name
    )
    distance = read_mask(
        root
        / "evaluation"
        / "sealed_test"
        / "masks"
        / "exact_area_distance"
        / args.setting
        / clip
        / "masks"
        / frame_name
    )
    proposed = read_mask(
        root
        / "evaluation"
        / "sealed_test"
        / "masks"
        / "hybrid_frontier"
        / args.setting
        / clip
        / "masks"
        / frame_name
    )
    if not (gt.shape == raw.shape == distance.shape == proposed.shape):
        raise RuntimeError("Mask shapes differ")

    rows = metric_rows(root, clip, args.frame, args.setting)
    distance_row = rows["exact_area_distance"]
    proposed_row = rows["hybrid_frontier"]
    distance_added = int((distance & ~raw).sum())
    proposed_added = int((proposed & ~raw).sum())
    registered_budget = int(distance_row["predicted_budget"])
    if distance_added != proposed_added or distance_added != registered_budget:
        raise RuntimeError("Figure inputs violate the registered exact-area rule")

    left, top, right, bottom = crop_box(gt, raw, distance, proposed)
    crop = np.s_[top:bottom, left:right]
    input_view = categorical_image(gt, raw, None)[crop]
    distance_view = categorical_image(gt, raw, distance)[crop]
    proposed_view = categorical_image(gt, raw, proposed)[crop]
    gt_view = np.full((*gt.shape, 3), BACKGROUND, dtype=np.uint8)
    gt_view[gt] = GT_COLOR
    gt_view = gt_view[crop]

    canvas_width = 1600
    canvas_height = 620
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    centered_text(
        draw,
        (0, 18, canvas_width, 66),
        "Same-area mask correction on a locked test frame",
        font(31, bold=True),
        INK,
    )
    centered_text(
        draw,
        (0, 62, canvas_width, 94),
        "Representative near-median source | 10% prompt jitter | frame 8 / 16",
        font(18),
        MUTED,
    )

    panel_width = 350
    image_height = 335
    gap = 28
    start_x = (canvas_width - (4 * panel_width + 3 * gap)) // 2
    panels = [
        (
            "1. Imperfect input",
            f"{int(distance_row['hidden_area']):,} target pixels missed",
            input_view,
        ),
        (
            "2. Distance expansion",
            f"{100 * float(distance_row['missing_recovery']):.1f}% recovered  |  +{distance_added:,} px",
            distance_view,
        ),
        (
            "3. Ours: density-guided",
            f"{100 * float(proposed_row['missing_recovery']):.1f}% recovered  |  +{proposed_added:,} px",
            proposed_view,
        ),
        ("4. Removal-support GT", "evaluation only; never an input", gt_view),
    ]
    for index, (title, subtitle, image) in enumerate(panels):
        x = start_x + index * (panel_width + gap)
        centered_text(
            draw, (x, 102, x + panel_width, 137), title, font(22, bold=True), INK
        )
        centered_text(
            draw, (x, 135, x + panel_width, 166), subtitle, font(16), MUTED
        )
        rendered = fit_panel(image, panel_width, image_height)
        canvas.paste(rendered, (x, 172))
        draw.rounded_rectangle(
            (x, 172, x + panel_width, 172 + image_height),
            radius=7,
            outline=(203, 213, 225),
            width=2,
        )

    legend_y = 546
    legend_labels = [
        (RAW_CORRECT, "raw mask on target"),
        (RECOVERED, "correctly recovered"),
        (STILL_MISSING, "still missed"),
        (FALSE_COVERED, "covered background"),
    ]
    widths = []
    legend_font = font(18)
    for _, label in legend_labels:
        widths.append(22 + 9 + int(draw.textlength(label, font=legend_font)) + 34)
    x = (canvas_width - sum(widths)) // 2
    for color, label in legend_labels:
        x = legend_item(draw, x, legend_y, color, label)
    centered_text(
        draw,
        (0, 580, canvas_width, 612),
        "Distance and ours add exactly the same number of pixels; only placement differs.",
        font(17, bold=True),
        INK,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(
        f"wrote {args.output} | exact-area={distance_added} | "
        f"recovery={float(distance_row['missing_recovery']):.4f} -> "
        f"{float(proposed_row['missing_recovery']):.4f}"
    )


if __name__ == "__main__":
    main()
