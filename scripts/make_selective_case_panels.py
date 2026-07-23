#!/usr/bin/env python3
"""Make paper-style case panels for selective temporal expansion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


CASES = [
    (
        "01_success_balanced",
        "I-210715_I03012_W06",
        "frame_0016",
        "Success: residue is reduced while outside changes remain controlled",
    ),
    (
        "02_temporal_union_overchange",
        "I-210618_I01001_W01",
        "frame_0011",
        "Temporal-union over-change: ours suppresses outside edits",
    ),
    (
        "03_failure_large_extra",
        "I-210720_O12052_T04",
        "frame_0001",
        "Failure/limit: selected extra mask becomes broad",
    ),
]

METHODS = [
    ("Boundary-only", "Boundary-only", Path("results/selective_expansion/basic/boundary_only_mask/propainter_outputs")),
    ("Temporal union", "Temporal union", Path("results/mask_union/temporal_union/propainter_outputs")),
    ("Ours r10 t0.10", "Aggressive pixel r10 t0.10", Path("results/selective_expansion/aggressive/pixel_r10_t0p10/propainter_outputs")),
    ("Ours r10 t0.15", "Aggressive pixel r10 t0.15", Path("results/selective_expansion/aggressive/pixel_r10_t0p15/propainter_outputs")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/selective_expansion/full20_evaluation/frame_metrics.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/selective_expansion/full20_evaluation/case_panels"))
    return parser.parse_args()


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip"], row["frame"], row["method"])
            metrics[key] = {
                "boundary_te": float(row["boundary_te"]),
                "outside": float(row["outside_changed_fraction"]),
                "extra": float(row["extra_mask_ratio"]),
                "residue": float(row["residue_diff_le_10"]),
            }
    return metrics


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def footer(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.putText(out, text, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_tile(image: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    idx = mask > 0
    out[idx] = ((1.0 - alpha) * out[idx] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def diff_heat(output: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(output.astype(np.float32) - frame.astype(np.float32)), axis=2)
    vis = np.clip(diff * 5.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    return draw_contour(heat, mask, (255, 255, 255))


def crop_bounds(mask: np.ndarray, pad: int = 80) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, w)
    y2 = min(int(ys.max()) + pad + 1, h)
    return x1, y1, x2, y2


def crop(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2]


def method_footer(metrics: dict[tuple[str, str, str], dict[str, float]], clip: str, frame: str, method: str) -> str:
    m = metrics[(clip, frame, method)]
    return f"out {m['outside']:.3f} | extra {m['extra']:.2f} | res {m['residue']:.3f}"


def make_panel(
    args: argparse.Namespace,
    metrics: dict[tuple[str, str, str], dict[str, float]],
    case_id: str,
    clip: str,
    frame: str,
    title: str,
    cropped: bool,
) -> Path:
    name = f"{frame}.png"
    clip_dir = args.probe_root / clip
    input_frame = read_color(clip_dir / "frames" / name)
    mask = read_gray(clip_dir / "masks" / name)

    outputs = [(label_name, metric_name, read_color(root / clip / name)) for label_name, metric_name, root in METHODS]
    bounds = crop_bounds(mask) if cropped else (0, 0, input_frame.shape[1], input_frame.shape[0])

    input_vis = draw_contour(overlay_mask(input_frame, mask, (0, 255, 255), 0.32), mask, (0, 255, 255))
    top_tiles = [footer(label(resize_tile(crop(input_vis, bounds)), "Input + mask"), title[:48])]
    bottom_tiles = [label(resize_tile(crop(draw_contour(input_frame, mask, (0, 255, 255)), bounds)), "Mask contour")]

    for label_name, metric_name, output in outputs:
        top = resize_tile(crop(draw_contour(output, mask, (0, 255, 255)), bounds))
        top_tiles.append(footer(label(top, label_name), method_footer(metrics, clip, frame, metric_name)))
        bottom_tiles.append(label(resize_tile(crop(diff_heat(output, input_frame, mask), bounds)), f"Change heat: {label_name}"))

    spacer = np.full((16, top_tiles[0].shape[1] * len(top_tiles), 3), 255, dtype=np.uint8)
    panel = np.vstack([np.hstack(top_tiles), spacer, np.hstack(bottom_tiles)])

    suffix = "crop" if cropped else "full"
    out_path = args.output_root / f"{case_id}_{clip}_{frame}_{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), panel):
        raise RuntimeError(f"Failed to write panel: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    written = []
    for case_id, clip, frame, title in CASES:
        written.append(make_panel(args, metrics, case_id, clip, frame, title, cropped=False))
        written.append(make_panel(args, metrics, case_id, clip, frame, title, cropped=True))

    lines = ["# Selective Expansion Case Panels", ""]
    for path in written:
        rel = path.relative_to(args.output_root)
        lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")
    lines.extend(
        [
            "",
            "Cases:",
            "",
            "- `01_success_balanced`: Boundary-only leaves residue, temporal union over-edits outside, ours provides a trade-off.",
            "- `02_temporal_union_overchange`: Temporal union causes very large outside changes; ours suppresses them.",
            "- `03_failure_large_extra`: Ours also selects a broad extra region, showing a limitation case.",
        ]
    )
    (args.output_root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
