#!/usr/bin/env python3
"""Evaluate mask-union inpainting outputs for flicker and over-removal."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = {
    "Original mask ProPainter": {
        "output": "results/aihub_subset_20/propainter_outputs",
        "mask": "experiments/aihub_subset_20_real",
    },
    "Boundary-only smoothing": {
        "output": "results/aihub_subset_20/refined_outputs/boundary_only",
        "mask": "experiments/aihub_subset_20_real",
    },
    "Temporal union mask ProPainter": {
        "output": "results/mask_union/temporal_union/propainter_outputs",
        "mask": "experiments/mask_union/temporal_union",
    },
    "Depth-limited union mask ProPainter": {
        "output": "results/mask_union/depth_limited_union/propainter_outputs",
        "mask": "experiments/mask_union/depth_limited_union",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/mask_union/evaluation"))
    parser.add_argument("--change-threshold", type=float, default=10.0)
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


def backward_flow(curr_bgr: np.ndarray, prev_bgr: np.ndarray) -> np.ndarray:
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        curr,
        prev,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def warp_with_backward_flow(prev: np.ndarray, flow_curr_to_prev: np.ndarray) -> np.ndarray:
    h, w = flow_curr_to_prev.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev, grid_x + flow_curr_to_prev[..., 0], grid_y + flow_curr_to_prev[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if np.any(mask) else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    keys = [
        "boundary_te",
        "bd_te",
        "outside_change_mean",
        "outside_changed_fraction",
        "outside_changed_area_over_gt",
        "extra_mask_ratio",
        "extra_region_change_mean",
    ]
    out = []
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        item = {"method": method, "frames": len(method_rows)}
        for key in keys:
            values = np.array([r[key] for r in method_rows], dtype=np.float32)
            item[key] = float(np.nanmean(values))
        out.append(item)
    return out


def write_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "# Mask Union Evaluation",
        "",
        "Over-removal is measured as output change outside the original GT mask. This is a proxy because ground-truth clean background is unavailable.",
        "",
        "| Method | Boundary TE ↓ | B&D TE ↓ | Outside Change ↓ | Outside Changed Fraction ↓ | Outside Changed Area / GT ↓ | Extra Mask Ratio ↓ | Extra Region Change ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['method']} | {item['boundary_te']:.6f} | {item['bd_te']:.6f} | "
            f"{item['outside_change_mean']:.6f} | {item['outside_changed_fraction']:.4f} | "
            f"{item['outside_changed_area_over_gt']:.4f} | "
            f"{item['extra_mask_ratio']:.4f} | {item['extra_region_change_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Temporal union masks test whether temporally stable mask support reduces boundary flicker.",
            "- Depth-limited union masks keep only temporal-union pixels connected to the current mask without crossing high depth-gradient barriers.",
            "- If depth-limited union reduces over-removal relative to full temporal union while keeping similar boundary flicker, this direction is stronger than post-hoc smoothing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for clip_dir in sorted([p for p in args.probe_root.iterdir() if (p / "frames").exists()]):
        clip = clip_dir.name
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        prev_frame = None
        prev_outputs: dict[str, np.ndarray] = {}
        for idx, frame_path in enumerate(frame_paths):
            name = frame_path.name
            frame = read_color(frame_path)
            gt_mask = read_gray(clip_dir / "masks" / name) > 0
            boundary = read_gray(clip_dir / "mask_boundaries" / name) > 0
            depth_edge = read_gray(clip_dir / "depth_edge_binary" / name) > 0
            bd = boundary & depth_edge
            raw_gt_area = int(gt_mask.sum())
            valid_gt = raw_gt_area > 100
            gt_area = max(raw_gt_area, 1)
            flow = backward_flow(frame, prev_frame) if prev_frame is not None else None
            for method, spec in METHODS.items():
                output = read_color(Path(spec["output"]) / clip / name)
                variant_mask_path = Path(spec["mask"]) / clip / "masks" / name
                variant_mask = read_gray(variant_mask_path) > 0
                extra = variant_mask & (~gt_mask)
                diff = np.mean(np.abs(output.astype(np.float32) - frame.astype(np.float32)), axis=2)
                outside = ~gt_mask
                outside_changed = (diff > args.change_threshold) & outside
                if flow is None:
                    te = np.zeros(frame.shape[:2], dtype=np.float32)
                else:
                    warped_prev = warp_with_backward_flow(prev_outputs[method], flow)
                    te = temporal_error(output, warped_prev)
                rows.append(
                    {
                        "clip": clip,
                        "frame": frame_path.stem,
                        "frame_index": idx,
                        "method": method,
                        "boundary_te": safe_mean(te, boundary),
                        "bd_te": safe_mean(te, bd),
                        "outside_change_mean": safe_mean(diff / 255.0, outside),
                        "outside_changed_fraction": float(outside_changed.sum() / max(int(outside.sum()), 1)),
                        "outside_changed_area_over_gt": float(outside_changed.sum() / gt_area) if valid_gt else np.nan,
                        "extra_mask_ratio": float(extra.sum() / gt_area) if valid_gt else np.nan,
                        "extra_region_change_mean": safe_mean(diff / 255.0, extra),
                    }
                )
                prev_outputs[method] = output
            prev_frame = frame

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
