#!/usr/bin/env python3
"""Make visual comparison panels for depth-aware vs boundary-only refinement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/02_boundary_refinement/metrics/variant_comparison/frame_metrics.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/02_boundary_refinement/visualizations/comparison_panels"))
    parser.add_argument("--max-cases", type=int, default=6)
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


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def heat(diff: np.ndarray) -> np.ndarray:
    vis = np.clip(diff * 255 * 8, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    idx = mask > 0
    out[idx] = ((1 - alpha) * out[idx] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2) / 255.0


def rows_by_method(metrics_path: Path) -> dict[tuple[str, str], dict[str, dict]]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip"], row["frame"])
            grouped.setdefault(key, {})[row["method"]] = row
    return grouped


def score_cases(grouped: dict[tuple[str, str], dict[str, dict]], mode: str) -> list[tuple[float, str, str]]:
    cases: list[tuple[float, str, str]] = []
    for (clip, frame), item in grouped.items():
        if "depth_aware" not in item or "boundary_only" not in item:
            continue
        if item["depth_aware"]["frame_index"] == "0":
            continue
        da = item["depth_aware"]
        bo = item["boundary_only"]
        if mode == "preservation_win":
            score = (
                float(da["bf_bd"]) - float(bo["bf_bd"])
                + 0.1 * (float(bo["chamfer_bd"]) - float(da["chamfer_bd"]))
                + 2.0 * (float(bo["change_bd"]) - float(da["change_bd"]))
            )
        elif mode == "temporal_win":
            score = float(da["te_boundary"]) - float(bo["te_boundary"])
        else:
            score = float(bo["te_boundary"])
        cases.append((score, clip, frame))
    return sorted(cases, reverse=True)


def existing_frame_name(clip_dir: Path, frame: str) -> str | None:
    candidates = [
        f"{frame}.png",
        f"{frame}.jpg",
    ]
    if frame.startswith("frame_"):
        candidates.append(f"{frame}.png")
    for name in candidates:
        if (clip_dir / "frames" / name).exists():
            return name
    frames = sorted((clip_dir / "frames").glob("*.png"))
    if frame.startswith("frame_"):
        try:
            idx = int(frame.split("_")[-1])
            if 0 <= idx < len(frames):
                return frames[idx].name
        except ValueError:
            pass
    return None


def make_panel(args: argparse.Namespace, clip: str, frame: str, out_path: Path) -> bool:
    clip_dir = args.probe_root / clip
    name = existing_frame_name(clip_dir, frame)
    if name is None:
        return False

    original = read_color(clip_dir / "frames" / name)
    mask = read_gray(clip_dir / "masks" / name)
    boundary = read_gray(clip_dir / "mask_boundaries" / name)
    depth_edge = read_gray(clip_dir / "depth_edge_binary" / name)
    baseline = read_color(Path("results/aihub_subset_20/propainter_outputs") / clip / name)
    depth_aware = read_color(Path("results/aihub_subset_20/refined_outputs/depth_aware") / clip / name)
    boundary_only = read_color(Path("results/aihub_subset_20/refined_outputs/boundary_only") / clip / name)

    original_overlay = overlay_mask(original, mask, (0, 255, 255), 0.35)
    boundary_overlay = overlay_mask(original, boundary, (0, 255, 255), 0.95)
    boundary_overlay = overlay_mask(boundary_overlay, depth_edge, (0, 255, 0), 0.75)
    da_diff = heat(diff_image(depth_aware, baseline))
    bo_diff = heat(diff_image(boundary_only, baseline))
    bo_vs_da = heat(diff_image(boundary_only, depth_aware))

    top = np.hstack([
        label(original_overlay, "Original + mask"),
        label(boundary_overlay, "Boundary + depth edge"),
        label(baseline, "ProPainter baseline"),
    ])
    bottom = np.hstack([
        label(depth_aware, "Depth-aware"),
        label(boundary_only, "Boundary-only"),
        label(bo_vs_da, "Boundary-only vs depth-aware diff"),
    ])
    diff_row = np.hstack([
        label(da_diff, "Depth-aware change from baseline"),
        label(bo_diff, "Boundary-only change from baseline"),
        label(cv2.addWeighted(boundary_overlay, 0.55, bo_vs_da, 0.45, 0), "Diff over boundary/depth"),
    ])
    panel = np.vstack([top, bottom, diff_row])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return True


def main() -> None:
    args = parse_args()
    grouped = rows_by_method(args.metrics)
    selections = {
        "depth_aware_preserves_boundary": score_cases(grouped, "preservation_win")[: args.max_cases],
        "boundary_only_reduces_flicker": score_cases(grouped, "temporal_win")[: args.max_cases],
    }

    written = []
    for category, cases in selections.items():
        for rank, (_, clip, frame) in enumerate(cases, start=1):
            out_path = args.output_root / category / f"{rank:02d}_{clip}_{frame}.png"
            if make_panel(args, clip, frame, out_path):
                written.append(out_path)
                print(out_path)

    index_lines = ["# Refinement Comparison Panels", ""]
    for path in written:
        rel = path.relative_to(args.output_root)
        index_lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(written)} panels to {args.output_root}")


if __name__ == "__main__":
    main()
