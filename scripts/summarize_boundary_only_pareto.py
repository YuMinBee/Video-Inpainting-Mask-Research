#!/usr/bin/env python3
"""Summarize boundary-only sweep against depth-aware refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/02_boundary_refinement/metrics/boundary_only_pareto"))
    parser.add_argument("--edge-tolerance", type=int, default=2)
    return parser.parse_args()


def method_roots() -> dict[str, Path]:
    roots = {
        "Baseline": Path("results/aihub_subset_20/propainter_outputs"),
        "Depth-aware": Path("results/aihub_subset_20/refined_outputs/depth_aware"),
    }
    for value in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tag = f"lambda_{value:.1f}".replace(".", "p")
        roots[f"Boundary-only λ={value:.1f}"] = Path("results/aihub_subset_20/refined_outputs/boundary_only_sweep") / tag
    return roots


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
    height, width = flow_curr_to_prev.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        prev,
        grid_x + flow_curr_to_prev[..., 0],
        grid_y + flow_curr_to_prev[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150) > 0


def fscore(pred_edges: np.ndarray, ref_edges: np.ndarray, roi: np.ndarray, tolerance: int) -> float:
    pred = pred_edges & roi
    ref = ref_edges & roi
    if not np.any(pred) and not np.any(ref):
        return 1.0
    if not np.any(pred) or not np.any(ref):
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    ref_d = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8), kernel) > 0
    precision = float((pred & ref_d).sum() / max(int(pred.sum()), 1))
    recall = float((ref & pred_d).sum() / max(int(ref.sum()), 1))
    return float(2.0 * precision * recall / (precision + recall + 1e-8))


def chamfer(pred_edges: np.ndarray, ref_edges: np.ndarray, roi: np.ndarray) -> float:
    pred = pred_edges & roi
    ref = ref_edges & roi
    if not np.any(pred) and not np.any(ref):
        return 0.0
    if not np.any(pred) or not np.any(ref):
        return 999.0
    dist_to_ref = cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dist_to_ref[pred].mean() + dist_to_pred[ref].mean()))


def gradient_ratio(output: np.ndarray, reference: np.ndarray, roi: np.ndarray) -> float:
    if not np.any(roi):
        return 1.0
    out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out_g = cv2.magnitude(cv2.Sobel(out, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(out, cv2.CV_32F, 0, 1, ksize=3))
    ref_g = cv2.magnitude(cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3))
    return float((out_g[roi].mean() + 1e-6) / (ref_g[roi].mean() + 1e-6))


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if np.any(mask) else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Boundary-Only Strength Sweep",
        "",
        "Boundary-only strength λ is implemented as the maximum temporal blending alpha inside the mask-boundary band.",
        "",
        "| Method | Boundary TE ↓ | B&D TE ↓ | B&D BF ↑ | B&D Chamfer ↓ | B&D Gradient Ratio ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['boundary_te']:.6f} | {row['bd_te']:.6f} | "
            f"{row['bd_bf']:.4f} | {row['bd_chamfer']:.4f} | {row['bd_gradient_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The table should be read as a Pareto trade-off rather than a single-winner ranking. "
            "Lower boundary-only λ preserves edges better but reduces flicker less; higher λ reduces flicker more but increasingly degrades boundary contours.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = method_roots()
    accum = {method: {"boundary_te": [], "bd_te": [], "bd_bf": [], "bd_chamfer": [], "bd_gradient_ratio": []} for method in roots}

    for clip_dir in sorted([p for p in args.probe_root.iterdir() if (p / "frames").exists()]):
        clip = clip_dir.name
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        prev_frame = None
        prev_outputs: dict[str, np.ndarray] = {}
        for idx, frame_path in enumerate(frame_paths):
            frame = read_color(frame_path)
            boundary = read_gray(clip_dir / "mask_boundaries" / frame_path.name) > 0
            depth_edge = read_gray(clip_dir / "depth_edge_binary" / frame_path.name) > 0
            bd = boundary & depth_edge
            baseline = read_color(roots["Baseline"] / clip / frame_path.name)
            ref_edges = edge_map(baseline)
            flow = backward_flow(frame, prev_frame) if prev_frame is not None else None

            for method, root in roots.items():
                output = read_color(root / clip / frame_path.name)
                edges = edge_map(output)
                if flow is not None:
                    warped_prev = warp_with_backward_flow(prev_outputs[method], flow)
                    te = temporal_error(output, warped_prev)
                    accum[method]["boundary_te"].append(safe_mean(te, boundary))
                    accum[method]["bd_te"].append(safe_mean(te, bd))
                accum[method]["bd_bf"].append(fscore(edges, ref_edges, bd, args.edge_tolerance))
                accum[method]["bd_chamfer"].append(chamfer(edges, ref_edges, bd))
                accum[method]["bd_gradient_ratio"].append(gradient_ratio(output, baseline, bd))
                prev_outputs[method] = output
            prev_frame = frame

    rows = []
    for method, values in accum.items():
        rows.append(
            {
                "method": method,
                "boundary_te": float(np.mean(values["boundary_te"])),
                "bd_te": float(np.mean(values["bd_te"])),
                "bd_bf": float(np.mean(values["bd_bf"])),
                "bd_chamfer": float(np.mean(values["bd_chamfer"])),
                "bd_gradient_ratio": float(np.mean(values["bd_gradient_ratio"])),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "pareto_table.csv", rows)
    (args.output_root / "pareto_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
