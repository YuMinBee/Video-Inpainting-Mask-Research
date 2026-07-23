#!/usr/bin/env python3
"""Compare temporal flicker and boundary preservation across refinement variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = {
    "baseline": "results/aihub_subset_20/propainter_outputs",
    "depth_aware": "results/aihub_subset_20/refined_outputs/depth_aware",
    "boundary_only": "results/aihub_subset_20/refined_outputs/boundary_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/02_boundary_refinement/metrics/variant_comparison"))
    parser.add_argument("--edge-tolerance", type=int, default=2)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
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
    height, width = flow_curr_to_prev.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow_curr_to_prev[..., 0]
    map_y = grid_y + flow_curr_to_prev[..., 1]
    return cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def edge_map(image: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high) > 0


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def boundary_fscore(pred_edges: np.ndarray, ref_edges: np.ndarray, roi: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    pred = pred_edges & roi
    ref = ref_edges & roi
    if not np.any(pred) and not np.any(ref):
        return 1.0, 1.0, 1.0
    if not np.any(pred) or not np.any(ref):
        return 0.0, 0.0, 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    ref_d = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8), kernel) > 0
    precision = float((pred & ref_d).sum() / max(int(pred.sum()), 1))
    recall = float((ref & pred_d).sum() / max(int(ref.sum()), 1))
    fscore = 2.0 * precision * recall / (precision + recall + 1e-8)
    return fscore, precision, recall


def chamfer_distance(pred_edges: np.ndarray, ref_edges: np.ndarray, roi: np.ndarray) -> float:
    pred = pred_edges & roi
    ref = ref_edges & roi
    if not np.any(pred) and not np.any(ref):
        return 0.0
    if not np.any(pred) or not np.any(ref):
        return 999.0
    inv_ref = (~ref).astype(np.uint8)
    inv_pred = (~pred).astype(np.uint8)
    dist_to_ref = cv2.distanceTransform(inv_ref, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(inv_pred, cv2.DIST_L2, 3)
    return float(0.5 * (dist_to_ref[pred].mean() + dist_to_pred[ref].mean()))


def gradient_ratio(output: np.ndarray, reference: np.ndarray, roi: np.ndarray) -> float:
    if not np.any(roi):
        return 1.0
    out_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out_grad = cv2.magnitude(
        cv2.Sobel(out_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(out_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    ref_grad = cv2.magnitude(
        cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    return float((out_grad[roi].mean() + 1e-6) / (ref_grad[roi].mean() + 1e-6))


def change_from_reference(output: np.ndarray, reference: np.ndarray, roi: np.ndarray) -> float:
    diff = np.mean(np.abs(output.astype(np.float32) - reference.astype(np.float32)), axis=2) / 255.0
    return safe_mean(diff, roi)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {"overall": {}, "by_clip": {}}
    numeric = [k for k, v in rows[0].items() if isinstance(v, (float, int)) and k != "frame_index"]
    for method in sorted({r["method"] for r in rows}):
        method_rows = [r for r in rows if r["method"] == method]
        summary["overall"][method] = {
            key: float(np.mean([r[key] for r in method_rows])) for key in numeric
        }
    for clip in sorted({r["clip"] for r in rows}):
        summary["by_clip"][clip] = {}
        clip_rows_all = [r for r in rows if r["clip"] == clip]
        for method in sorted({r["method"] for r in clip_rows_all}):
            method_rows = [r for r in clip_rows_all if r["method"] == method]
            summary["by_clip"][clip][method] = {
                key: float(np.mean([r[key] for r in method_rows])) for key in numeric
            }
    return summary


def write_markdown(path: Path, summary: dict, cases: dict, args: argparse.Namespace) -> None:
    o = summary["overall"]

    def pct_change(base: float, value: float) -> float:
        return 100.0 * (base - value) / base if base else 0.0

    lines = [
        "# Refinement Variant Comparison",
        "",
        f"Probe root: `{args.probe_root}`",
        "",
        "Temporal flicker is computed as the mean RGB difference between the current output and the previous output warped to the current frame using Farneback flow.",
        "",
        "Boundary preservation uses ProPainter baseline edges as the reference, because the tested methods are post-processing variants of ProPainter.",
        "",
        "## Final Output Flicker",
        "",
        "| Method | BTE ↓ | B&D TE ↓ | B&!D TE ↓ | All-frame TE ↓ | BTE reduction vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base_bte = o["baseline"]["te_boundary"]
    for method in ["baseline", "depth_aware", "boundary_only"]:
        item = o[method]
        lines.append(
            f"| {method} | {item['te_boundary']:.6f} | {item['te_bd']:.6f} | "
            f"{item['te_b_not_d']:.6f} | {item['te_all']:.6f} | "
            f"{pct_change(base_bte, item['te_boundary']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Boundary Preservation Against ProPainter Baseline",
            "",
            "| Method | B&D BF ↑ | B&D Chamfer ↓ | B&D Gradient Ratio ↑ | B&D Change ↓ | B BF ↑ | B Chamfer ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ["baseline", "depth_aware", "boundary_only"]:
        item = o[method]
        lines.append(
            f"| {method} | {item['bf_bd']:.4f} | {item['chamfer_bd']:.4f} | "
            f"{item['grad_ratio_bd']:.4f} | {item['change_bd']:.6f} | "
            f"{item['bf_boundary']:.4f} | {item['chamfer_boundary']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Case Counts",
            "",
            "| Case | Frames | Ratio |",
            "|---|---:|---:|",
        ]
    )
    total = cases["total_frames"]
    for key, value in cases.items():
        if key == "total_frames":
            continue
        lines.append(f"| {key} | {value} | {100.0 * value / total:.2f}% |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Boundary-only smoothing reduces temporal flicker more aggressively.",
            "- Depth-aware refinement preserves ProPainter boundary contours better, especially on depth-edge boundary pixels.",
            "- If the paper claims temporal-error superiority over boundary-only, the current result does not support it.",
            "- The defensible claim is a stability-preservation trade-off: depth-aware refinement reduces flicker while modifying depth-discontinuity boundaries less than boundary-only smoothing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_roots = {name: Path(root) for name, root in METHODS.items()}
    rows: list[dict] = []
    case_counts = {
        "total_frames": 0,
        "depth_aware_beats_boundary_only_bte": 0,
        "depth_aware_beats_boundary_only_bd_te": 0,
        "depth_aware_beats_boundary_only_bf_bd": 0,
        "depth_aware_beats_boundary_only_chamfer_bd": 0,
        "depth_aware_beats_boundary_only_grad_bd": 0,
        "depth_aware_lower_bd_change": 0,
        "depth_aware_preservation_win_temporal_loss": 0,
    }

    frame_cache: dict[tuple[str, str, str], dict] = {}
    for clip_dir in sorted([p for p in args.probe_root.iterdir() if (p / "frames").exists()]):
        clip = clip_dir.name
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        outputs_prev: dict[str, np.ndarray] = {}
        frames_prev: np.ndarray | None = None
        for idx, frame_path in enumerate(frame_paths):
            frame = read_color(frame_path)
            boundary = read_gray(clip_dir / "mask_boundaries" / frame_path.name) > 0
            depth_edge = read_gray(clip_dir / "depth_edge_binary" / frame_path.name) > 0
            bd = boundary & depth_edge
            b_not_d = boundary & (~depth_edge)
            baseline_output = read_color(method_roots["baseline"] / clip / frame_path.name)
            ref_edges = edge_map(baseline_output, args.canny_low, args.canny_high)

            flow = backward_flow(frame, frames_prev) if frames_prev is not None else None
            frame_metrics: dict[str, dict] = {}
            for method, root in method_roots.items():
                output = read_color(root / clip / frame_path.name)
                if flow is None:
                    te = np.zeros(frame.shape[:2], dtype=np.float32)
                else:
                    warped_prev = warp_with_backward_flow(outputs_prev[method], flow)
                    te = temporal_error(output, warped_prev)

                pred_edges = edge_map(output, args.canny_low, args.canny_high)
                bf_b, prec_b, rec_b = boundary_fscore(pred_edges, ref_edges, boundary, args.edge_tolerance)
                bf_bd, prec_bd, rec_bd = boundary_fscore(pred_edges, ref_edges, bd, args.edge_tolerance)
                row = {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "frame_index": idx,
                    "method": method,
                    "te_all": safe_mean(te, np.ones(te.shape, dtype=bool)),
                    "te_boundary": safe_mean(te, boundary),
                    "te_bd": safe_mean(te, bd),
                    "te_b_not_d": safe_mean(te, b_not_d),
                    "bf_boundary": bf_b,
                    "bf_boundary_precision": prec_b,
                    "bf_boundary_recall": rec_b,
                    "bf_bd": bf_bd,
                    "bf_bd_precision": prec_bd,
                    "bf_bd_recall": rec_bd,
                    "chamfer_boundary": chamfer_distance(pred_edges, ref_edges, boundary),
                    "chamfer_bd": chamfer_distance(pred_edges, ref_edges, bd),
                    "grad_ratio_boundary": gradient_ratio(output, baseline_output, boundary),
                    "grad_ratio_bd": gradient_ratio(output, baseline_output, bd),
                    "change_boundary": change_from_reference(output, baseline_output, boundary),
                    "change_bd": change_from_reference(output, baseline_output, bd),
                }
                rows.append(row)
                frame_metrics[method] = row
                outputs_prev[method] = output

            if idx > 0:
                case_counts["total_frames"] += 1
                da = frame_metrics["depth_aware"]
                bo = frame_metrics["boundary_only"]
                if da["te_boundary"] < bo["te_boundary"]:
                    case_counts["depth_aware_beats_boundary_only_bte"] += 1
                if da["te_bd"] < bo["te_bd"]:
                    case_counts["depth_aware_beats_boundary_only_bd_te"] += 1
                if da["bf_bd"] > bo["bf_bd"]:
                    case_counts["depth_aware_beats_boundary_only_bf_bd"] += 1
                if da["chamfer_bd"] < bo["chamfer_bd"]:
                    case_counts["depth_aware_beats_boundary_only_chamfer_bd"] += 1
                if da["grad_ratio_bd"] > bo["grad_ratio_bd"]:
                    case_counts["depth_aware_beats_boundary_only_grad_bd"] += 1
                if da["change_bd"] < bo["change_bd"]:
                    case_counts["depth_aware_lower_bd_change"] += 1
                if da["te_boundary"] > bo["te_boundary"] and da["bf_bd"] > bo["bf_bd"] and da["change_bd"] < bo["change_bd"]:
                    case_counts["depth_aware_preservation_win_temporal_loss"] += 1

            frames_prev = frame

    summary = summarize(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "frame_metrics.csv", rows)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "case_counts.json").write_text(json.dumps(case_counts, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary, case_counts, args)
    print(f"Wrote {len(rows)} method-frame rows to {args.output_root}")
    print(json.dumps(case_counts, indent=2))


if __name__ == "__main__":
    main()
