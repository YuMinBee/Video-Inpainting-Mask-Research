#!/usr/bin/env python3
"""Evaluate selective expansion on representative clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CLIPS = [
    "I-211025_O02012_T01",
    "I-210618_I01001_W01",
    "I-210718_O08036_W04",
]

METHODS = {
    "Original": {
        "output": "results/aihub_subset_20/propainter_outputs",
        "mask": "experiments/aihub_subset_20_real",
    },
    "Boundary-only": {
        "output": "results/selective_expansion/basic/boundary_only_mask/propainter_outputs",
        "mask": "experiments/selective_expansion/basic/boundary_only_mask",
    },
    "Temporal union": {
        "output": "results/mask_union/temporal_union/propainter_outputs",
        "mask": "experiments/mask_union/temporal_union",
    },
    "Depth-limited union": {
        "output": "results/mask_union/depth_limited_union/propainter_outputs",
        "mask": "experiments/mask_union/depth_limited_union",
    },
    "Selective expansion": {
        "output": "results/selective_expansion/basic/selective_expansion/propainter_outputs",
        "mask": "experiments/selective_expansion/basic/selective_expansion",
    },
    "Aggressive pixel r10 t0.05": {
        "output": "results/selective_expansion/aggressive/pixel_r10_t0p05/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r10_t0p05",
    },
    "Aggressive pixel r10 t0.10": {
        "output": "results/selective_expansion/aggressive/pixel_r10_t0p10/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r10_t0p10",
    },
    "Aggressive pixel r10 t0.15": {
        "output": "results/selective_expansion/aggressive/pixel_r10_t0p15/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r10_t0p15",
    },
    "Ours r5 t0.10": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r5_t0p10/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r5_t0p10",
    },
    "Ours r10 t0.05": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r10_t0p05/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r10_t0p05",
    },
    "Ours r10 t0.10": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r10_t0p10/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r10_t0p10",
    },
    "Ours r10 t0.15": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r10_t0p15/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r10_t0p15",
    },
    "Ours r10 t0.20": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r10_t0p20/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r10_t0p20",
    },
    "Ours r15 t0.10": {
        "output": "results/4090_rerun/aihub_subset_100/ours_r15_t0p10/propainter_outputs",
        "mask": "experiments/aihub_subset_100/masks/ours_r15_t0p10",
    },
    "Aggressive pixel r15 t0.05": {
        "output": "results/selective_expansion/aggressive/pixel_r15_t0p05/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r15_t0p05",
    },
    "Aggressive pixel r15 t0.10": {
        "output": "results/selective_expansion/aggressive/pixel_r15_t0p10/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r15_t0p10",
    },
    "Aggressive pixel r15 t0.15": {
        "output": "results/selective_expansion/aggressive/pixel_r15_t0p15/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/pixel_r15_t0p15",
    },
    "Aggressive component r10 t0.10": {
        "output": "results/selective_expansion/aggressive/component_r10_t0p10/propainter_outputs",
        "mask": "experiments/selective_expansion/aggressive/component_r10_t0p10",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/selective_expansion/aggressive/evaluation"))
    parser.add_argument("--clips", nargs="+", default=CLIPS)
    parser.add_argument("--all-clips", action="store_true", help="Evaluate every clip directory under --probe-root.")
    parser.add_argument("--methods", nargs="+", choices=METHODS.keys(), default=list(METHODS.keys()))
    parser.add_argument(
        "--method-output",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method output root, e.g. 'Aggressive pixel r10 t0.10=results/...'.",
    )
    parser.add_argument(
        "--method-mask",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method mask root, e.g. 'Aggressive pixel r10 t0.10=experiments/...'.",
    )
    parser.add_argument("--change-threshold", type=float, default=10.0)
    return parser.parse_args()


def apply_method_overrides(output_overrides: list[str], mask_overrides: list[str]) -> dict:
    methods = {name: dict(spec) for name, spec in METHODS.items()}
    for override in output_overrides:
        if "=" not in override:
            raise RuntimeError(f"Invalid --method-output value: {override}")
        method, output = override.split("=", 1)
        if method not in methods:
            raise RuntimeError(f"Unknown method in --method-output: {method}")
        methods[method]["output"] = output
    for override in mask_overrides:
        if "=" not in override:
            raise RuntimeError(f"Invalid --method-mask value: {override}")
        method, mask = override.split("=", 1)
        if method not in methods:
            raise RuntimeError(f"Unknown method in --method-mask: {method}")
        methods[method]["mask"] = mask
    return methods


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
    return cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 25, 3, 5, 1.2, 0)


def warp(prev: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if np.any(mask) else 0.0


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    keys = [
        "boundary_te",
        "bd_te",
        "outside_changed_fraction",
        "outside_change_mean",
        "extra_mask_ratio",
        "inside_change_mean",
        "residue_diff_le_10",
    ]
    summary = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        item = {"method": method, "frames": len(method_rows)}
        for key in keys:
            item[key] = float(np.nanmean(np.array([r[key] for r in method_rows], dtype=np.float32)))
        summary.append(item)
    return summary


def write_markdown(path: Path, summary: list[dict], clips: list[str]) -> None:
    lines = [
        "# Selective Expansion Evaluation",
        "",
        f"Representative clips: `{', '.join(clips)}`",
        "",
        "| Method | Boundary TE ??| B&D TE ??| Outside Changed Fraction ??| Extra Mask Ratio ??| Inside Change ??| Residue diff??0 ??|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['boundary_te']:.6f} | {row['bd_te']:.6f} | "
            f"{row['outside_changed_fraction']:.6f} | {row['extra_mask_ratio']:.4f} | "
            f"{row['inside_change_mean']:.6f} | {row['residue_diff_le_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- Boundary-only mask: `original ??((temporal_union - original) ??boundary_band(r=5))`.",
            "- Selective expansion: `original ??((temporal_union - original) ??boundary_band(r=5) ??occupancy>0.4)`.",
            "- Aggressive pixel: `base ??((temporal_union - base) ??occupancy>?)`, where `base = original ??((temporal_union - original) ??boundary_band(r))`.",
            "- Aggressive component: same extra region as aggressive pixel, but connected components are selected when max occupancy exceeds ?.",
            "- Residue proxy is the fraction of original-mask pixels where output remains close to the original frame (`diff<=10`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = apply_method_overrides(args.method_output, args.method_mask)
    clips = sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir()) if args.all_clips else args.clips
    rows = []
    for clip in clips:
        clip_dir = args.probe_root / clip
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        prev_frame = None
        prev_outputs: dict[str, np.ndarray] = {}
        for idx, frame_path in enumerate(frame_paths):
            name = frame_path.name
            frame = read_color(frame_path)
            gt = read_gray(clip_dir / "masks" / name) > 0
            boundary = read_gray(clip_dir / "mask_boundaries" / name) > 0
            depth_edge = read_gray(clip_dir / "depth_edge_binary" / name) > 0
            bd = boundary & depth_edge
            flow = backward_flow(frame, prev_frame) if prev_frame is not None else None
            valid_gt = gt.sum() > 100
            gt_area = max(int(gt.sum()), 1)
            for method in args.methods:
                spec = methods[method]
                output = read_color(Path(spec["output"]) / clip / name)
                variant_mask = read_gray(Path(spec["mask"]) / clip / "masks" / name) > 0
                extra = variant_mask & (~gt)
                diff = np.mean(np.abs(output.astype(np.float32) - frame.astype(np.float32)), axis=2)
                if flow is None:
                    te = np.zeros(frame.shape[:2], dtype=np.float32)
                else:
                    te = temporal_error(output, warp(prev_outputs[method], flow))
                rows.append(
                    {
                        "clip": clip,
                        "frame": frame_path.stem,
                        "frame_index": idx,
                        "method": method,
                        "boundary_te": safe_mean(te, boundary),
                        "bd_te": safe_mean(te, bd),
                        "outside_changed_fraction": float(((diff > args.change_threshold) & (~gt)).sum() / max(int((~gt).sum()), 1)),
                        "outside_change_mean": safe_mean(diff / 255.0, ~gt),
                        "extra_mask_ratio": float(extra.sum() / gt_area) if valid_gt else np.nan,
                        "inside_change_mean": safe_mean(diff / 255.0, gt) if valid_gt else np.nan,
                        "residue_diff_le_10": float((diff[gt] <= args.change_threshold).mean()) if valid_gt else np.nan,
                    }
                )
                prev_outputs[method] = output
            prev_frame = frame

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args.methods)
    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary, clips)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

