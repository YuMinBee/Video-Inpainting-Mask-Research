#!/usr/bin/env python3
"""Evaluate arbitrary inpainting outputs with the selective-expansion metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_METHODS = [
    {
        "name": "Boundary-only",
        "output": Path("results/4090_rerun/aihub_subset_100/boundary_only/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/boundary_only"),
    },
    {
        "name": "Temporal union",
        "output": Path("results/4090_rerun/aihub_subset_100/temporal_union/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/temporal_union"),
    },
    {
        "name": "Ours r10 t0.10",
        "output": Path("results/4090_rerun/aihub_subset_100/ours_r10_t0p10/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/ours_r10_t0p10"),
    },
    {
        "name": "Ours r10 t0.15",
        "output": Path("results/4090_rerun/aihub_subset_100/ours_r10_t0p15/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/ours_r10_t0p15"),
    },
    {
        "name": "Blend TU occupancy",
        "output": Path("results/4090_rerun/aihub_subset_100/v2_confidence_blend/temporal_union_occupancy/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/temporal_union"),
    },
    {
        "name": "Blend TU distance",
        "output": Path("results/4090_rerun/aihub_subset_100/v2_confidence_blend/temporal_union_distance/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/temporal_union"),
    },
    {
        "name": "Blend TU occ_dist",
        "output": Path("results/4090_rerun/aihub_subset_100/v2_confidence_blend/temporal_union_occ_dist/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/temporal_union"),
    },
    {
        "name": "Blend Ours occ_dist",
        "output": Path("results/4090_rerun/aihub_subset_100/v2_confidence_blend/ours_r10_t0p10_occ_dist/propainter_outputs"),
        "mask": Path("experiments/aihub_subset_100/masks/ours_r10_t0p10"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/4090_rerun/aihub_subset_100/v2_confidence_blending/evaluation"))
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=OUTPUT_ROOT=MASK_ROOT",
        help="Add a method. If omitted, the built-in v2 comparison set is used.",
    )
    parser.add_argument("--change-threshold", type=float, default=10.0)
    return parser.parse_args()


def parse_methods(values: list[str]) -> list[dict[str, Path | str]]:
    if not values:
        return [dict(item) for item in DEFAULT_METHODS]
    methods = []
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise RuntimeError(f"Invalid --method value: {value}")
        name, output, mask = parts
        methods.append({"name": name, "output": Path(output), "mask": Path(mask)})
    return methods


def read_color(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def resolve_mask(mask_root: Path, clip: str, name: str) -> Path:
    candidates = [
        mask_root / clip / "masks" / name,
        mask_root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mask for {clip}/{name} under {mask_root}")


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


def summarize(rows: list[dict], method_names: list[str], group_key: str | None = None) -> list[dict]:
    keys = [
        "boundary_te",
        "bd_te",
        "outside_changed_fraction",
        "outside_change_mean",
        "extra_mask_ratio",
        "inside_change_mean",
        "residue_diff_le_10",
    ]
    groups: list[tuple[str | None, str]] = []
    if group_key is None:
        groups = [(None, method) for method in method_names]
    else:
        for group in sorted({row[group_key] for row in rows}):
            for method in method_names:
                groups.append((group, method))

    summary = []
    for group, method in groups:
        method_rows = [r for r in rows if r["method"] == method and (group_key is None or r[group_key] == group)]
        if not method_rows:
            continue
        item = {"method": method, "frames": len(method_rows)}
        if group_key is not None:
            item = {group_key: group, **item}
        for key in keys:
            item[key] = float(np.nanmean(np.array([r[key] for r in method_rows], dtype=np.float32)))
        summary.append(item)
    return summary


def write_markdown(path: Path, summary: list[dict], clips: list[str]) -> None:
    lines = [
        "# Generic Output Evaluation",
        "",
        f"Clips: `{', '.join(clips)}`",
        "",
        "| Method | Boundary TE ↓ | B&D TE ↓ | Outside Changed ↓ | Extra Mask ↓ | Inside Change ↑ | Residue diff<=10 ↓ |",
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
            "- `Outside Changed` is the fraction of non-mask pixels whose RGB change exceeds the threshold.",
            "- `Residue diff<=10` is the fraction of original-mask pixels still close to the input frame; lower means less residue.",
            "- Blended variants reuse the listed hard mask for extra-mask accounting, but their pixel output is confidence-gated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    method_names = [str(item["name"]) for item in methods]
    clips = args.clips or sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir())

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
            depth_edge_path = clip_dir / "depth_edge_binary" / name
            depth_edge = read_gray(depth_edge_path) > 0 if depth_edge_path.exists() else boundary
            bd = boundary & depth_edge
            flow = backward_flow(frame, prev_frame) if prev_frame is not None else None
            valid_gt = gt.sum() > 100
            gt_area = max(int(gt.sum()), 1)

            for spec in methods:
                method = str(spec["name"])
                output = read_color(Path(spec["output"]) / clip / name)
                variant_mask = read_gray(resolve_mask(Path(spec["mask"]), clip, name)) > 0
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
    summary = summarize(rows, method_names)
    clip_summary = summarize(rows, method_names, group_key="clip")
    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "clip_summary.csv", clip_summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary, clips)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

