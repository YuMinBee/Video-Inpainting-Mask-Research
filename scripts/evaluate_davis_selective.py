#!/usr/bin/env python3
"""Evaluate selective temporal expansion on DAVIS validation clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = {
    "Original": {
        "output": "results/davis2017_val/original/propainter_outputs",
        "mask": "experiments/davis2017_val/probe",
    },
    "Boundary-only": {
        "output": "results/davis2017_val/boundary_only/propainter_outputs",
        "mask": "experiments/davis2017_val/masks/basic/boundary_only_mask",
    },
    "Temporal union": {
        "output": "results/davis2017_val/temporal_union/propainter_outputs",
        "mask": "experiments/davis2017_val/masks/temporal_union",
    },
    "Ours r10 t0.10": {
        "output": "results/davis2017_val/ours_r10_t0p10/propainter_outputs",
        "mask": "experiments/davis2017_val/masks/ours_r10_t0p10",
    },
    "Ours r10 t0.15": {
        "output": "results/davis2017_val/ours_r10_t0p15/propainter_outputs",
        "mask": "experiments/davis2017_val/masks/ours_r10_t0p15",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/davis2017_val/probe"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/davis2017_val/evaluation"))
    parser.add_argument("--methods", nargs="+", choices=METHODS.keys(), default=list(METHODS.keys()))
    parser.add_argument("--clips", nargs="+")
    parser.add_argument(
        "--method-output",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method output root.",
    )
    parser.add_argument(
        "--method-mask",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method mask root.",
    )
    parser.add_argument("--change-threshold", type=float, default=10.0)
    parser.add_argument("--large-residue-gap", type=float, default=0.10)
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


def aggregate(rows: list[dict], group_keys: list[str]) -> list[dict]:
    metric_keys = [
        "boundary_te",
        "outside_changed_fraction",
        "outside_change_mean",
        "extra_mask_ratio",
        "inside_change_mean",
        "residue_diff_le_10",
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    out = []
    for key, group in sorted(groups.items()):
        item = dict(zip(group_keys, key))
        item["frames"] = len(group)
        for metric in metric_keys:
            item[metric] = float(np.nanmean(np.array([r[metric] for r in group], dtype=np.float32)))
        out.append(item)
    return out


def write_results_md(path: Path, summary: list[dict], clips: list[str]) -> None:
    lines = [
        "# DAVIS 2017 Val Selective Expansion Evaluation",
        "",
        f"Clips: {len(clips)} DAVIS 2017 val clips",
        "",
        "| Method | Boundary TE ↓ | Outside Changed ↓ | Extra Mask ↓ | Residue diff≤10 ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['boundary_te']:.6f} | "
            f"{row['outside_changed_fraction']:.6f} | {row['extra_mask_ratio']:.4f} | "
            f"{row['residue_diff_le_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Original uses DAVIS annotation masks directly.",
            "- Boundary-only expands only near the original mask boundary.",
            "- Temporal union uses the full temporal mask union.",
            "- Ours r10 t0.10 is the main selective expansion variant.",
            "- Ours r10 t0.15 is the conservative selective expansion variant.",
            "- Residue proxy is the fraction of original-mask pixels where the output remains close to the original frame (`diff<=10`).",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_win_counts(path: Path, clip_summary: list[dict], large_gap: float) -> None:
    by_clip_method = {(row["clip"], row["method"]): row for row in clip_summary}
    clips = sorted({row["clip"] for row in clip_summary})
    available_methods = {row["method"] for row in clip_summary}
    ours_methods = [m for m in ["Ours r10 t0.10", "Ours r10 t0.15"] if m in available_methods]
    has_boundary = "Boundary-only" in available_methods

    rows = []
    for method in ours_methods:
        residue_wins = 0
        outside_wins = 0
        both = 0
        boundary_te_wins = 0
        large_gaps = 0
        for clip in clips:
            ours = by_clip_method[(clip, method)]
            original = by_clip_method[(clip, "Original")]
            union = by_clip_method[(clip, "Temporal union")]
            residue_ok = False
            if has_boundary:
                boundary = by_clip_method[(clip, "Boundary-only")]
                residue_ok = ours["residue_diff_le_10"] < boundary["residue_diff_le_10"]
            outside_ok = ours["outside_changed_fraction"] < union["outside_changed_fraction"]
            boundary_te_ok = ours["boundary_te"] < original["boundary_te"]
            gap = ours["residue_diff_le_10"] - union["residue_diff_le_10"]
            residue_wins += int(residue_ok)
            outside_wins += int(outside_ok)
            both += int(residue_ok and outside_ok)
            boundary_te_wins += int(boundary_te_ok)
            large_gaps += int(gap > large_gap)
        rows.append(
            {
                "method": method,
                "residue_lt_boundary": residue_wins,
                "outside_lt_temporal_union": outside_wins,
                "both_satisfied": both,
                "boundary_te_lt_original": boundary_te_wins,
                "large_residue_gap_vs_union": large_gaps,
                "total": len(clips),
            }
        )

    lines = [
        "# DAVIS 2017 Val Clip-level Win Counts",
        "",
        "| Criterion | " + " | ".join(ours_methods) + " |",
        "|---|" + "---:|" * len(ours_methods),
    ]
    criteria = [
        ("Outside < Temporal union", "outside_lt_temporal_union"),
        ("Boundary TE < Original", "boundary_te_lt_original"),
        (f"Large residue gap vs union > {large_gap:.2f}", "large_residue_gap_vs_union"),
    ]
    if has_boundary:
        criteria.insert(0, ("Residue < Boundary-only", "residue_lt_boundary"))
        criteria.insert(2, ("Both satisfied", "both_satisfied"))
    by_method = {row["method"]: row for row in rows}
    for label, key in criteria:
        vals = [f"{by_method[m][key]}/{by_method[m]['total']}" for m in ours_methods]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = apply_method_overrides(args.method_output, args.method_mask)
    clips = args.clips or sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir())
    rows = []

    for clip in clips:
        frame_paths = sorted((args.probe_root / clip / "frames").glob("*.png"))
        prev_frame = None
        prev_outputs: dict[str, np.ndarray] = {}
        for idx, frame_path in enumerate(frame_paths):
            name = frame_path.name
            frame = read_color(frame_path)
            gt = read_gray(args.probe_root / clip / "masks" / name) > 0
            boundary = read_gray(args.probe_root / clip / "mask_boundaries" / name) > 0
            flow = backward_flow(frame, prev_frame) if prev_frame is not None else None
            gt_area = max(int(gt.sum()), 1)
            valid_gt = gt.sum() > 100

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
    clip_summary = aggregate(rows, ["clip", "method"])
    summary = aggregate(rows, ["method"])

    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "clip_summary.csv", clip_summary)
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_md(args.output_root / "RESULTS.md", summary, clips)
    write_win_counts(args.output_root / "WIN_COUNTS.md", clip_summary, args.large_residue_gap)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
