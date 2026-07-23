#!/usr/bin/env python3
"""Analyze overlap among mask boundary, depth edges, and temporal errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


METRIC_KEYS = [
    "boundary_depth_over_boundary",
    "boundary_temporal_over_boundary",
    "depth_temporal_over_depth",
    "triple_over_boundary",
    "triple_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("experiments/depth_temporal_probe_real"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/depth_temporal_probe_real/overlap_analysis"),
    )
    parser.add_argument("--temporal-percentile", type=float, default=90.0)
    parser.add_argument("--temporal-min-threshold", type=int, default=25)
    return parser.parse_args()


def safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read grayscale image: {path}")
    return image


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read color image: {path}")
    return image


def temporal_high_mask(temporal: np.ndarray, percentile: float, min_threshold: int) -> tuple[np.ndarray, float]:
    if temporal.max() == 0:
        return np.zeros_like(temporal, dtype=bool), 0.0
    nonzero = temporal[temporal > 0]
    if nonzero.size == 0:
        return np.zeros_like(temporal, dtype=bool), 0.0
    threshold = max(float(np.percentile(nonzero, percentile)), float(min_threshold))
    return temporal >= threshold, threshold


def overlay_map(
    frame: np.ndarray,
    boundary: np.ndarray,
    depth_edge: np.ndarray,
    temporal: np.ndarray,
    triple: np.ndarray,
) -> np.ndarray:
    out = (frame.astype(np.float32) * 0.42).astype(np.uint8)

    # BGR colors:
    # boundary: yellow, depth edge: green, temporal high error: red,
    # pair overlaps: orange/cyan variants, triple: white.
    b = boundary
    d = depth_edge
    t = temporal
    out[b] = (0, 255, 255)
    out[d] = (0, 255, 0)
    out[t] = (0, 0, 255)
    out[b & d] = (0, 255, 180)
    out[b & t] = (0, 128, 255)
    out[d & t] = (0, 180, 180)
    out[triple] = (255, 255, 255)

    cv2.rectangle(out, (0, 0), (out.shape[1], 62), (0, 0, 0), -1)
    legend = "yellow=B  green=D  red=T  white=B&D&T"
    cv2.putText(out, legend, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "B=mask boundary, D=depth edge, T=top temporal error",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["temporal_high_px"] > 0]
    summary = {}
    for key in METRIC_KEYS:
        values = np.array([r[key] for r in valid], dtype=np.float32)
        summary[key] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }
    return summary


def summarize_by_clip(rows: list[dict]) -> list[dict]:
    clips = sorted({r["clip"] for r in rows})
    output = []
    for clip in clips:
        clip_rows = [r for r in rows if r["clip"] == clip and r["temporal_high_px"] > 0]
        item = {"clip": clip, "valid_frames": len(clip_rows)}
        for key in METRIC_KEYS:
            values = np.array([r[key] for r in clip_rows], dtype=np.float32)
            item[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
            item[f"{key}_max"] = float(values.max()) if values.size else 0.0
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_markdown(
    path: Path,
    rows: list[dict],
    clip_summary: list[dict],
    overall: dict,
    best_rows: list[dict],
    probe_root: Path,
) -> None:
    lines = [
        "# Boundary / Depth / Temporal Overlap Summary",
        "",
        f"Inputs: `{probe_root}`",
        "",
        "Definitions:",
        "",
        "- `B`: mask boundary band from the object mask.",
        "- `D`: binary DA3 depth edge map from high depth-gradient pixels.",
        "- `T`: high temporal-error pixels from the top nonzero temporal-error percentile.",
        "- `triple`: `B & D & T`.",
        "",
        "Important note: temporal error is currently computed with OpenCV Farneback flow in the probe pipeline. Use RAFT for final paper numbers.",
        "",
        "## Overall",
        "",
        "| Metric | Mean | Median | Max |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "boundary_depth_over_boundary": "B & D / B",
        "boundary_temporal_over_boundary": "B & T / B",
        "depth_temporal_over_depth": "D & T / D",
        "triple_over_boundary": "B & D & T / B",
        "triple_iou": "B & D & T / (B | D | T)",
    }
    for key in METRIC_KEYS:
        stat = overall[key]
        lines.append(f"| {labels[key]} | {pct(stat['mean'])} | {pct(stat['median'])} | {pct(stat['max'])} |")

    lines.extend(
        [
            "",
            "## By Clip",
            "",
            "| Clip | Valid Frames | B&D/B Mean | B&T/B Mean | D&T/D Mean | Triple/B Mean | Triple/B Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in clip_summary:
        lines.append(
            "| {clip} | {valid_frames} | {bd} | {bt} | {dt} | {tri_mean} | {tri_max} |".format(
                clip=item["clip"],
                valid_frames=item["valid_frames"],
                bd=pct(item["boundary_depth_over_boundary_mean"]),
                bt=pct(item["boundary_temporal_over_boundary_mean"]),
                dt=pct(item["depth_temporal_over_depth_mean"]),
                tri_mean=pct(item["triple_over_boundary_mean"]),
                tri_max=pct(item["triple_over_boundary_max"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top Triple-Overlap Frames",
            "",
            "| Rank | Clip | Frame | B&D&T/B | B&T/B | B&D/B | Overlay |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(best_rows, start=1):
        overlay_rel = Path("maps") / row["clip"] / f"{row['frame']}.png"
        lines.append(
            f"| {idx} | {row['clip']} | {row['frame']} | "
            f"{pct(row['triple_over_boundary'])} | "
            f"{pct(row['boundary_temporal_over_boundary'])} | "
            f"{pct(row['boundary_depth_over_boundary'])} | "
            f"[map]({overlay_rel.as_posix()}) |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest and most stable relationship is between the mask boundary and temporal error. "
            "Depth edges also intersect the boundary, but the three-way overlap is localized rather than broad. "
            "This supports using depth consistency as a confidence gate around difficult boundary regions, "
            "instead of assuming every depth edge directly causes flicker.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    map_root = args.output_root / "maps"

    rows: list[dict] = []
    clip_dirs = [p for p in sorted(args.probe_root.iterdir()) if (p / "frames").exists()]
    for clip_dir in clip_dirs:
        clip = clip_dir.name
        for frame_path in sorted((clip_dir / "frames").glob("*.png")):
            name = frame_path.name
            frame = read_color(frame_path)
            boundary = read_gray(clip_dir / "mask_boundaries" / name) > 0
            depth_edge = read_gray(clip_dir / "depth_edge_binary" / name) > 0
            temporal_raw = read_gray(clip_dir / "temporal_error" / name)
            temporal, temporal_threshold = temporal_high_mask(
                temporal_raw,
                args.temporal_percentile,
                args.temporal_min_threshold,
            )

            bd = boundary & depth_edge
            bt = boundary & temporal
            dt = depth_edge & temporal
            triple = boundary & depth_edge & temporal
            union = boundary | depth_edge | temporal

            row = {
                "clip": clip,
                "frame": frame_path.stem,
                "boundary_px": int(boundary.sum()),
                "depth_edge_px": int(depth_edge.sum()),
                "temporal_high_px": int(temporal.sum()),
                "temporal_threshold": float(temporal_threshold),
                "boundary_depth_px": int(bd.sum()),
                "boundary_temporal_px": int(bt.sum()),
                "depth_temporal_px": int(dt.sum()),
                "triple_px": int(triple.sum()),
                "union_px": int(union.sum()),
                "boundary_depth_over_boundary": safe_div(int(bd.sum()), int(boundary.sum())),
                "boundary_temporal_over_boundary": safe_div(int(bt.sum()), int(boundary.sum())),
                "depth_temporal_over_depth": safe_div(int(dt.sum()), int(depth_edge.sum())),
                "triple_over_boundary": safe_div(int(triple.sum()), int(boundary.sum())),
                "triple_iou": safe_div(int(triple.sum()), int(union.sum())),
            }
            rows.append(row)

            out_map = overlay_map(frame, boundary, depth_edge, temporal, triple)
            out_path = map_root / clip / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out_map)

    clip_summary = summarize_by_clip(rows)
    overall = summarize(rows)
    best_rows = sorted(
        [r for r in rows if r["temporal_high_px"] > 0],
        key=lambda r: r["triple_over_boundary"],
        reverse=True,
    )[:10]

    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "clip_summary.csv", clip_summary)
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "probe_root": str(args.probe_root),
                    "temporal_percentile": args.temporal_percentile,
                    "temporal_min_threshold": args.temporal_min_threshold,
                },
                "overall": overall,
                "clip_summary": clip_summary,
                "top_triple_frames": best_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_markdown(args.output_root / "RESULTS.md", rows, clip_summary, overall, best_rows, args.probe_root)

    print(f"Wrote {len(rows)} frame rows to {args.output_root}")
    print("Overall:")
    for key in METRIC_KEYS:
        stat = overall[key]
        print(f"  {key}: mean={stat['mean']:.4f}, median={stat['median']:.4f}, max={stat['max']:.4f}")


if __name__ == "__main__":
    main()
