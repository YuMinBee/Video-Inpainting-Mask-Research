#!/usr/bin/env python3
"""Evaluate mask-level correction efficiency against GT masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mask-root", type=Path, required=True)
    parser.add_argument("--raw-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        required=True,
        metavar="NAME=MASK_ROOT",
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> list[dict[str, Path | str]]:
    methods = []
    for value in values:
        parts = value.split("=", 1)
        if len(parts) != 2:
            raise RuntimeError(f"Invalid --method value: {value}")
        methods.append({"name": parts[0], "mask": Path(parts[1])})
    return methods


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return image > 0


def resolve_mask(root: Path, clip: str, name: str) -> Path:
    candidates = [
        root / clip / "masks" / name,
        root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mask for {clip}/{name} under {root}")


def safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den > 0 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], method_names: list[str], group_key: str | None = None) -> list[dict]:
    keys = [
        "raw_missing_ratio",
        "raw_over_ratio",
        "missing_recovery",
        "added_precision",
        "false_added",
        "extra_budget",
        "correct_added_ratio",
        "added_area",
        "hit_area",
        "hidden_area",
    ]
    groups = sorted({r[group_key] for r in rows}) if group_key else [None]
    summary = []
    for group in groups:
        for method in method_names:
            subset = [r for r in rows if r["method"] == method and (group_key is None or r[group_key] == group)]
            if not subset:
                continue
            item = {"method": method, "frames": len(subset)}
            if group_key is not None:
                item = {group_key: group, **item}
            for key in keys:
                item[key] = float(np.nanmean(np.array([r[key] for r in subset], dtype=np.float32)))
            summary.append(item)
    return summary


def write_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "# Mask Correction Efficiency",
        "",
        "| Method | Missing recovery ↑ | Added precision ↑ | False added ↓ | Extra budget ↓ | Raw missing ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['missing_recovery']:.4f} | "
            f"{row['added_precision']:.4f} | {row['false_added']:.4f} | "
            f"{row['extra_budget']:.4f} | {row['raw_missing_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- `H = M_gt \\ M_raw` is the missed GT object region.",
            "- `A = M_v \\ M_raw` is the area newly added by a method.",
            "- `Missing recovery = |A ∩ H| / |H|`.",
            "- `Added precision = |A ∩ H| / |A|`.",
            "- `False added = |A \\ M_gt| / |M_gt|`.",
            "- `Extra budget = |A| / |M_gt|`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    method_names = [str(item["name"]) for item in methods]
    clips = args.clips or sorted(p.name for p in args.gt_mask_root.iterdir() if (p / "masks").is_dir())

    rows: list[dict] = []
    for clip in clips:
        frame_paths = sorted((args.gt_mask_root / clip / "masks").glob("*.png"))
        for gt_path in frame_paths:
            name = gt_path.name
            gt = read_mask(gt_path)
            raw = read_mask(resolve_mask(args.raw_mask_root, clip, name))
            if raw.shape != gt.shape:
                raw = cv2.resize(raw.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            gt_area = max(int(gt.sum()), 1)
            hidden = gt & ~raw
            raw_over = raw & ~gt
            hidden_area = int(hidden.sum())
            raw_over_area = int(raw_over.sum())

            for spec in methods:
                method = str(spec["name"])
                variant = read_mask(resolve_mask(Path(spec["mask"]), clip, name))
                if variant.shape != gt.shape:
                    variant = cv2.resize(
                        variant.astype(np.uint8),
                        (gt.shape[1], gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0

                added = variant & ~raw
                hit = added & hidden
                false_added = added & ~gt
                added_area = int(added.sum())
                hit_area = int(hit.sum())
                false_area = int(false_added.sum())

                rows.append(
                    {
                        "clip": clip,
                        "frame": gt_path.stem,
                        "method": method,
                        "gt_area": int(gt.sum()),
                        "raw_area": int(raw.sum()),
                        "variant_area": int(variant.sum()),
                        "hidden_area": hidden_area,
                        "raw_over_area": raw_over_area,
                        "added_area": added_area,
                        "hit_area": hit_area,
                        "false_added_area": false_area,
                        "raw_missing_ratio": safe_ratio(hidden_area, gt_area),
                        "raw_over_ratio": safe_ratio(raw_over_area, gt_area),
                        "missing_recovery": safe_ratio(hit_area, hidden_area),
                        "added_precision": safe_ratio(hit_area, added_area),
                        "false_added": safe_ratio(false_area, gt_area),
                        "extra_budget": safe_ratio(added_area, gt_area),
                        "correct_added_ratio": safe_ratio(hit_area, gt_area),
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, method_names)
    clip_summary = summarize(rows, method_names, group_key="clip")
    write_csv(args.output_root / "frame_mask_correction.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "clip_summary.csv", clip_summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

