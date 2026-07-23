#!/usr/bin/env python3
"""Rank success and failure cases from frame-level evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHOD_MAP = {
    "Original": "orig",
    "Boundary-only": "boundary",
    "Temporal union": "union",
    "Aggressive pixel r10 t0.10": "ours10",
    "Aggressive pixel r10 t0.15": "ours15",
    "Ours r10 t0.10": "ours10",
    "Ours r10 t0.15": "ours15",
}
KEYS = ["boundary_te", "outside_changed_fraction", "extra_mask_ratio", "residue_diff_le_10"]
REQUIRED = ["orig", "boundary", "union", "ours10", "ours15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("experiments/4090_rerun/aihub_subset_100/evaluation_with_boundary/frame_metrics.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/4090_rerun/aihub_subset_100/failure_case_ranking"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def collect_clip_rows(metrics_path: Path) -> list[dict[str, float | str]]:
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with metrics_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method not in METHOD_MAP:
                continue
            prefix = METHOD_MAP[method]
            for key in KEYS:
                value = row[key]
                if value == "" or value.lower() == "nan":
                    continue
                acc[row["clip"]][prefix][key].append(float(value))

    rows = []
    for clip, by_method in sorted(acc.items()):
        if not all(method in by_method for method in REQUIRED):
            continue
        item: dict[str, float | str] = {"clip": clip}
        for prefix in REQUIRED:
            for key in KEYS:
                item[f"{prefix}_{key}"] = mean(by_method[prefix][key])
        item["success_tradeoff_score"] = (
            item["boundary_residue_diff_le_10"] - item["ours10_residue_diff_le_10"]
            + item["union_outside_changed_fraction"] - item["ours10_outside_changed_fraction"]
        )
        item["union_over_removal_score"] = item["union_outside_changed_fraction"] - item["ours10_outside_changed_fraction"]
        item["ours_residue_limitation_score"] = item["ours10_residue_diff_le_10"] - item["union_residue_diff_le_10"]
        item["ours_over_removal_limitation_score"] = item["ours10_outside_changed_fraction"] - item["boundary_outside_changed_fraction"]
        item["conservative_outside_gain_score"] = item["ours10_outside_changed_fraction"] - item["ours15_outside_changed_fraction"]
        rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["clip"]
    for prefix in REQUIRED:
        for key in KEYS:
            fieldnames.append(f"{prefix}_{key}")
    fieldnames.extend([
        "success_tradeoff_score",
        "union_over_removal_score",
        "ours_residue_limitation_score",
        "ours_over_removal_limitation_score",
        "conservative_outside_gain_score",
    ])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_clip_rows(args.metrics)
    write_csv(args.output_root / "clip_failure_ranking.csv", rows)

    categories = {
        "balanced_success": ("success_tradeoff_score", "Ours improves residue over Boundary-only while reducing outside changes relative to Temporal union."),
        "temporal_union_over_removal": ("union_over_removal_score", "Temporal union changes much more outside region than Ours."),
        "ours_residue_limitation": ("ours_residue_limitation_score", "Ours leaves more residue than Temporal union."),
        "ours_over_removal_limitation": ("ours_over_removal_limitation_score", "Ours changes more outside region than Boundary-only."),
        "conservative_variant_helpful": ("conservative_outside_gain_score", "The conservative variant reduces outside changes relative to the main setting."),
    }
    top = {}
    for name, (score_key, note) in categories.items():
        top[name] = {
            "score": score_key,
            "note": note,
            "clips": sorted(rows, key=lambda r: r[score_key], reverse=True)[: args.top_k],
        }
    (args.output_root / "top_failure_cases.json").write_text(json.dumps(top, indent=2), encoding="utf-8")

    lines = [
        "# Metric-Based Case Ranking",
        "",
        "All scores are computed from clip-level means.",
        "",
    ]
    for name, data in top.items():
        score_key = data["score"]
        lines.extend([
            f"## {name}",
            "",
            data["note"],
            "",
            "| Rank | Clip | Score | Ours outside | Ours residue | Union outside | Union residue | Boundary outside | Boundary residue |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(data["clips"], 1):
            lines.append(
                f"| {rank} | `{row['clip']}` | {row[score_key]:.4f} | "
                f"{row['ours10_outside_changed_fraction']:.4f} | {row['ours10_residue_diff_le_10']:.4f} | "
                f"{row['union_outside_changed_fraction']:.4f} | {row['union_residue_diff_le_10']:.4f} | "
                f"{row['boundary_outside_changed_fraction']:.4f} | {row['boundary_residue_diff_le_10']:.4f} |"
            )
        lines.append("")
    (args.output_root / "TOP_CASES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote rankings for {len(rows)} clips to {args.output_root}")


if __name__ == "__main__":
    main()