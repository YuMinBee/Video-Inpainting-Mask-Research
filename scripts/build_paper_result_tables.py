#!/usr/bin/env python3
"""Build paper-ready result tables across Driving clips and DAVIS."""

from __future__ import annotations

import csv
from pathlib import Path


DATASETS = [
    {
        "name": "Driving clips",
        "summary": Path("experiments/selective_expansion/full20_evaluation/summary.csv"),
        "clip_summary": Path("experiments/selective_expansion/full20_evaluation/clip_summary.csv"),
        "methods": {
            "Original": "Original",
            "Boundary-only": "Boundary-only",
            "Temporal union": "Temporal union",
            "Ours r10 t0.10": "Aggressive pixel r10 t0.10",
            "Ours r10 t0.15": "Aggressive pixel r10 t0.15",
        },
    },
    {
        "name": "DAVIS 2017 val",
        "summary": Path("experiments/davis2017_val/evaluation/summary.csv"),
        "clip_summary": Path("experiments/davis2017_val/evaluation/clip_summary.csv"),
        "methods": {
            "Original": "Original",
            "Boundary-only": "Boundary-only",
            "Temporal union": "Temporal union",
            "Ours r10 t0.10": "Ours r10 t0.10",
            "Ours r10 t0.15": "Ours r10 t0.15",
        },
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def format_result_row(label: str, row: dict[str, str]) -> str:
    return (
        f"| {label} | {float(row['boundary_te']):.6f} | "
        f"{float(row['outside_changed_fraction']):.6f} | "
        f"{float(row['extra_mask_ratio']):.4f} | "
        f"{float(row['residue_diff_le_10']):.4f} |"
    )


def win_counts(rows: list[dict[str, str]], methods: dict[str, str], ours_label: str) -> dict[str, int]:
    internal = methods[ours_label]
    clips = sorted({row["clip"] for row in rows})
    by_key = {(row["clip"], row["method"]): row for row in rows}
    counts = {
        "total": len(clips),
        "residue_lt_boundary": 0,
        "outside_lt_temporal_union": 0,
        "both": 0,
        "boundary_te_lt_original": 0,
        "large_residue_gap_vs_union": 0,
    }
    for clip in clips:
        ours = by_key[(clip, internal)]
        original = by_key[(clip, methods["Original"])]
        boundary = by_key[(clip, methods["Boundary-only"])]
        union = by_key[(clip, methods["Temporal union"])]
        residue_ok = float(ours["residue_diff_le_10"]) < float(boundary["residue_diff_le_10"])
        outside_ok = float(ours["outside_changed_fraction"]) < float(union["outside_changed_fraction"])
        boundary_te_ok = float(ours["boundary_te"]) < float(original["boundary_te"])
        residue_gap = float(ours["residue_diff_le_10"]) - float(union["residue_diff_le_10"])
        counts["residue_lt_boundary"] += int(residue_ok)
        counts["outside_lt_temporal_union"] += int(outside_ok)
        counts["both"] += int(residue_ok and outside_ok)
        counts["boundary_te_lt_original"] += int(boundary_te_ok)
        counts["large_residue_gap_vs_union"] += int(residue_gap > 0.10)
    return counts


def frac(count: int, total: int) -> str:
    return f"{count}/{total}"


def build() -> str:
    lines = [
        "# Paper Result Tables",
        "",
        "## Table 1. Driving Clips Results",
        "",
        "| Method | Boundary TE ↓ | Outside Changed ↓ | Extra Mask ↓ | Residue diff≤10 ↓ |",
        "|---|---:|---:|---:|---:|",
    ]

    all_win_rows = []
    for dataset in DATASETS:
        summary = row_by_method(read_csv(dataset["summary"]))
        if dataset["name"] != "Driving clips":
            lines.extend(
                [
                    "",
                    "## Table 2. DAVIS 2017 Val Results",
                    "",
                    "| Method | Boundary TE ↓ | Outside Changed ↓ | Extra Mask ↓ | Residue diff≤10 ↓ |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
        for label in ["Original", "Boundary-only", "Temporal union", "Ours r10 t0.10", "Ours r10 t0.15"]:
            lines.append(format_result_row(label, summary[dataset["methods"][label]]))

        clip_rows = read_csv(dataset["clip_summary"])
        for ours in ["Ours r10 t0.10", "Ours r10 t0.15"]:
            counts = win_counts(clip_rows, dataset["methods"], ours)
            all_win_rows.append((dataset["name"], ours, counts))

    lines.extend(
        [
            "",
            "## Table 3. Clip-Level Win Counts",
            "",
            "| Dataset | Method | Residue < Boundary-only ↑ | Outside < Temporal union ↑ | Both ↑ | Boundary TE < Original ↑ | Large residue gap vs union ↓ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, method, counts in all_win_rows:
        total = counts["total"]
        lines.append(
            f"| {dataset} | {method} | "
            f"{frac(counts['residue_lt_boundary'], total)} | "
            f"{frac(counts['outside_lt_temporal_union'], total)} | "
            f"{frac(counts['both'], total)} | "
            f"{frac(counts['boundary_te_lt_original'], total)} | "
            f"{frac(counts['large_residue_gap_vs_union'], total)} |"
        )

    lines.extend(
        [
            "",
            "## Short Interpretation",
            "",
            "- Boundary-only is the conservative flicker baseline, but it gives limited residue reduction.",
            "- Temporal union is the aggressive residue baseline, but it consistently increases outside-region changes.",
            "- The proposed selective expansion sits between the two: it keeps the residue benefit over Boundary-only in most clips while reducing over-removal relative to Temporal union.",
            "- DAVIS is used as an external validation set; it follows the same trade-off trend, with lower outside changes than Temporal union in every clip.",
            "",
            "## Ablation Summary",
            "",
            "| Method | Meaning |",
            "|---|---|",
            "| Original | raw mask |",
            "| Boundary-only | flicker smoothing only |",
            "| Temporal union | full temporal expansion |",
            "| Ours r10 t0.10 | residue-oriented selective expansion |",
            "| Ours r10 t0.15 | conservative selective expansion |",
            "",
            "Optional appendix variants:",
            "",
            "| Variant | Reason |",
            "|---|---|",
            "| r7 t0.10 | smaller radius |",
            "| r15 t0.10 | larger radius |",
            "| r10 t0.20 | higher threshold |",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = Path("experiments/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "SELECTIVE_EXPANSION_TABLES.md"
    out.write_text(build(), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
