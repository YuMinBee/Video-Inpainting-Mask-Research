#!/usr/bin/env python3
"""Build confidence-budget mask variants.

The variants keep the two-radius Ours-Balanced structure, but replace the
occupancy threshold with a per-frame top-budget selection.  The total extra
mask budget is matched to a reference mask, typically Ours-Balanced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--reference-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--base-radius", type=int, default=5)
    parser.add_argument("--search-radius", type=int, default=30)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[10.0, 20.0, 30.0])
    parser.add_argument("--budget-multipliers", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    return parser.parse_args()


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write mask: {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    outside = (~mask).astype(np.uint8)
    return cv2.distanceTransform(outside, cv2.DIST_L2, 5).astype(np.float32)


def choose_top_budget(base: np.ndarray, candidate: np.ndarray, score: np.ndarray, selected_budget: int) -> np.ndarray:
    final = base.copy()
    idx = np.flatnonzero(candidate.reshape(-1))
    if selected_budget <= 0 or len(idx) == 0:
        return final
    k = min(selected_budget, len(idx))
    scores = score.reshape(-1)[idx]
    order = np.lexsort((idx, -scores))[:k]
    chosen = idx[order]
    flat = final.reshape(-1)
    flat[chosen] = True
    return final


def resolve_reference_mask(reference_root: Path, clip: str, name: str) -> Path:
    candidates = [
        reference_root / clip / "masks" / name,
        reference_root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing reference mask for {clip}/{name} under {reference_root}")


def variant_configs(alphas: list[float], sigmas: list[float], budgets: list[float]) -> list[dict]:
    configs: list[dict] = []
    for budget in budgets:
        configs.append(
            {
                "variant": f"conf_budget_occ_b{tag(budget)}",
                "source": "occupancy",
                "alpha": 1.0,
                "sigma": math.inf,
                "budget_multiplier": budget,
            }
        )
    for alpha in alphas:
        for sigma in sigmas:
            for budget in budgets:
                configs.append(
                    {
                        "variant": f"conf_budget_a{tag(alpha)}_s{tag(sigma)}_b{tag(budget)}",
                        "source": "occupancy_distance",
                        "alpha": alpha,
                        "sigma": sigma,
                        "budget_multiplier": budget,
                    }
                )
    return configs


def process_clip(
    clip_dir: Path,
    reference_root: Path,
    output_root: Path,
    configs: list[dict],
    base_radius: int,
    search_radius: int,
) -> list[dict]:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    stack = np.stack(masks, axis=0)
    union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0).astype(np.float32)

    rows: list[dict] = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        original_area = max(int(mask.sum()), 1)
        base = dilate(mask, base_radius)
        search = dilate(mask, search_radius)
        candidate = (union & ~base) & search
        dist = boundary_distance(mask)
        ref_mask = read_gray(resolve_reference_mask(reference_root, clip, name))
        ref_extra = int((ref_mask & ~mask).sum())
        base_extra = int((base & ~mask).sum())

        for cfg in configs:
            target_total_extra = int(round(ref_extra * cfg["budget_multiplier"]))
            selected_budget = max(target_total_extra - base_extra, 0)
            if cfg["source"] == "occupancy":
                score = occupancy
            else:
                score = np.power(occupancy, cfg["alpha"]) * np.exp(-dist / cfg["sigma"])
            final = choose_top_budget(base, candidate, score.astype(np.float32), selected_budget)
            variant = cfg["variant"]
            write_mask(output_root / variant / clip / "masks" / name, final)

            selected_extra = int((final & ~base).sum())
            extra = int((final & ~mask).sum())
            rows.append(
                {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "variant": variant,
                    "source": cfg["source"],
                    "alpha": cfg["alpha"],
                    "sigma": cfg["sigma"],
                    "budget_multiplier": cfg["budget_multiplier"],
                    "original_area": int(mask.sum()),
                    "reference_extra_area": ref_extra,
                    "target_total_extra_area": target_total_extra,
                    "base_extra_area": base_extra,
                    "selected_budget_area": selected_budget,
                    "selected_extra_area": selected_extra,
                    "extra_area": extra,
                    "extra_mask_ratio": float(extra / original_area),
                    "target_extra_ratio": float(target_total_extra / original_area),
                    "candidate_area": int(candidate.sum()),
                    "budget_shortfall_area": int(max(selected_budget - selected_extra, 0)),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "source": group[0]["source"],
                "alpha": group[0]["alpha"],
                "sigma": group[0]["sigma"],
                "budget_multiplier": group[0]["budget_multiplier"],
                "frames": len(group),
                "reference_extra_ratio_mean": float(np.mean([r["reference_extra_area"] / max(r["original_area"], 1) for r in group])),
                "target_extra_ratio_mean": float(np.mean([r["target_extra_ratio"] for r in group])),
                "extra_mask_ratio_mean": float(np.mean([r["extra_mask_ratio"] for r in group])),
                "base_extra_area_mean": float(np.mean([r["base_extra_area"] for r in group])),
                "selected_extra_area_mean": float(np.mean([r["selected_extra_area"] for r in group])),
                "budget_shortfall_area_mean": float(np.mean([r["budget_shortfall_area"] for r in group])),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    configs = variant_configs(args.alphas, args.sigmas, args.budget_multipliers)
    clips = args.clips
    if clips is None:
        clips = [p.name for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]

    all_rows: list[dict] = []
    for clip in clips:
        all_rows.extend(
            process_clip(
                args.probe_root / clip,
                args.reference_mask_root,
                args.output_root,
                configs,
                args.base_radius,
                args.search_radius,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "mask_stats.csv", all_rows)
    (args.output_root / "mask_stats.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    summary = summarize(all_rows)
    write_csv(args.output_root / "mask_summary.csv", summary)
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
