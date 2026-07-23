#!/usr/bin/env python3
"""Build mask variants for the synthetic-GT probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


VARIANTS = [
    "boundary_only",
    "temporal_union",
    "ours_balanced",
    "area_matched_dilation",
    "area_matched_distance_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--base-radius", type=int, default=5)
    parser.add_argument("--search-radius", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-dilation-radius", type=int, default=120)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return image > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def choose_area_matched_dilation(mask: np.ndarray, target_extra: int, max_radius: int) -> tuple[np.ndarray, int]:
    if target_extra <= 0:
        return mask.copy(), 0
    best = mask.copy()
    best_radius = 0
    best_diff = abs(target_extra)
    for radius in range(1, max_radius + 1):
        candidate = dilate(mask, radius)
        extra = int((candidate & ~mask).sum())
        diff = abs(extra - target_extra)
        if diff < best_diff:
            best = candidate
            best_radius = radius
            best_diff = diff
        if extra >= target_extra and diff > best_diff:
            break
    return best, best_radius


def select_nearest_union(mask: np.ndarray, union: np.ndarray, target_extra: int) -> np.ndarray:
    final = mask.copy()
    candidate = union & ~mask
    idx = np.flatnonzero(candidate.reshape(-1))
    if target_extra <= 0 or len(idx) == 0:
        return final
    outside = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5).reshape(-1)
    k = min(target_extra, len(idx))
    order = np.lexsort((idx, dist[idx]))[:k]
    final.reshape(-1)[idx[order]] = True
    return final


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_clip(
    clip_dir: Path,
    output_root: Path,
    base_radius: int,
    search_radius: int,
    threshold: float,
    max_dilation_radius: int,
) -> list[dict]:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_mask(clip_dir / "masks" / p.name) for p in frame_paths]
    stack = np.stack(masks, axis=0)
    union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0).astype(np.float32)

    rows: list[dict] = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        raw_area = max(int(mask.sum()), 1)
        base = dilate(mask, base_radius)
        search = dilate(mask, search_radius)
        selected = (union & ~base) & search & (occupancy > threshold)
        ours = base | selected
        target_extra = int((ours & ~mask).sum())
        boundary_only = base
        temporal_union = union
        area_matched, matched_radius = choose_area_matched_dilation(mask, target_extra, max_dilation_radius)
        distance_only = select_nearest_union(mask, union, target_extra)

        variants = {
            "boundary_only": boundary_only,
            "temporal_union": temporal_union,
            "ours_balanced": ours,
            "area_matched_dilation": area_matched,
            "area_matched_distance_only": distance_only,
        }
        for variant, variant_mask in variants.items():
            write_mask(output_root / variant / clip / "masks" / name, variant_mask)
            extra = int((variant_mask & ~mask).sum())
            rows.append(
                {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "variant": variant,
                    "raw_area": int(mask.sum()),
                    "variant_area": int(variant_mask.sum()),
                    "extra_area": extra,
                    "extra_mask_ratio": float(extra / raw_area),
                    "target_extra_area": target_extra,
                    "matched_radius": matched_radius if variant == "area_matched_dilation" else "",
                    "candidate_limited": variant == "area_matched_distance_only" and extra < target_extra,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    clips = args.clips
    if clips is None:
        clips = [p.name for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir()]

    rows: list[dict] = []
    for clip in clips:
        rows.extend(
            process_clip(
                args.probe_root / clip,
                args.output_root,
                args.base_radius,
                args.search_radius,
                args.threshold,
                args.max_dilation_radius,
            )
        )

    write_csv(args.output_root / "mask_stats.csv", rows)
    summary = []
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "frames": len(subset),
                "extra_mask_ratio_mean": float(np.mean([r["extra_mask_ratio"] for r in subset])),
                "extra_area_mean": float(np.mean([r["extra_area"] for r in subset])),
                "candidate_limited_frames": int(sum(bool(r["candidate_limited"]) for r in subset)),
            }
        )
    write_csv(args.output_root / "mask_summary.csv", summary)
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

