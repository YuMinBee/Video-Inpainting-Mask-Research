#!/usr/bin/env python3
r"""Build hard-mask ablations for temporal occupancy memory.

The main variant follows the Choice-B definition used in the paper:
    S_t = (U \ M_t) & D_r(M_t) & {A > tau}
    M_t^ours = M_t | S_t

Additional variants test whether the gain comes from occupancy rather than just
mask area or boundary proximity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--occupancy-threshold", type=float, default=0.10)
    parser.add_argument("--max-dilation-radius", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


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


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def distance_closeness(mask: np.ndarray, radius: int) -> np.ndarray:
    outside = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    if radius <= 0:
        return np.zeros_like(dist, dtype=np.float32)
    return np.clip(1.0 - (dist / float(radius)), 0.0, 1.0).astype(np.float32)


def choose_area_matched_dilation(mask: np.ndarray, target_extra: int, max_radius: int) -> np.ndarray:
    if target_extra <= 0:
        return mask.copy()
    best = mask.copy()
    best_diff = abs(target_extra)
    for radius in range(1, max_radius + 1):
        candidate = dilate(mask, radius)
        extra = int((candidate & ~mask).sum())
        diff = abs(extra - target_extra)
        if diff < best_diff:
            best = candidate
            best_diff = diff
        if extra >= target_extra and diff > best_diff:
            break
    return best


def random_union_selection(mask: np.ndarray, union: np.ndarray, target_extra: int, rng: np.random.Generator) -> np.ndarray:
    final = mask.copy()
    candidates = np.flatnonzero((union & ~mask).reshape(-1))
    if target_extra <= 0 or len(candidates) == 0:
        return final
    k = min(target_extra, len(candidates))
    chosen = rng.choice(candidates, size=k, replace=False)
    flat = final.reshape(-1)
    flat[chosen] = True
    return final


def topk_score_selection(mask: np.ndarray, union: np.ndarray, score: np.ndarray, target_extra: int) -> np.ndarray:
    final = mask.copy()
    candidate = union & ~mask
    idx = np.flatnonzero(candidate.reshape(-1))
    if target_extra <= 0 or len(idx) == 0:
        return final
    k = min(target_extra, len(idx))
    scores = score.reshape(-1)[idx]
    # Stable tie-breaker: lexicographic through index after score ordering.
    order = np.lexsort((idx, -scores))[:k]
    chosen = idx[order]
    flat = final.reshape(-1)
    flat[chosen] = True
    return final


def process_clip(clip_dir: Path, output_root: Path, radius: int, tau: float, max_radius: int, seed: int) -> list[dict]:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    stack = np.stack(masks, axis=0)
    union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed + sum(ord(c) for c in clip))

    rows: list[dict] = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        gt_area = max(int(mask.sum()), 1)
        band = dilate(mask, radius)
        choice_b_extra = (union & ~mask) & band & (occupancy > tau)
        choice_b = mask | choice_b_extra
        target_extra = int(choice_b_extra.sum())

        occ_only = mask | ((union & ~mask) & (occupancy > tau))
        dist_only = mask | ((union & ~mask) & band)
        area_dilation = choose_area_matched_dilation(mask, target_extra, max_radius)
        random_union = random_union_selection(mask, union, target_extra, rng)
        closeness = distance_closeness(mask, radius)
        occ_dist_score = topk_score_selection(mask, union, occupancy * closeness, target_extra)

        variants = {
            "ours_choice_b_r10_t0p10": choice_b,
            "area_matched_dilation": area_dilation,
            "area_matched_random_union": random_union,
            "occupancy_only_t0p10": occ_only,
            "distance_only_r10": dist_only,
            "occ_dist_score_area_matched": occ_dist_score,
        }
        for variant, variant_mask in variants.items():
            write_mask(output_root / variant / clip / "masks" / name, variant_mask)
            rows.append(
                {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "variant": variant,
                    "original_area": int(mask.sum()),
                    "variant_area": int(variant_mask.sum()),
                    "extra_area": int((variant_mask & ~mask).sum()),
                    "extra_mask_ratio": float((variant_mask & ~mask).sum() / gt_area),
                    "target_choice_b_extra": target_extra,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    clips = args.clips
    if clips is None:
        clips = [p.name for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    all_rows = []
    for clip in clips:
        all_rows.extend(process_clip(args.probe_root / clip, args.output_root, args.radius, args.occupancy_threshold, args.max_dilation_radius, args.seed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_stats.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    variants = sorted({row["variant"] for row in all_rows})
    summary = []
    for variant in variants:
        rows = [row for row in all_rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "frames": len(rows),
                "extra_mask_ratio_mean": float(np.mean([row["extra_mask_ratio"] for row in rows])),
                "extra_area_mean": float(np.mean([row["extra_area"] for row in rows])),
            }
        )
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
