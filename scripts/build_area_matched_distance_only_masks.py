#!/usr/bin/env python3
"""Build area-matched distance-only temporal-union masks.

For each frame, this control matches the extra area of a reference mask while
selecting temporal-union candidates solely by proximity to the current mask.
It removes the occupancy gate but keeps the temporal-union candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--reference-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
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


def resolve_reference(reference_root: Path, clip: str, name: str) -> Path:
    candidates = [
        reference_root / clip / "masks" / name,
        reference_root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing reference mask for {clip}/{name} under {reference_root}")


def select_nearest_union(mask: np.ndarray, union: np.ndarray, target_extra: int) -> np.ndarray:
    final = mask.copy()
    candidate = union & ~mask
    idx = np.flatnonzero(candidate.reshape(-1))
    if target_extra <= 0 or len(idx) == 0:
        return final

    outside = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5).reshape(-1)
    k = min(target_extra, len(idx))
    # Sort by distance to current mask, then by index for deterministic ties.
    order = np.lexsort((idx, dist[idx]))[:k]
    chosen = idx[order]
    flat = final.reshape(-1)
    flat[chosen] = True
    return final


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_clip(clip_dir: Path, reference_root: Path, output_root: Path) -> list[dict]:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    union = np.any(np.stack(masks, axis=0), axis=0)

    rows: list[dict] = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        reference = read_gray(resolve_reference(reference_root, clip, name))
        if reference.shape != mask.shape:
            reference = cv2.resize(
                reference.astype(np.uint8),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        target_extra = int((reference & ~mask).sum())
        final = select_nearest_union(mask, union, target_extra)
        write_mask(output_root / clip / "masks" / name, final)

        gt_area = max(int(mask.sum()), 1)
        selected_extra = int((final & ~mask).sum())
        rows.append(
            {
                "clip": clip,
                "frame": frame_path.stem,
                "original_area": int(mask.sum()),
                "union_extra_area": int((union & ~mask).sum()),
                "reference_extra_area": target_extra,
                "selected_extra_area": selected_extra,
                "reference_extra_mask_ratio": float(target_extra / gt_area),
                "selected_extra_mask_ratio": float(selected_extra / gt_area),
                "extra_area_abs_error": abs(selected_extra - target_extra),
                "candidate_limited": selected_extra < target_extra,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    clips = args.clips
    if clips is None:
        clips = [
            p.name
            for p in sorted(args.probe_root.iterdir())
            if (p / "frames").is_dir() and (p / "masks").is_dir()
        ]

    rows: list[dict] = []
    for clip in clips:
        rows.extend(process_clip(args.probe_root / clip, args.reference_mask_root, args.output_root))

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "mask_stats.csv", rows)
    (args.output_root / "mask_stats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "frames": len(rows),
        "reference_extra_mask_ratio_mean": float(np.mean([r["reference_extra_mask_ratio"] for r in rows])),
        "selected_extra_mask_ratio_mean": float(np.mean([r["selected_extra_mask_ratio"] for r in rows])),
        "extra_area_abs_error_mean": float(np.mean([r["extra_area_abs_error"] for r in rows])),
        "candidate_limited_frames": int(sum(1 for r in rows if r["candidate_limited"])),
    }
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
