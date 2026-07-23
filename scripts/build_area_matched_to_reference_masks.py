#!/usr/bin/env python3
"""Build dilation masks whose extra area matches a reference mask per frame."""

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
    parser.add_argument("--max-dilation-radius", type=int, default=120)
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


def resolve_reference(reference_root: Path, clip: str, name: str) -> Path:
    candidates = [
        reference_root / clip / "masks" / name,
        reference_root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing reference mask for {clip}/{name} under {reference_root}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_clip(clip_dir: Path, reference_root: Path, output_root: Path, max_radius: int) -> list[dict]:
    clip = clip_dir.name
    rows: list[dict] = []
    for frame_path in sorted((clip_dir / "frames").glob("*.png")):
        name = frame_path.name
        original = read_gray(clip_dir / "masks" / name)
        reference = read_gray(resolve_reference(reference_root, clip, name))
        if reference.shape != original.shape:
            reference = cv2.resize(
                reference.astype(np.uint8),
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        target_extra = int((reference & ~original).sum())
        matched, radius = choose_area_matched_dilation(original, target_extra, max_radius)
        write_mask(output_root / clip / "masks" / name, matched)

        gt_area = max(int(original.sum()), 1)
        matched_extra = int((matched & ~original).sum())
        rows.append(
            {
                "clip": clip,
                "frame": frame_path.stem,
                "original_area": int(original.sum()),
                "reference_extra_area": target_extra,
                "matched_extra_area": matched_extra,
                "matched_radius": radius,
                "reference_extra_mask_ratio": float(target_extra / gt_area),
                "matched_extra_mask_ratio": float(matched_extra / gt_area),
                "extra_area_abs_error": abs(matched_extra - target_extra),
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
        rows.extend(process_clip(args.probe_root / clip, args.reference_mask_root, args.output_root, args.max_dilation_radius))

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "mask_stats.csv", rows)
    (args.output_root / "mask_stats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "frames": len(rows),
        "reference_extra_mask_ratio_mean": float(np.mean([r["reference_extra_mask_ratio"] for r in rows])),
        "matched_extra_mask_ratio_mean": float(np.mean([r["matched_extra_mask_ratio"] for r in rows])),
        "extra_area_abs_error_mean": float(np.mean([r["extra_area_abs_error"] for r in rows])),
        "matched_radius_mean": float(np.mean([r["matched_radius"] for r in rows])),
        "matched_radius_median": float(np.median([r["matched_radius"] for r in rows])),
    }
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
