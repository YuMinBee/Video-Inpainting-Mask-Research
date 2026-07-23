#!/usr/bin/env python3
"""Build local boundary-only and occupancy-selective expansion masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLIPS = [
    "I-211025_O02012_T01",
    "I-210618_I01001_W01",
    "I-210718_O08036_W04",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/selective_expansion/basic"))
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--occupancy-threshold", type=float, default=0.4)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {src}")
        write_image(dst, image)


def boundary_support(mask: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0


def process_clip(clip_dir: Path, output_root: Path, radius: int, tau: float) -> dict:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) > 0 for p in frame_paths]
    stack = np.stack(masks, axis=0)
    temporal_union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0)

    stats = {
        "clip": clip,
        "frames": len(frame_paths),
        "radius": radius,
        "occupancy_threshold": tau,
        "original_px_mean": float(np.mean([m.sum() for m in masks])),
        "boundary_only_px_mean": 0.0,
        "selective_px_mean": 0.0,
    }
    bo_sizes = []
    sel_sizes = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        extra = temporal_union & (~mask)
        band = boundary_support(mask, radius)
        local_extra = extra & band
        selected = local_extra & (occupancy > tau)
        boundary_only = mask | local_extra
        selective = mask | selected

        bo_sizes.append(boundary_only.sum())
        sel_sizes.append(selective.sum())
        for variant, variant_mask in [
            ("boundary_only_mask", boundary_only),
            ("selective_expansion", selective),
        ]:
            link_or_copy(frame_path, output_root / variant / clip / "frames" / name)
            write_image(output_root / variant / clip / "masks" / name, variant_mask.astype(np.uint8) * 255)

    stats["boundary_only_px_mean"] = float(np.mean(bo_sizes))
    stats["selective_px_mean"] = float(np.mean(sel_sizes))
    return stats


def main() -> None:
    args = parse_args()
    stats = []
    for clip in args.clips:
        stats.append(process_clip(args.probe_root / clip, args.output_root, args.radius, args.occupancy_threshold))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
