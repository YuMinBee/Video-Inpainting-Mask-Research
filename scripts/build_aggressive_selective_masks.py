#!/usr/bin/env python3
"""Build aggressive selective-expansion masks for small ablations."""

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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--occupancy-threshold", type=float, required=True)
    parser.add_argument("--mode", choices=["pixel", "component"], default="pixel")
    parser.add_argument("--component-stat", choices=["mean", "max"], default="max")
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def select_components(extra: np.ndarray, occupancy: np.ndarray, tau: float, stat: str) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(extra.astype(np.uint8), connectivity=8)
    selected = np.zeros(extra.shape, dtype=bool)
    for label in range(1, num_labels):
        comp = labels == label
        if not np.any(comp):
            continue
        value = float(occupancy[comp].mean()) if stat == "mean" else float(occupancy[comp].max())
        if value > tau:
            selected |= comp
    return selected


def process_clip(clip_dir: Path, output_root: Path, radius: int, tau: float, mode: str, component_stat: str) -> dict:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) > 0 for p in frame_paths]
    stack = np.stack(masks, axis=0)
    temporal_union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0)

    sizes = {"original": [], "base": [], "selected": [], "final": []}
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        band = boundary_support(mask, radius)
        base = mask | ((temporal_union & (~mask)) & band)
        extra = temporal_union & (~base)
        if mode == "pixel":
            selected = extra & (occupancy > tau)
        else:
            selected = select_components(extra, occupancy, tau, component_stat)
        final = base | selected

        sizes["original"].append(int(mask.sum()))
        sizes["base"].append(int(base.sum()))
        sizes["selected"].append(int(selected.sum()))
        sizes["final"].append(int(final.sum()))
        link_or_copy(frame_path, output_root / clip / "frames" / name)
        write_image(output_root / clip / "masks" / name, final.astype(np.uint8) * 255)

    return {
        "clip": clip,
        "frames": len(frame_paths),
        "radius": radius,
        "occupancy_threshold": tau,
        "mode": mode,
        "component_stat": component_stat if mode == "component" else None,
        "original_px_mean": float(np.mean(sizes["original"])),
        "base_px_mean": float(np.mean(sizes["base"])),
        "selected_px_mean": float(np.mean(sizes["selected"])),
        "final_px_mean": float(np.mean(sizes["final"])),
        "final_over_original_mean": float(np.mean(sizes["final"]) / max(float(np.mean(sizes["original"])), 1.0)),
    }


def main() -> None:
    args = parse_args()
    clips = args.clips
    if clips is None:
        clips = [p.name for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    stats = [
        process_clip(args.probe_root / clip, args.output_root, args.radius, args.occupancy_threshold, args.mode, args.component_stat)
        for clip in clips
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
