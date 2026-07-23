#!/usr/bin/env python3
"""Prepare DAVIS clips in the probe layout used by the inpainting experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-root", type=Path, default=Path("dataset/davis/DAVIS"))
    parser.add_argument("--split-file", type=Path, default=Path("dataset/davis/DAVIS/ImageSets/2017/val.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/davis2017_val/probe"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 keeps all frames")
    return parser.parse_args()


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = (mask > 0).astype(np.uint8)
    edge = cv2.dilate(binary, kernel) ^ cv2.erode(binary, kernel)
    return edge.astype(np.uint8) * 255


def process_clip(davis_root: Path, output_root: Path, clip: str, max_frames: int) -> dict:
    image_dir = davis_root / "JPEGImages" / "480p" / clip
    mask_dir = davis_root / "Annotations" / "480p" / clip
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]

    stats = {"clip": clip, "frames": len(image_paths), "mask_px_mean": 0.0}
    mask_sizes = []
    for image_path in image_paths:
        stem = image_path.stem
        image = read_color(image_path)
        ann = read_gray(mask_dir / f"{stem}.png")
        mask = (ann > 0).astype(np.uint8) * 255
        mask_sizes.append(int(np.count_nonzero(mask)))
        write_image(output_root / clip / "frames" / f"{stem}.png", image)
        write_image(output_root / clip / "masks" / f"{stem}.png", mask)
        write_image(output_root / clip / "mask_boundaries" / f"{stem}.png", boundary(mask))

    stats["mask_px_mean"] = float(np.mean(mask_sizes)) if mask_sizes else 0.0
    return stats


def main() -> None:
    args = parse_args()
    clips = [line.strip() for line in args.split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = [process_clip(args.davis_root, args.output_root, clip, args.max_frames) for clip in clips]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "probe_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
