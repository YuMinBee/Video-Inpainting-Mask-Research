#!/usr/bin/env python3
"""Build temporal-union masks without depth inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--union-close", type=int, default=5)
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


def clean(mask: np.ndarray, close_size: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if close_size > 1:
        k = close_size if close_size % 2 == 1 else close_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def process_clip(clip_dir: Path, output_root: Path, union_close: int) -> dict:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    union = clean(np.maximum.reduce(masks), union_close)
    for frame_path in frame_paths:
        link_or_copy(frame_path, output_root / clip / "frames" / frame_path.name)
        write_image(output_root / clip / "masks" / frame_path.name, union)
    return {
        "clip": clip,
        "frames": len(frame_paths),
        "original_px_mean": float(np.mean([np.count_nonzero(m) for m in masks])),
        "temporal_union_px": int(np.count_nonzero(union)),
    }


def main() -> None:
    args = parse_args()
    clip_dirs = [p for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    if args.clips:
        wanted = set(args.clips)
        clip_dirs = [p for p in clip_dirs if p.name in wanted]
    stats = [process_clip(p, args.output_root, args.union_close) for p in clip_dirs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
