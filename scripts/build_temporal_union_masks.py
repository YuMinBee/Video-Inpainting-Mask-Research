#!/usr/bin/env python3
"""Build temporal-union and depth-limited temporal-union masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_real"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/mask_union"))
    parser.add_argument("--depth-edge-dilate", type=int, default=5)
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


def clean_mask(mask: np.ndarray, close_size: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if close_size > 1:
        k = close_size if close_size % 2 == 1 else close_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def depth_limited_union(current: np.ndarray, union: np.ndarray, depth_edge: np.ndarray, edge_dilate: int) -> np.ndarray:
    current_b = current > 0
    union_b = union > 0
    if not np.any(current_b):
        return current.copy()

    edge = depth_edge > 0
    if edge_dilate > 1:
        k = edge_dilate if edge_dilate % 2 == 1 else edge_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.dilate(edge.astype(np.uint8), kernel) > 0

    allowed = union_b & (~edge)
    seed = current_b & allowed
    if not np.any(seed):
        return current.copy()

    # Keep only temporal-union pixels connected to the current mask without
    # crossing a high-depth-gradient barrier.
    num_labels, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    seed_labels = np.unique(labels[seed])
    keep = np.isin(labels, seed_labels) & union_b
    keep |= current_b
    return keep.astype(np.uint8) * 255


def process_clip(clip_dir: Path, output_root: Path, edge_dilate: int, union_close: int) -> dict:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    union = clean_mask(np.maximum.reduce(masks), union_close)

    stats = {
        "clip": clip,
        "frames": len(frame_paths),
        "original_mask_px_mean": float(np.mean([np.count_nonzero(m) for m in masks])),
        "temporal_union_px": int(np.count_nonzero(union)),
        "depth_limited_px_mean": 0.0,
    }
    dl_sizes = []

    for variant in ["temporal_union", "depth_limited_union"]:
        for frame_path in frame_paths:
            link_or_copy(frame_path, output_root / variant / clip / "frames" / frame_path.name)

    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        depth_edge = read_gray(clip_dir / "depth_edge_binary" / name)
        limited = depth_limited_union(mask, union, depth_edge, edge_dilate)
        dl_sizes.append(np.count_nonzero(limited))
        write_image(output_root / "temporal_union" / clip / "masks" / name, union)
        write_image(output_root / "depth_limited_union" / clip / "masks" / name, limited)

    stats["depth_limited_px_mean"] = float(np.mean(dl_sizes)) if dl_sizes else 0.0
    return stats


def main() -> None:
    args = parse_args()
    clip_dirs = sorted([p for p in args.probe_root.iterdir() if (p / "masks").exists()])
    stats = [process_clip(p, args.output_root, args.depth_edge_dilate, args.union_close) for p in clip_dirs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_union_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote mask variants for {len(stats)} clips to {args.output_root}")


if __name__ == "__main__":
    main()
