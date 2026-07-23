#!/usr/bin/env python3
"""Run Depth Anything 3 on prepared probe frames and save depth maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, default=Path("experiments/depth_temporal_probe"))
    parser.add_argument("--output-root", type=Path, default=Path("results/da3_depths"))
    parser.add_argument("--model", default="depth-anything/DA3-BASE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=3)
    return parser.parse_args()


def normalize_u8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo = float(np.percentile(depth, 2))
    hi = float(np.percentile(depth, 98))
    if hi <= lo + 1e-6:
        return np.zeros(depth.shape, dtype=np.uint8)
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return (depth * 255.0).astype(np.uint8)


def frame_number(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def save_depths(clip_dir: Path, output_root: Path, model: DepthAnything3) -> None:
    frame_dir = clip_dir / "frames"
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return

    prediction = model.inference([str(p) for p in frames])
    depths = prediction.depth
    out_dir = output_root / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, depth in zip(frames, depths):
        number = frame_number(frame_path)
        np.save(out_dir / f"depth_{number}.npy", depth.astype(np.float32))
        depth_u8 = normalize_u8(depth)
        cv2.imwrite(str(out_dir / f"depth_{number}.png"), depth_u8)
        cv2.imwrite(str(out_dir / frame_path.name), depth_u8)
    print(f"Saved {len(frames)} DA3 depth maps to {out_dir}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.eval()

    clip_dirs = [p for p in sorted(args.frames_root.iterdir()) if (p / "frames").exists()]
    for clip_dir in clip_dirs[: args.max_clips]:
        save_depths(clip_dir, args.output_root, model)


if __name__ == "__main__":
    main()
