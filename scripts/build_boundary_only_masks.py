#!/usr/bin/env python3
"""Build boundary-only mask variants from original masks and temporal unions."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"(?:frame_|_F)(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--union-root", type=Path, required=True)
    parser.add_argument("--original-mask-root", type=Path, default=None)
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help="Root containing <clip>/masks. Prefer this when normalized original masks already exist.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--clips", nargs="+", default=None)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    path.write_bytes(encoded.tobytes())


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        data = np.frombuffer(src.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {src}")
        write_image(dst, image)


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise RuntimeError(f"Cannot parse frame number from {path}")
    return int(match.group(1))


def numbered_pngs(path: Path) -> list[Path]:
    return sorted([p for p in path.glob("*.png") if FRAME_RE.search(p.name)], key=frame_number)


def build_original_mask_index(root: Path) -> dict[str, Path]:
    index = {}
    for clip_dir in root.rglob("*"):
        if clip_dir.is_dir() and any(clip_dir.glob("*.png")):
            index[clip_dir.name] = clip_dir
    return index


def boundary_support(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0


def by_frame_number(paths: list[Path]) -> dict[int, Path]:
    return {frame_number(p): p for p in paths}


def pngs_by_name(path: Path) -> dict[str, Path]:
    return {p.name: p for p in path.glob("*.png")}


def process_clip(clip_dir: Path, original_mask_dir: Path, output_root: Path, radius: int) -> dict:
    clip = clip_dir.name
    frame_by_name = pngs_by_name(clip_dir / "frames")
    union_by_name = pngs_by_name(clip_dir / "masks")
    original_by_name = pngs_by_name(original_mask_dir)
    names = sorted(set(frame_by_name) & set(union_by_name) & set(original_by_name))
    if not names:
        raise RuntimeError(f"No matching frames for {clip}")

    sizes = {"original": [], "union": [], "boundary_only": []}
    for name in names:
        frame_path = frame_by_name[name]
        union_path = union_by_name[name]
        original_path = original_by_name[name]
        original_raw = read_gray(original_path)
        union = read_gray(union_path) > 0
        if original_raw.shape != union.shape:
            original_raw = cv2.resize(
                original_raw,
                (union.shape[1], union.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        original = original_raw > 0
        band = boundary_support(original, radius)
        boundary_only = original | ((union & ~original) & band)

        sizes["original"].append(int(original.sum()))
        sizes["union"].append(int(union.sum()))
        sizes["boundary_only"].append(int(boundary_only.sum()))

        link_or_copy(frame_path, output_root / clip / "frames" / name)
        write_image(output_root / clip / "masks" / name, boundary_only.astype(np.uint8) * 255)

    return {
        "clip": clip,
        "frames": len(names),
        "radius": radius,
        "original_px_mean": float(np.mean(sizes["original"])),
        "union_px_mean": float(np.mean(sizes["union"])),
        "boundary_only_px_mean": float(np.mean(sizes["boundary_only"])),
        "boundary_only_extra_ratio": float(
            (np.mean(sizes["boundary_only"]) - np.mean(sizes["original"])) / max(np.mean(sizes["original"]), 1.0)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.probe_root is None and args.original_mask_root is None:
        raise RuntimeError("Provide either --probe-root or --original-mask-root")
    original_index = build_original_mask_index(args.original_mask_root) if args.original_mask_root else {}
    clip_dirs = [p for p in sorted(args.union_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    if args.clips:
        wanted = set(args.clips)
        clip_dirs = [p for p in clip_dirs if p.name in wanted]

    stats = []
    for clip_dir in clip_dirs:
        if args.probe_root is not None:
            original_mask_dir = args.probe_root / clip_dir.name / "masks"
        else:
            original_mask_dir = original_index.get(clip_dir.name)
        if original_mask_dir is None or not original_mask_dir.is_dir():
            root = args.probe_root if args.probe_root is not None else args.original_mask_root
            raise RuntimeError(f"Missing original masks for {clip_dir.name} under {root}")
        stats.append(process_clip(clip_dir, original_mask_dir, args.output_root, args.radius))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mask_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote boundary-only masks for {len(stats)} clips to {args.output_root}")


if __name__ == "__main__":
    main()
