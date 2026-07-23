#!/usr/bin/env python3
r"""Build Choice-A style mask grid.

This variant first creates a boundary-dilated base mask and only then selects
additional temporal-union pixels outside that base:

    M_base = D_rb(M_t)
    S_t = (U \ M_base) & D_rs(M_t) & {A > tau}
    M_ours = M_base | S_t

The selection radius is required to be larger than the base radius.
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
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--base-radii", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--select-radii", nargs="+", type=int, default=[20, 30, 50])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15])
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


def variant_name(rb: int, rs: int, tau: float) -> str:
    tau_tag = f"{tau:.2f}".replace(".", "p")
    return f"a_rb{rb}_rs{rs}_t{tau_tag}"


def process_clip(clip_dir: Path, output_root: Path, configs: list[tuple[int, int, float]]) -> list[dict]:
    clip = clip_dir.name
    frame_paths = sorted((clip_dir / "frames").glob("*.png"))
    masks = [read_gray(clip_dir / "masks" / p.name) for p in frame_paths]
    stack = np.stack(masks, axis=0)
    union = np.any(stack, axis=0)
    occupancy = np.mean(stack, axis=0).astype(np.float32)

    rows: list[dict] = []
    for frame_path, mask in zip(frame_paths, masks):
        name = frame_path.name
        original_area = max(int(mask.sum()), 1)
        base_by_radius = {rb: dilate(mask, rb) for rb, _, _ in configs}
        select_by_radius = {rs: dilate(mask, rs) for _, rs, _ in configs}
        for rb, rs, tau in configs:
            base = base_by_radius[rb]
            selected = (union & ~base) & select_by_radius[rs] & (occupancy > tau)
            final = base | selected
            variant = variant_name(rb, rs, tau)
            write_mask(output_root / variant / clip / "masks" / name, final)
            rows.append(
                {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "variant": variant,
                    "rb": rb,
                    "rs": rs,
                    "tau": tau,
                    "original_area": int(mask.sum()),
                    "base_area": int(base.sum()),
                    "selected_area": int(selected.sum()),
                    "variant_area": int(final.sum()),
                    "base_extra_area": int((base & ~mask).sum()),
                    "selected_extra_area": int(selected.sum()),
                    "extra_area": int((final & ~mask).sum()),
                    "extra_mask_ratio": float((final & ~mask).sum() / original_area),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configs = [
        (rb, rs, tau)
        for rb in args.base_radii
        for rs in args.select_radii
        for tau in args.thresholds
        if rs > rb
    ]
    if not configs:
        raise RuntimeError("No valid configs: selection radius must be larger than base radius.")

    clips = args.clips
    if clips is None:
        clips = [p.name for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]

    all_rows: list[dict] = []
    for clip in clips:
        all_rows.extend(process_clip(args.probe_root / clip, args.output_root, configs))

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "mask_stats.csv", all_rows)
    (args.output_root / "mask_stats.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    summary = []
    for variant in sorted({row["variant"] for row in all_rows}):
        rows = [row for row in all_rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "rb": rows[0]["rb"],
                "rs": rows[0]["rs"],
                "tau": rows[0]["tau"],
                "frames": len(rows),
                "extra_mask_ratio_mean": float(np.mean([row["extra_mask_ratio"] for row in rows])),
                "base_extra_area_mean": float(np.mean([row["base_extra_area"] for row in rows])),
                "selected_extra_area_mean": float(np.mean([row["selected_extra_area"] for row in rows])),
                "extra_area_mean": float(np.mean([row["extra_area"] for row in rows])),
            }
        )
    write_csv(args.output_root / "mask_summary.csv", summary)
    (args.output_root / "mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
