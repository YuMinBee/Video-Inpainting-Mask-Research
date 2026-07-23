#!/usr/bin/env python3
"""Build a small synthetic-GT object-removal probe.

The clean target is the original frame before inserting a synthetic object.
Objects are harvested from existing masked AI-Hub frames and pasted onto other
clips with a simple smooth trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLIPS = [
    "I-210618_I01001_W01",
    "I-210618_I01006_W04",
    "I-210627_O01004_W04",
    "I-210627_O04018_W05",
    "I-210714_O01002_T04",
    "I-210714_O01002_W02",
    "I-210714_O01003_T03",
    "I-210715_I03012_W06",
    "I-210715_I06019_T02",
    "I-210715_I09026_W02",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--frames-per-clip", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-area-frac", type=float, default=0.05)
    parser.add_argument("--max-area-frac", type=float, default=0.15)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return image > 0


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel) > 0
    eroded = cv2.erode(u8, kernel) > 0
    return dilated & ~eroded


def feather_alpha(mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    alpha = mask.astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(alpha, 0.0, 1.0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def collect_donors(source_root: Path, clips: list[str]) -> list[dict]:
    donors: list[dict] = []
    for clip in clips:
        clip_dir = source_root / clip
        for frame_path in sorted((clip_dir / "frames").glob("*.png")):
            mask_path = clip_dir / "masks" / frame_path.name
            if not mask_path.exists():
                continue
            mask = read_mask(mask_path)
            area = int(mask.sum())
            if area < 500:
                continue
            x0, y0, x1, y1 = mask_bbox(mask)
            w = x1 - x0
            h = y1 - y0
            if w < 20 or h < 20:
                continue
            donors.append({"clip": clip, "frame": frame_path.name, "area": area, "bbox": [x0, y0, x1, y1]})
    if not donors:
        raise RuntimeError(f"No donor objects found under {source_root}")
    return donors


def make_cutout(source_root: Path, donor: dict) -> tuple[np.ndarray, np.ndarray]:
    frame = read_color(source_root / donor["clip"] / "frames" / donor["frame"])
    mask = read_mask(source_root / donor["clip"] / "masks" / donor["frame"])
    x0, y0, x1, y1 = donor["bbox"]
    pad = 8
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(frame.shape[1], x1 + pad)
    y1 = min(frame.shape[0], y1 + pad)
    cutout = frame[y0:y1, x0:x1].copy()
    alpha = feather_alpha(mask[y0:y1, x0:x1])
    return cutout, alpha


def resize_cutout(cutout: np.ndarray, alpha: np.ndarray, target_area: float) -> tuple[np.ndarray, np.ndarray]:
    source_area = float(max(np.count_nonzero(alpha > 0.2), 1))
    scale = math.sqrt(target_area / source_area)
    scale = float(np.clip(scale, 0.25, 3.5))
    new_w = max(8, int(round(cutout.shape[1] * scale)))
    new_h = max(8, int(round(cutout.shape[0] * scale)))
    resized_cutout = cv2.resize(cutout, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_cutout, np.clip(resized_alpha, 0.0, 1.0)


def paste_object(background: np.ndarray, cutout: np.ndarray, alpha: np.ndarray, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = background.shape[:2]
    oh, ow = cutout.shape[:2]
    x0 = int(round(cx - ow / 2))
    y0 = int(round(cy - oh / 2))
    x0 = max(0, min(w - ow, x0))
    y0 = max(0, min(h - oh, y0))
    x1 = x0 + ow
    y1 = y0 + oh

    output = background.copy()
    roi = output[y0:y1, x0:x1].astype(np.float32)
    a = alpha[..., None].astype(np.float32)
    blended = roi * (1.0 - a) + cutout.astype(np.float32) * a
    output[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = alpha > 0.1
    return output, mask


def choose_trajectory(rng: np.random.Generator, width: int, height: int, obj_w: int, obj_h: int) -> tuple[np.ndarray, np.ndarray]:
    margin_x = max(obj_w // 2 + 8, int(width * 0.08))
    margin_y = max(obj_h // 2 + 8, int(height * 0.08))
    x_start = rng.uniform(margin_x, width - margin_x)
    y_start = rng.uniform(margin_y, height - margin_y)
    dx = rng.uniform(width * 0.12, width * 0.28) * rng.choice([-1, 1])
    dy = rng.uniform(height * 0.04, height * 0.16) * rng.choice([-1, 1])
    x_end = np.clip(x_start + dx, margin_x, width - margin_x)
    y_end = np.clip(y_start + dy, margin_y, height - margin_y)
    return np.array([x_start, y_start], dtype=np.float32), np.array([x_end, y_end], dtype=np.float32)


def process_clip(
    source_root: Path,
    output_root: Path,
    clip: str,
    out_name: str,
    donor: dict,
    rng: np.random.Generator,
    frames_per_clip: int,
    min_area_frac: float,
    max_area_frac: float,
    boundary_radius: int,
) -> dict:
    source_clip = source_root / clip
    frame_paths = sorted((source_clip / "frames").glob("*.png"))[:frames_per_clip]
    if len(frame_paths) < frames_per_clip:
        raise RuntimeError(f"Not enough frames for {clip}: {len(frame_paths)}")

    first = read_color(frame_paths[0])
    h, w = first.shape[:2]
    target_frac = float(rng.uniform(min_area_frac, max_area_frac))
    cutout, alpha = make_cutout(source_root, donor)
    cutout, alpha = resize_cutout(cutout, alpha, target_frac * h * w)
    start, end = choose_trajectory(rng, w, h, cutout.shape[1], cutout.shape[0])

    clip_root = output_root / out_name
    if clip_root.exists():
        shutil.rmtree(clip_root)
    (clip_root / "frames").mkdir(parents=True, exist_ok=True)
    (clip_root / "masks").mkdir(parents=True, exist_ok=True)
    (clip_root / "gt_background").mkdir(parents=True, exist_ok=True)
    (clip_root / "mask_boundaries").mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, frame_path in enumerate(frame_paths):
        t = idx / max(frames_per_clip - 1, 1)
        smooth_t = 0.5 - 0.5 * math.cos(math.pi * t)
        center = start * (1.0 - smooth_t) + end * smooth_t
        background = read_color(frame_path)
        synthetic, mask = paste_object(background, cutout, alpha, float(center[0]), float(center[1]))
        name = frame_path.name
        write_image(clip_root / "frames" / name, synthetic)
        write_image(clip_root / "masks" / name, mask.astype(np.uint8) * 255)
        write_image(clip_root / "gt_background" / name, background)
        write_image(clip_root / "mask_boundaries" / name, boundary_band(mask, boundary_radius).astype(np.uint8) * 255)
        rows.append(
            {
                "frame": name,
                "mask_area": int(mask.sum()),
                "mask_area_frac": float(mask.sum() / (h * w)),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
            }
        )

    meta = {
        "synthetic_clip": out_name,
        "background_clip": clip,
        "donor": donor,
        "frames": frames_per_clip,
        "frame_size": [w, h],
        "target_area_frac": target_frac,
        "mean_mask_area_frac": float(np.mean([r["mask_area_frac"] for r in rows])),
        "trajectory_start": start.tolist(),
        "trajectory_end": end.tolist(),
        "boundary_radius": boundary_radius,
    }
    (clip_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    donors = collect_donors(args.source_probe_root, args.clips)
    metas = []
    for index, clip in enumerate(args.clips, start=1):
        candidates = [d for d in donors if d["clip"] != clip]
        donor = candidates[int(rng.integers(0, len(candidates)))]
        out_name = f"synthetic_{index:03d}_{clip}"
        metas.append(
            process_clip(
                args.source_probe_root,
                args.output_root,
                clip,
                out_name,
                donor,
                rng,
                args.frames_per_clip,
                args.min_area_frac,
                args.max_area_frac,
                args.boundary_radius,
            )
        )

    rows = [
        {
            "synthetic_clip": m["synthetic_clip"],
            "background_clip": m["background_clip"],
            "donor_clip": m["donor"]["clip"],
            "donor_frame": m["donor"]["frame"],
            "frames": m["frames"],
            "target_area_frac": m["target_area_frac"],
            "mean_mask_area_frac": m["mean_mask_area_frac"],
        }
        for m in metas
    ]
    write_csv(args.output_root / "synthetic_summary.csv", rows)
    (args.output_root / "synthetic_summary.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

