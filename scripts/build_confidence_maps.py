#!/usr/bin/env python3
"""Build temporal-occupancy confidence maps and hard-mask sanity summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIZ_CLIPS = [
    "I-210618_I01001_W01",
    "I-210720_O12052_T04",
    "I-210729_I03011_W04",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/aihub_subset_100/confidence_maps"))
    parser.add_argument("--hard-reference-root", type=Path, default=Path("experiments/aihub_subset_100/masks/ours_r10_t0p10"))
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--viz-root", type=Path, default=Path("experiments/paper_visualizations/confidence_maps"))
    parser.add_argument("--viz-clips", nargs="+", default=DEFAULT_VIZ_CLIPS)
    parser.add_argument("--viz-frame-index", type=int, default=10)
    return parser.parse_args()


def read_color(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


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
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    outside = (~mask).astype(np.uint8)
    return cv2.distanceTransform(outside, cv2.DIST_L2, 5)


def confidence_maps(mask: np.ndarray, union: np.ndarray, occupancy: np.ndarray, sigma: float, alpha: float, beta: float) -> dict[str, np.ndarray]:
    candidate = union & (~mask)
    distance = boundary_distance(mask)
    distance_weight = np.exp(-distance / max(sigma, 1e-6)).astype(np.float32)
    occ = np.power(np.clip(occupancy, 0.0, 1.0), alpha).astype(np.float32)
    dist = np.power(np.clip(distance_weight, 0.0, 1.0), beta).astype(np.float32)

    maps = {
        "occupancy": np.zeros(mask.shape, dtype=np.float32),
        "distance": np.zeros(mask.shape, dtype=np.float32),
        "occ_dist": np.zeros(mask.shape, dtype=np.float32),
    }
    maps["occupancy"][candidate] = occ[candidate]
    maps["distance"][candidate] = dist[candidate]
    maps["occ_dist"][candidate] = (occ * dist)[candidate]
    for value in maps.values():
        value[mask] = 1.0
    return maps


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap(values: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(values * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = frame.copy()
    idx = mask > 0
    out[idx] = ((1 - alpha) * out[idx] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def make_viz(frame: np.ndarray, mask: np.ndarray, union: np.ndarray, maps: dict[str, np.ndarray], hard: np.ndarray, out_path: Path) -> None:
    h, w = frame.shape[:2]
    tile_w = 300
    tile_h = int(tile_w * h / w)
    tiles = [
        label(cv2.resize(frame, (tile_w, tile_h)), "Input"),
        label(cv2.resize(overlay_mask(frame, mask, (0, 255, 255)), (tile_w, tile_h)), "Original mask"),
        label(cv2.resize(overlay_mask(frame, union, (255, 0, 0)), (tile_w, tile_h)), "Temporal union"),
        label(cv2.resize(heatmap(maps["occupancy"]), (tile_w, tile_h)), "Occupancy only"),
        label(cv2.resize(heatmap(maps["distance"]), (tile_w, tile_h)), "Distance only"),
        label(cv2.resize(heatmap(maps["occ_dist"]), (tile_w, tile_h)), "Occupancy x distance"),
        label(cv2.resize(overlay_mask(frame, hard, (0, 180, 255)), (tile_w, tile_h)), "C > 0.10 hard mask"),
    ]
    panel = np.hstack(tiles)
    write_image(out_path, panel)


def main() -> None:
    args = parse_args()
    rows = []
    stats = []
    clip_dirs = [p for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    for clip_dir in clip_dirs:
        clip = clip_dir.name
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        masks = [read_gray(clip_dir / "masks" / p.name) > 0 for p in frame_paths]
        stack = np.stack(masks, axis=0)
        union = np.any(stack, axis=0)
        occupancy = np.mean(stack, axis=0).astype(np.float32)

        clip_stats = {"clip": clip, "frames": len(frame_paths)}
        for idx, (frame_path, mask) in enumerate(zip(frame_paths, masks)):
            maps = confidence_maps(mask, union, occupancy, args.sigma, args.alpha, args.beta)
            name = frame_path.name
            for mode, values in maps.items():
                write_image(args.output_root / mode / clip / "maps" / name, np.clip(values * 255.0, 0, 255).astype(np.uint8))

            ref_mask_path = args.hard_reference_root / clip / "masks" / name
            ref = read_gray(ref_mask_path) > 0 if ref_mask_path.exists() else None
            for threshold in args.thresholds:
                hard = mask | (maps["occ_dist"] > threshold)
                rows.append(
                    {
                        "clip": clip,
                        "frame": frame_path.stem,
                        "threshold": threshold,
                        "hard_area": int(hard.sum()),
                        "orig_area": int(mask.sum()),
                        "union_area": int(union.sum()),
                        "hard_over_orig": float(hard.sum() / max(int(mask.sum()), 1)),
                        "iou_with_ours_r10_t0p10": mask_iou(hard, ref) if ref is not None else math.nan,
                    }
                )
            if clip in args.viz_clips and idx == min(args.viz_frame_index, len(frame_paths) - 1):
                frame = read_color(frame_path)
                hard = mask | (maps["occ_dist"] > 0.10)
                make_viz(frame, mask, union, maps, hard, args.viz_root / f"{clip}_{frame_path.stem}_confidence_panel.png")

        for mode_dir in ["occupancy", "distance", "occ_dist"]:
            map_paths = sorted((args.output_root / mode_dir / clip / "maps").glob("*.png"))
            values = [read_gray(p).mean() / 255.0 for p in map_paths]
            clip_stats[f"{mode_dir}_mean"] = float(np.mean(values)) if values else math.nan
        stats.append(clip_stats)

    args.output_root.mkdir(parents=True, exist_ok=True)
    sanity_csv = args.output_root / "hard_mask_sanity.csv"
    with sanity_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "confidence_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    summary = []
    for threshold in args.thresholds:
        subset = [r for r in rows if abs(r["threshold"] - threshold) < 1e-9]
        summary.append(
            {
                "threshold": threshold,
                "frames": len(subset),
                "hard_over_orig_mean": float(np.mean([r["hard_over_orig"] for r in subset])),
                "iou_with_ours_r10_t0p10_mean": float(np.nanmean([r["iou_with_ours_r10_t0p10"] for r in subset])),
            }
        )
    (args.output_root / "hard_mask_sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()