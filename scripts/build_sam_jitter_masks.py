#!/usr/bin/env python3
"""Build SAM raw masks from jittered GT boxes and diagnose mask errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--jitter-fracs", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--box-expand-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda")
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


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty GT mask")
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def jitter_box(box: np.ndarray, shape: tuple[int, int], jitter_frac: float, expand_frac: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = box.astype(np.float32)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    x0 -= bw * expand_frac
    x1 += bw * expand_frac
    y0 -= bh * expand_frac
    y1 += bh * expand_frac

    jitter = np.array(
        [
            rng.uniform(-jitter_frac, jitter_frac) * bw,
            rng.uniform(-jitter_frac, jitter_frac) * bh,
            rng.uniform(-jitter_frac, jitter_frac) * bw,
            rng.uniform(-jitter_frac, jitter_frac) * bh,
        ],
        dtype=np.float32,
    )
    x0, y0, x1, y1 = np.array([x0, y0, x1, y1], dtype=np.float32) + jitter
    x0 = float(np.clip(x0, 0, w - 2))
    y0 = float(np.clip(y0, 0, h - 2))
    x1 = float(np.clip(x1, x0 + 1, w - 1))
    y1 = float(np.clip(y1, y0 + 1, h - 1))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u8 = mask.astype(np.uint8)
    return (cv2.dilate(u8, kernel) != cv2.erode(u8, kernel))


def chamfer_boundary_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = boundary(pred)
    gt_b = boundary(gt)
    if not np.any(pred_b) or not np.any(gt_b):
        return float("nan")
    pred_dist = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 5)
    gt_dist = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (pred_dist[gt_b].mean() + gt_dist[pred_b].mean()))


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    gt_area = max(int(gt.sum()), 1)
    pred_area = max(int(pred.sum()), 1)
    return {
        "mask_iou": float(inter / max(union, 1)),
        "missing": float((gt & ~pred).sum() / gt_area),
        "over_mask": float((pred & ~gt).sum() / gt_area),
        "pred_gt_area_ratio": float(pred_area / gt_area),
        "boundary_error": chamfer_boundary_error(pred, gt),
    }


def temporal_jitter(curr: np.ndarray, prev: np.ndarray | None) -> float:
    if prev is None:
        return 0.0
    inter = int((curr & prev).sum())
    union = int((curr | prev).sum())
    return float(1.0 - inter / max(union, 1))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "setting",
        "clip",
        "frame",
        "box_jitter_frac",
        "sam_score",
        "mask_iou",
        "missing",
        "over_mask",
        "pred_gt_area_ratio",
        "boundary_error",
        "temporal_jitter",
        "box_x0",
        "box_y0",
        "box_x1",
        "box_y1",
        "frames",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["mask_iou", "missing", "over_mask", "pred_gt_area_ratio", "boundary_error", "temporal_jitter"]
    summary = []
    for setting in sorted({r["setting"] for r in rows}):
        subset = [r for r in rows if r["setting"] == setting]
        item = {"setting": setting, "frames": len(subset)}
        for key in keys:
            item[key] = float(np.nanmean(np.array([r[key] for r in subset], dtype=np.float32)))
        summary.append(item)
    order = {"perfect": 0, "sam_jitter_5": 1, "sam_jitter_10": 2}
    return sorted(summary, key=lambda r: order.get(r["setting"], 99))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    clips = args.clips or sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir())
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for clip in clips:
        clip_dir = args.probe_root / clip
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        prev_by_setting: dict[str, np.ndarray | None] = {"perfect": None}
        for frac in args.jitter_fracs:
            prev_by_setting[f"sam_jitter_{int(round(frac * 100))}"] = None

        for frame_path in frame_paths:
            name = frame_path.name
            frame_bgr = read_color(frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gt = read_mask(clip_dir / "masks" / name)

            perfect_metrics = mask_metrics(gt, gt)
            perfect_metrics["temporal_jitter"] = temporal_jitter(gt, prev_by_setting["perfect"])
            rows.append({"setting": "perfect", "clip": clip, "frame": frame_path.stem, "box_jitter_frac": 0.0, **perfect_metrics})
            prev_by_setting["perfect"] = gt

            predictor.set_image(frame_rgb)
            gt_box = bbox_from_mask(gt)
            for frac in args.jitter_fracs:
                setting = f"sam_jitter_{int(round(frac * 100))}"
                box = jitter_box(gt_box, gt.shape, frac, args.box_expand_frac, rng)
                masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                pred = masks[int(np.argmax(scores))].astype(bool)
                write_mask(args.output_root / setting / clip / "masks" / name, pred)
                metrics = mask_metrics(pred, gt)
                metrics["temporal_jitter"] = temporal_jitter(pred, prev_by_setting[setting])
                rows.append(
                    {
                        "setting": setting,
                        "clip": clip,
                        "frame": frame_path.stem,
                        "box_jitter_frac": frac,
                        "box_x0": float(box[0]),
                        "box_y0": float(box[1]),
                        "box_x1": float(box[2]),
                        "box_y1": float(box[3]),
                        "sam_score": float(np.max(scores)),
                        **metrics,
                    }
                )
                prev_by_setting[setting] = pred

    args.diagnostic_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.diagnostic_root / "frame_mask_metrics.csv", rows)
    summary = summarize(rows)
    write_csv(args.diagnostic_root / "mask_error_summary.csv", summary)
    (args.diagnostic_root / "mask_error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


