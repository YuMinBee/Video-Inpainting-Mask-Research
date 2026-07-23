#!/usr/bin/env python3
"""Evaluate synthetic-GT inpainting outputs against clean background frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument(
        "--gt-mask-root",
        type=Path,
        default=None,
        help="Optional root containing <clip>/masks GT alpha masks. Defaults to probe-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=OUTPUT_ROOT=MASK_ROOT",
        required=True,
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> list[dict[str, str | Path]]:
    methods = []
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise RuntimeError(f"Invalid --method value: {value}")
        name, output, mask = parts
        methods.append({"name": name, "output": Path(output), "mask": Path(mask)})
    return methods


def read_color(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return image > 0


def resolve_mask(mask_root: Path, clip: str, name: str) -> Path:
    candidates = [
        mask_root / clip / "masks" / name,
        mask_root / clip / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mask for {clip}/{name} under {mask_root}")


def boundary_band(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel) > 0
    eroded = cv2.erode(u8, kernel) > 0
    return dilated & ~eroded


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if np.any(mask) else float("nan")


def masked_psnr(output: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    diff = output.astype(np.float32) - gt.astype(np.float32)
    if not np.any(mask):
        return float("nan")
    mse = float(np.mean(diff[mask] ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_map_gray(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return numerator / np.maximum(denominator, 1e-12)


def backward_flow(curr_bgr: np.ndarray, prev_bgr: np.ndarray) -> np.ndarray:
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 25, 3, 5, 1.2, 0)


def warp(prev: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], method_names: list[str], group_key: str | None = None) -> list[dict]:
    keys = [
        "masked_psnr",
        "masked_ssim",
        "masked_mae",
        "outside_mae",
        "boundary_te",
        "extra_mask_ratio",
    ]
    summary = []
    groups = sorted({r[group_key] for r in rows}) if group_key else [None]
    for group in groups:
        for method in method_names:
            subset = [r for r in rows if r["method"] == method and (group_key is None or r[group_key] == group)]
            if not subset:
                continue
            item = {"method": method, "frames": len(subset)}
            if group_key is not None:
                item = {group_key: group, **item}
            for key in keys:
                item[key] = float(np.nanmean(np.array([r[key] for r in subset], dtype=np.float32)))
            summary.append(item)
    return summary


def write_markdown(path: Path, summary: list[dict], clips: list[str]) -> None:
    lines = [
        "# Synthetic-GT Evaluation",
        "",
        f"Clips: `{', '.join(clips)}`",
        "",
        "| Method | mPSNR ??| mSSIM ??| mMAE ??| Outside MAE ??| BTE ??| Extra ??|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['masked_psnr']:.3f} | {row['masked_ssim']:.4f} | "
            f"{row['masked_mae']:.5f} | {row['outside_mae']:.5f} | "
            f"{row['boundary_te']:.6f} | {row['extra_mask_ratio']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    method_names = [str(m["name"]) for m in methods]
    clips = args.clips or sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir())
    gt_mask_root = args.gt_mask_root or args.probe_root

    rows: list[dict] = []
    for clip in clips:
        clip_dir = args.probe_root / clip
        gt_clip_dir = gt_mask_root / clip
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        prev_gt = None
        prev_outputs: dict[str, np.ndarray] = {}
        for idx, frame_path in enumerate(frame_paths):
            name = frame_path.name
            gt = read_color(clip_dir / "gt_background" / name)
            gt_mask = read_mask(resolve_mask(gt_mask_root, clip, name))
            boundary_path = gt_clip_dir / "mask_boundaries" / name
            boundary = read_mask(boundary_path) if boundary_path.exists() else boundary_band(gt_mask)
            flow = backward_flow(gt, prev_gt) if prev_gt is not None else None
            gt_area = max(int(gt_mask.sum()), 1)

            for spec in methods:
                method = str(spec["name"])
                output = read_color(Path(spec["output"]) / clip / name)
                variant_mask = read_mask(resolve_mask(Path(spec["mask"]), clip, name))
                if variant_mask.shape != gt_mask.shape:
                    variant_mask = cv2.resize(
                        variant_mask.astype(np.uint8),
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                diff_abs = np.mean(np.abs(output.astype(np.float32) - gt.astype(np.float32)), axis=2) / 255.0
                ssim_map = ssim_map_gray(output, gt)
                if flow is None:
                    te = np.zeros(gt_mask.shape, dtype=np.float32)
                else:
                    te = temporal_error(output, warp(prev_outputs[method], flow))

                rows.append(
                    {
                        "clip": clip,
                        "frame": frame_path.stem,
                        "frame_index": idx,
                        "method": method,
                        "masked_psnr": masked_psnr(output, gt, gt_mask),
                        "masked_ssim": safe_mean(ssim_map, gt_mask),
                        "masked_mae": safe_mean(diff_abs, gt_mask),
                        "outside_mae": safe_mean(diff_abs, ~gt_mask),
                        "boundary_te": safe_mean(te, boundary),
                        "extra_mask_ratio": float((variant_mask & ~gt_mask).sum() / gt_area),
                    }
                )
                prev_outputs[method] = output
            prev_gt = gt

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, method_names)
    clip_summary = summarize(rows, method_names, group_key="clip")
    write_csv(args.output_root / "frame_metrics.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "clip_summary.csv", clip_summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "RESULTS.md", summary, clips)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


