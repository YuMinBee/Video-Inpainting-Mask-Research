#!/usr/bin/env python3
"""Depth-aware boundary temporal refinement.

This is the first executable version of the paper idea:

  - reduce boundary flicker with temporal blending
  - use pseudo-depth consistency as a trust signal
  - suppress blending on strong depth discontinuities to avoid over-smoothing

The current flow backend is OpenCV Farneback so the script can run while the
dataset is still downloading. The interface is intentionally simple so RAFT flow
can replace `backward_flow` later without changing the refinement logic.
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
    parser.add_argument("--config", type=Path, default=Path("configs/boundary_refinement_probe.json"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def resize_exact(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == (height, width):
        return image
    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def normalize_depth_clip(depths: list[np.ndarray]) -> list[np.ndarray]:
    values = np.concatenate([d.reshape(-1) for d in depths]).astype(np.float32)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo + 1e-6:
        return [np.zeros_like(d, dtype=np.float32) for d in depths]
    return [np.clip((d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) for d in depths]


def frame_number(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_")[-1] if "_" in stem else stem


def read_depth(depth_dir: Path, frame_name: str, shape: tuple[int, int]) -> np.ndarray:
    number = frame_number(frame_name)
    candidates = [
        depth_dir / f"depth_{number}.npy",
        depth_dir / f"depth_{number}.png",
        depth_dir / frame_name,
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return resize_exact(np.load(path).astype(np.float32), shape)
        depth = read_gray(path).astype(np.float32)
        return resize_exact(depth, shape)
    raise RuntimeError(f"Missing depth for {frame_name} in {depth_dir}")


def backward_flow(curr_bgr: np.ndarray, prev_bgr: np.ndarray) -> np.ndarray:
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        curr,
        prev,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def warp_with_backward_flow(prev: np.ndarray, flow_curr_to_prev: np.ndarray) -> np.ndarray:
    height, width = flow_curr_to_prev.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow_curr_to_prev[..., 0]
    map_y = grid_y + flow_curr_to_prev[..., 1]
    return cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def boundary_weight(boundary_u8: np.ndarray, blur: int) -> np.ndarray:
    weight = (boundary_u8.astype(np.float32) / 255.0)
    if blur > 1:
        blur = blur if blur % 2 == 1 else blur + 1
        weight = cv2.GaussianBlur(weight, (blur, blur), 0)
    max_value = float(weight.max())
    return weight / max_value if max_value > 1e-6 else weight


def depth_edge_strength(depth: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, 98))
    if hi <= 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip(mag / hi, 0.0, 1.0)


def temporal_error(curr: np.ndarray, warped_prev: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(curr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2) / 255.0


def metric_row(
    clip: str,
    frame: str,
    boundary: np.ndarray,
    depth_edge: np.ndarray,
    base_error: np.ndarray,
    refined_error: np.ndarray,
    alpha: np.ndarray,
) -> dict:
    b = boundary > 0
    if not np.any(b):
        b = np.ones(boundary.shape, dtype=bool)
    return {
        "clip": clip,
        "frame": frame,
        "baseline_bte": float(base_error[b].mean()),
        "refined_bte": float(refined_error[b].mean()),
        "bte_delta": float(base_error[b].mean() - refined_error[b].mean()),
        "baseline_dwbf": float((base_error[b] * depth_edge[b]).mean()),
        "refined_dwbf": float((refined_error[b] * depth_edge[b]).mean()),
        "dwbf_delta": float((base_error[b] * depth_edge[b]).mean() - (refined_error[b] * depth_edge[b]).mean()),
        "mean_alpha_boundary": float(alpha[b].mean()),
    }


def save_debug_panel(
    path: Path,
    frame: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    alpha: np.ndarray,
    base_error: np.ndarray,
    refined_error: np.ndarray,
) -> None:
    alpha_vis = cv2.applyColorMap(np.clip(alpha * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    base_vis = cv2.applyColorMap(np.clip(base_error * 255 * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    ref_vis = cv2.applyColorMap(np.clip(refined_error * 255 * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    top = np.hstack([frame, baseline, refined])
    bottom = np.hstack([alpha_vis, base_vis, ref_vis])
    panel = np.vstack([top, bottom])
    write_image(path, panel)


def process_clip(clip: str, cfg: dict) -> list[dict]:
    frames_dir = Path(cfg["frames_root"]) / clip / "frames"
    boundary_dir = Path(cfg["frames_root"]) / clip / "mask_boundaries"
    baseline_dir = Path(cfg["baseline_root"]) / clip
    depth_dir = Path(cfg["depth_root"]) / clip
    out_dir = Path(cfg["output_root"]) / clip
    vis_dir = Path(cfg["visualization_root"]) / clip

    frame_paths = sorted(frames_dir.glob("*.png"))[: int(cfg["max_frames"])]
    frames = [read_color(p) for p in frame_paths]
    baselines = [resize_exact(read_color(baseline_dir / p.name), frames[i].shape[:2]) for i, p in enumerate(frame_paths)]
    boundaries = [resize_exact(read_gray(boundary_dir / p.name), frames[i].shape[:2]) for i, p in enumerate(frame_paths)]
    depths_raw = [read_depth(depth_dir, p.name, frames[i].shape[:2]) for i, p in enumerate(frame_paths)]
    depths = normalize_depth_clip(depths_raw)

    refined: list[np.ndarray] = []
    rows: list[dict] = []
    for idx, frame_path in enumerate(frame_paths):
        baseline = baselines[idx]
        if idx == 0:
            refined_frame = baseline.copy()
            refined.append(refined_frame)
            write_image(out_dir / frame_path.name, refined_frame)
            continue

        flow = backward_flow(frames[idx], frames[idx - 1])
        warped_refined = warp_with_backward_flow(refined[idx - 1], flow)
        warped_baseline = warp_with_backward_flow(baselines[idx - 1], flow)
        warped_depth = warp_with_backward_flow(depths[idx - 1], flow)

        rgb_diff = temporal_error(baseline, warped_refined)
        depth_diff = np.abs(depths[idx] - warped_depth)
        rgb_conf = np.exp(-rgb_diff / float(cfg["tau_rgb"]))
        depth_conf = np.exp(-depth_diff / float(cfg["tau_depth"]))
        edge = depth_edge_strength(depths[idx])
        edge_preserve = 1.0 - float(cfg["depth_edge_suppression"]) * edge
        b_weight = boundary_weight(boundaries[idx], int(cfg["boundary_blur"]))
        alpha = float(cfg["max_alpha"]) * b_weight * rgb_conf * depth_conf * edge_preserve
        alpha = np.clip(alpha, 0.0, float(cfg["max_alpha"]))

        refined_f = (
            alpha[..., None] * warped_refined.astype(np.float32)
            + (1.0 - alpha[..., None]) * baseline.astype(np.float32)
        )
        refined_frame = np.clip(refined_f, 0, 255).astype(np.uint8)
        refined.append(refined_frame)
        write_image(out_dir / frame_path.name, refined_frame)

        warped_refined_prev_for_metric = warp_with_backward_flow(refined[idx - 1], flow)
        base_error = temporal_error(baseline, warped_baseline)
        refined_error = temporal_error(refined_frame, warped_refined_prev_for_metric)
        rows.append(metric_row(clip, frame_path.stem, boundaries[idx], edge, base_error, refined_error, alpha))
        save_debug_panel(vis_dir / frame_path.name, frames[idx], baseline, refined_frame, alpha, base_error, refined_error)

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"config": cfg, "overall": {}, "by_clip": {}}
    for key in ["baseline_bte", "refined_bte", "bte_delta", "baseline_dwbf", "refined_dwbf", "dwbf_delta", "mean_alpha_boundary"]:
        values = np.array([r[key] for r in rows], dtype=np.float32)
        summary["overall"][key] = float(values.mean()) if values.size else 0.0
    for clip in sorted({r["clip"] for r in rows}):
        clip_rows = [r for r in rows if r["clip"] == clip]
        summary["by_clip"][clip] = {}
        for key in ["baseline_bte", "refined_bte", "bte_delta", "baseline_dwbf", "refined_dwbf", "dwbf_delta", "mean_alpha_boundary"]:
            values = np.array([r[key] for r in clip_rows], dtype=np.float32)
            summary["by_clip"][clip][key] = float(values.mean()) if values.size else 0.0
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    cfg = load_config(parse_args().config)
    baseline_root = Path(cfg["baseline_root"])
    clips = sorted([p.name for p in baseline_root.iterdir() if p.is_dir()])[: int(cfg["max_clips"])]
    rows: list[dict] = []
    for clip in clips:
        rows.extend(process_clip(clip, cfg))
    write_csv(Path(cfg["metrics_path"]), rows)
    write_summary(Path(cfg["summary_path"]), rows, cfg)
    print(f"Processed {len(clips)} clips, {len(rows)} metric rows")
    print(f"Outputs: {cfg['output_root']}")
    print(f"Metrics: {cfg['metrics_path']}")
    print(f"Summary: {cfg['summary_path']}")


if __name__ == "__main__":
    main()
