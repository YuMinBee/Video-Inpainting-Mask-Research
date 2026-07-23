#!/usr/bin/env python3
"""Build a small depth/temporal-flicker visualization probe.

This script is intentionally training-free. It prepares a small subset from the
sample AI-Hub-like layout, creates an OpenCV inpainting baseline, consumes DA3
depth maps when they are available, and otherwise creates a proxy depth map only
for pipeline smoke testing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"_F(\d+)")


@dataclass(frozen=True)
class ClipPaths:
    clip_id: str
    frame_dir: Path
    mask_json_dir: Path
    baseline_dir: Path | None
    da3_depth_dir: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/New_Sample"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/depth_temporal_probe"))
    parser.add_argument("--source-subdir", default="원천데이터/jpg")
    parser.add_argument("--mask-json-subdir", default="라벨링데이터/masking")
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--baseline-source-root", type=Path, default=None)
    parser.add_argument("--depth-source-root", type=Path, default=None)
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--edge-percentile", type=float, default=90.0)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    if not match:
        return -1
    return int(match.group(1))


def discover_clips(
    dataset_root: Path,
    baseline_source_root: Path | None,
    depth_source_root: Path | None,
    max_clips: int,
) -> list[ClipPaths]:
    frame_base = dataset_root / "원천데이터" / "jpg"
    mask_base = dataset_root / "라벨링데이터" / "masking"
    clips: list[ClipPaths] = []
    for frame_dir in sorted(frame_base.glob("*/*")):
        if not frame_dir.is_dir():
            continue
        rel = frame_dir.relative_to(frame_base)
        mask_json_dir = mask_base / rel
        if not mask_json_dir.exists():
            continue
        baseline_dir = None
        if baseline_source_root is not None:
            candidate = baseline_source_root / rel
            if not candidate.exists():
                candidate = baseline_source_root / frame_dir.name
            if candidate.exists():
                baseline_dir = candidate
        da3_depth_dir = None
        if depth_source_root is not None:
            candidate = depth_source_root / rel
            if not candidate.exists():
                candidate = depth_source_root / frame_dir.name
            if candidate.exists():
                da3_depth_dir = candidate
        clips.append(ClipPaths(frame_dir.name, frame_dir, mask_json_dir, baseline_dir, da3_depth_dir))
    return clips[:max_clips]


def discover_clips_from_subdirs(
    dataset_root: Path,
    source_subdir: str,
    mask_json_subdir: str,
    baseline_source_root: Path | None,
    depth_source_root: Path | None,
    max_clips: int,
) -> list[ClipPaths]:
    frame_base = dataset_root / source_subdir
    mask_base = dataset_root / mask_json_subdir
    clips: list[ClipPaths] = []
    for frame_dir in sorted(frame_base.glob("*/*")):
        if not frame_dir.is_dir():
            continue
        rel = frame_dir.relative_to(frame_base)
        mask_json_dir = mask_base / rel
        if not mask_json_dir.exists():
            continue
        baseline_dir = None
        if baseline_source_root is not None:
            candidate = baseline_source_root / rel
            if not candidate.exists():
                candidate = baseline_source_root / frame_dir.name
            if candidate.exists():
                baseline_dir = candidate
        da3_depth_dir = None
        if depth_source_root is not None:
            candidate = depth_source_root / rel
            if not candidate.exists():
                candidate = depth_source_root / frame_dir.name
            if candidate.exists():
                da3_depth_dir = candidate
        clips.append(ClipPaths(frame_dir.name, frame_dir, mask_json_dir, baseline_dir, da3_depth_dir))
    return clips[:max_clips]


def ensure_size(image: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or width <= 0 or image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_exact(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == (height, width):
        return image
    interpolation = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def read_frame(path: Path, width: int | None) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return ensure_size(image, width)


def rasterize_mask(json_path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not json_path.exists():
        return mask

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("Learning_Data_Info.", {})
    annotations = info.get("Annotation", [])
    for ann in annotations:
        points = ann.get("segmentation", [])
        if len(points) < 6:
            continue
        pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width / 1920.0
        pts[:, 1] *= height / 1080.0
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def mask_boundary(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    return cv2.subtract(dilated, eroded)


def basename_for_frame(frame_path: Path) -> str:
    number = frame_number(frame_path)
    return f"frame_{number:04d}" if number >= 0 else frame_path.stem


def depth_candidates(depth_dir: Path, frame_path: Path) -> list[Path]:
    number = frame_number(frame_path)
    names = [
        frame_path.name,
        frame_path.with_suffix(".png").name,
        frame_path.with_suffix(".jpg").name,
        f"depth_{number:04d}.png",
        f"depth_{number:04d}.jpg",
        f"depth_{number:04d}.npy",
        f"{frame_path.stem}.png",
        f"{frame_path.stem}.npy",
    ]
    return [depth_dir / name for name in names]


def image_candidates(image_dir: Path, frame_path: Path, prefix: str) -> list[Path]:
    number = frame_number(frame_path)
    names = [
        frame_path.name,
        frame_path.with_suffix(".png").name,
        f"{prefix}_{number:04d}.png",
        f"{prefix}_{number:04d}.jpg",
        f"frame_{number:04d}.png",
        f"frame_{number:04d}.jpg",
        f"{frame_path.stem}.png",
        f"{frame_path.stem}.jpg",
    ]
    return [image_dir / name for name in names]


def normalize_u8(values: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if hi <= lo + 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    out = (values - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def proxy_depth_from_frame(frame: np.ndarray) -> np.ndarray:
    """Cheap visual proxy. Replace with DA3 output for real experiments."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 50, 50)
    y = np.linspace(0, 255, gray.shape[0], dtype=np.float32)[:, None]
    proxy = 0.65 * smooth.astype(np.float32) + 0.35 * np.repeat(y, gray.shape[1], axis=1)
    return normalize_u8(proxy)


def read_depth(frame: np.ndarray, frame_path: Path, depth_dir: Path | None) -> tuple[np.ndarray, str]:
    if depth_dir is not None:
        for candidate in depth_candidates(depth_dir, frame_path):
            if not candidate.exists():
                continue
            if candidate.suffix.lower() == ".npy":
                return normalize_u8(np.load(candidate)), "da3"
            depth = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                if depth.ndim == 3:
                    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
                depth = resize_exact(depth, frame.shape[:2])
                return normalize_u8(depth), "da3"
    return proxy_depth_from_frame(frame), "proxy"


def depth_edge(depth_u8: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    depth_f = depth_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_vis = normalize_u8(mag, 1.0, 99.0)
    threshold = float(np.percentile(mag, percentile))
    edge_bin = (mag >= threshold).astype(np.uint8) * 255
    return edge_vis, edge_bin


def inpaint_baseline(frame: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return frame.copy()
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def read_or_create_baseline(
    frame: np.ndarray,
    frame_path: Path,
    mask: np.ndarray,
    baseline_dir: Path | None,
    radius: int,
) -> tuple[np.ndarray, str]:
    if baseline_dir is not None:
        for candidate in image_candidates(baseline_dir, frame_path, "baseline"):
            if not candidate.exists():
                continue
            image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if image is not None:
                return resize_exact(image, frame.shape[:2]), "external"
    return inpaint_baseline(frame, mask, radius), "opencv_telea"


def temporal_error(prev: np.ndarray | None, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prev is None:
        zeros = np.zeros(curr.shape[:2], dtype=np.uint8)
        return zeros, curr.copy()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    h, w = curr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    err = np.mean(np.abs(curr.astype(np.float32) - warped.astype(np.float32)), axis=2)
    return normalize_u8(err, 5.0, 99.0), warped


def heatmap(gray: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.applyColorMap(gray, colormap)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base.copy()
    idx = mask > 0
    out[idx] = ((1.0 - alpha) * out[idx] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def put_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_visual_panel(
    frame: np.ndarray,
    baseline: np.ndarray,
    boundary: np.ndarray,
    depth: np.ndarray,
    edge_bin: np.ndarray,
    temporal: np.ndarray,
    depth_source: str,
) -> np.ndarray:
    original_overlay = overlay_mask(frame, boundary, (0, 255, 255), 0.9)
    depth_overlay = overlay_mask(frame, edge_bin, (0, 255, 0), 0.85)
    temporal_overlay = cv2.addWeighted(frame, 0.55, heatmap(temporal), 0.45, 0)
    combined = overlay_mask(temporal_overlay, boundary, (0, 255, 255), 0.9)
    combined = overlay_mask(combined, edge_bin, (0, 255, 0), 0.6)

    depth_color = heatmap(depth, cv2.COLORMAP_INFERNO)
    top = np.hstack(
        [
            put_label(original_overlay, "Original + mask boundary"),
            put_label(depth_color, f"Depth map ({depth_source})"),
        ]
    )
    middle = np.hstack(
        [
            put_label(depth_overlay, "Original + depth edge"),
            put_label(temporal_overlay, "Temporal error heatmap"),
        ]
    )
    bottom = np.hstack(
        [
            put_label(baseline, "Baseline inpaint"),
            put_label(combined, "Boundary + depth edge + flicker"),
        ]
    )
    return np.vstack([top, middle, bottom])


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    encoded.tofile(str(path))


def process_clip(clip: ClipPaths, out_root: Path, max_frames: int, resize_width: int, radius: int, edge_pct: float) -> dict:
    frame_paths = sorted(clip.frame_dir.glob("*.jpg"), key=frame_number)[:max_frames]
    clip_out = out_root / clip.clip_id

    prev_baseline = None
    depth_sources: set[str] = set()
    frame_count = 0
    for frame_path in frame_paths:
        frame_count += 1
        name = basename_for_frame(frame_path)
        frame = read_frame(frame_path, resize_width)
        json_path = clip.mask_json_dir / f"{frame_path.stem}_M.json"
        mask = rasterize_mask(json_path, frame.shape[:2])
        boundary = mask_boundary(mask)
        baseline, baseline_source = read_or_create_baseline(frame, frame_path, mask, clip.baseline_dir, radius)
        depth, depth_source = read_depth(frame, frame_path, clip.da3_depth_dir)
        depth_sources.add(depth_source)
        edge_vis, edge_bin = depth_edge(depth, edge_pct)
        temporal, warped = temporal_error(prev_baseline, baseline)
        prev_baseline = baseline

        write_image(clip_out / "frames" / f"{name}.png", frame)
        write_image(clip_out / "masks" / f"{name}.png", mask)
        write_image(clip_out / "mask_boundaries" / f"{name}.png", boundary)
        write_image(clip_out / "baseline_inpaint" / f"{name}.png", baseline)
        write_image(clip_out / "depth_maps" / f"{name}.png", depth)
        write_image(clip_out / "depth_edges" / f"{name}.png", edge_vis)
        write_image(clip_out / "depth_edge_binary" / f"{name}.png", edge_bin)
        write_image(clip_out / "warped_previous" / f"{name}.png", warped)
        write_image(clip_out / "temporal_error" / f"{name}.png", temporal)
        panel = make_visual_panel(frame, baseline, boundary, depth, edge_bin, temporal, depth_source)
        write_image(clip_out / "visual_panels" / f"{name}.png", panel)

    return {
        "clip_id": clip.clip_id,
        "frames": frame_count,
        "baseline_source": "external" if clip.baseline_dir else "opencv_telea",
        "depth_sources": sorted(depth_sources),
        "output": str(clip_out),
    }


def clip_output_complete(clip: ClipPaths, out_root: Path, max_frames: int) -> bool:
    expected = len(sorted(clip.frame_dir.glob("*.jpg"), key=frame_number)[:max_frames])
    if expected == 0:
        return False
    clip_out = out_root / clip.clip_id
    required_dirs = ["frames", "masks", "mask_boundaries", "depth_edge_binary", "temporal_error"]
    return all(len(list((clip_out / name).glob("*.png"))) == expected for name in required_dirs)


def main() -> None:
    args = parse_args()
    clips = discover_clips_from_subdirs(
        args.dataset_root,
        args.source_subdir,
        args.mask_json_subdir,
        args.baseline_source_root,
        args.depth_source_root,
        args.max_clips,
    )
    if not clips:
        raise RuntimeError(f"No clips found under {args.dataset_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "max_frames": args.max_frames,
        "resize_width": args.resize_width,
        "baseline_source_root": str(args.baseline_source_root) if args.baseline_source_root else None,
        "depth_source_root": str(args.depth_source_root) if args.depth_source_root else None,
        "clips": [],
        "note": "Depth source 'proxy' is only for pipeline visualization. Use DA3 maps for research results.",
    }
    for clip in clips:
        if clip_output_complete(clip, args.output_root, args.max_frames):
            print(f"Skipping {clip.clip_id}: probe outputs already exist", flush=True)
            summary["clips"].append(
                {
                    "clip_id": clip.clip_id,
                    "frames": len(sorted(clip.frame_dir.glob("*.jpg"), key=frame_number)[: args.max_frames]),
                    "baseline_source": "skipped",
                    "depth_sources": ["skipped"],
                    "output": str(args.output_root / clip.clip_id),
                }
            )
            continue
        summary["clips"].append(
            process_clip(
                clip,
                args.output_root,
                args.max_frames,
                args.resize_width,
                args.inpaint_radius,
                args.edge_percentile,
            )
        )

    summary_path = args.output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
