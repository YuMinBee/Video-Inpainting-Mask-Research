#!/usr/bin/env python3
"""Blend ProPainter outputs with the input frame using confidence gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIZ_CLIPS = [
    "I-210618_I01001_W01",
    "I-210720_O12052_T04",
    "I-210729_I03011_W04",
]

VARIANTS = {
    "temporal_union_occupancy": {
        "source": Path("results/4090_rerun/aihub_subset_100/temporal_union/propainter_outputs"),
        "confidence": "occupancy",
    },
    "temporal_union_distance": {
        "source": Path("results/4090_rerun/aihub_subset_100/temporal_union/propainter_outputs"),
        "confidence": "distance",
    },
    "temporal_union_occ_dist": {
        "source": Path("results/4090_rerun/aihub_subset_100/temporal_union/propainter_outputs"),
        "confidence": "occ_dist",
    },
    "ours_r10_t0p10_occ_dist": {
        "source": Path("results/4090_rerun/aihub_subset_100/ours_r10_t0p10/propainter_outputs"),
        "confidence": "occ_dist",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--confidence-root", type=Path, default=Path("experiments/aihub_subset_100/confidence_maps"))
    parser.add_argument("--output-root", type=Path, default=Path("results/4090_rerun/aihub_subset_100/v2_confidence_blend"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--viz-root", type=Path, default=Path("experiments/paper_visualizations/confidence_blending"))
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


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap(values: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(values * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def resize_tile(image: np.ndarray, width: int = 300) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, int(width * h / w)))


def make_viz(frame: np.ndarray, source: np.ndarray, confidence: np.ndarray, blended: np.ndarray, variant: str, out_path: Path) -> None:
    diff = np.mean(np.abs(blended.astype(np.float32) - frame.astype(np.float32)), axis=2)
    diff_vis = cv2.applyColorMap(np.clip(diff * 6.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    tiles = [
        label(resize_tile(frame), "Input"),
        label(resize_tile(source), "Source output"),
        label(resize_tile(heatmap(confidence)), "Confidence gate"),
        label(resize_tile(blended), "Blended output"),
        label(resize_tile(diff_vis), "Blend-input diff"),
    ]
    panel = np.hstack(tiles)
    write_image(out_path, panel)


def blend_frame(frame: np.ndarray, source: np.ndarray, confidence: np.ndarray, mask: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    gate = np.power(np.clip(confidence, 0.0, 1.0), gamma).astype(np.float32)
    gate[mask] = 1.0
    gate3 = gate[..., None]
    blended = source.astype(np.float32) * gate3 + frame.astype(np.float32) * (1.0 - gate3)
    return np.clip(blended, 0, 255).astype(np.uint8), gate


def main() -> None:
    args = parse_args()
    rows = []
    clip_dirs = [p for p in sorted(args.probe_root.iterdir()) if (p / "frames").is_dir() and (p / "masks").is_dir()]
    for clip_dir in clip_dirs:
        clip = clip_dir.name
        frame_paths = sorted((clip_dir / "frames").glob("*.png"))
        for idx, frame_path in enumerate(frame_paths):
            name = frame_path.name
            frame = read_color(frame_path)
            mask = read_gray(clip_dir / "masks" / name) > 0
            for variant in args.variants:
                spec = VARIANTS[variant]
                source_path = spec["source"] / clip / name
                confidence_path = args.confidence_root / spec["confidence"] / clip / "maps" / name
                source = read_color(source_path)
                confidence = read_gray(confidence_path).astype(np.float32) / 255.0
                blended, gate = blend_frame(frame, source, confidence, mask, args.gamma)

                out_path = args.output_root / variant / "propainter_outputs" / clip / name
                write_image(out_path, blended)
                rows.append(
                    {
                        "variant": variant,
                        "clip": clip,
                        "frame": frame_path.stem,
                        "gate_mean": float(gate.mean()),
                        "gate_extra_mean": float(gate[~mask].mean()) if np.any(~mask) else 0.0,
                        "blend_input_diff_mean": float(np.mean(np.abs(blended.astype(np.float32) - frame.astype(np.float32))) / 255.0),
                    }
                )

                if clip in args.viz_clips and idx == min(args.viz_frame_index, len(frame_paths) - 1):
                    make_viz(
                        frame,
                        source,
                        gate,
                        blended,
                        variant,
                        args.viz_root / f"{variant}_{clip}_{frame_path.stem}_blend_panel.png",
                    )

    summary = []
    for variant in args.variants:
        subset = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "frames": len(subset),
                "gate_mean": float(np.mean([row["gate_mean"] for row in subset])),
                "gate_extra_mean": float(np.mean([row["gate_extra_mean"] for row in subset])),
                "blend_input_diff_mean": float(np.mean([row["blend_input_diff_mean"] for row in subset])),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "blend_frame_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "blend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
