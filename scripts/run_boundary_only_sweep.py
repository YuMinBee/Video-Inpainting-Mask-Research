#!/usr/bin/env python3
"""Run boundary-only temporal smoothing at multiple strengths."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--config-root", type=Path, default=Path("configs/boundary_only_sweep"))
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=40)
    return parser.parse_args()


def strength_tag(value: float) -> str:
    return f"lambda_{value:.1f}".replace(".", "p")


def make_config(strength: float, args: argparse.Namespace) -> Path:
    tag = strength_tag(strength)
    cfg = {
        "name": f"aihub_subset_20_boundary_only_{tag}",
        "frames_root": "experiments/aihub_subset_20_real",
        "baseline_root": "results/aihub_subset_20/propainter_outputs",
        "depth_root": "results/aihub_subset_20/da3_depths",
        "output_root": f"results/aihub_subset_20/refined_outputs/boundary_only_sweep/{tag}",
        "metrics_path": f"experiments/02_boundary_refinement/metrics/boundary_only_sweep/{tag}_metrics.csv",
        "summary_path": f"experiments/02_boundary_refinement/metrics/boundary_only_sweep/{tag}_summary.json",
        "visualization_root": f"experiments/02_boundary_refinement/visualizations/boundary_only_sweep/{tag}",
        "max_clips": args.max_clips,
        "max_frames": args.max_frames,
        "boundary_blur": 7,
        "max_alpha": strength,
        "tau_rgb": 1000000000.0,
        "tau_depth": 1000000000.0,
        "depth_edge_suppression": 0.0,
    }
    args.config_root.mkdir(parents=True, exist_ok=True)
    path = args.config_root / f"{tag}.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def expected_frame_count(output_root: Path) -> int:
    return len(list(output_root.glob("*/*.png"))) if output_root.exists() else 0


def main() -> None:
    args = parse_args()
    for strength in args.strengths:
        cfg_path = make_config(strength, args)
        cfg = json.loads(cfg_path.read_text())
        output_root = Path(cfg["output_root"])
        if expected_frame_count(output_root) >= 796:
            print(f"Skipping lambda={strength:.1f}: outputs already exist")
            continue
        print(f"Running boundary-only lambda={strength:.1f}")
        subprocess.run(
            ["python", "scripts/depth_aware_boundary_refine.py", "--config", str(cfg_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
