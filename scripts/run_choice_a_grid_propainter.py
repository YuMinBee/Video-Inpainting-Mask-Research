#!/usr/bin/env python3
"""Run ProPainter for Choice-A grid mask variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=Path, default=Path("experiments/aihub_subset_100/masks_choice_a_grid"))
    parser.add_argument("--frames-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--result-root", type=Path, default=Path("results/4090_rerun/aihub_subset_100/choice_a_grid"))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--max-clips", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--subvideo-length", type=int, default=40)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument("--conda-env", default="ai")
    parser.add_argument("--ascii-stage-root", type=Path, default=Path("C:/tmp/propainter_choice_a_grid_stage"))
    return parser.parse_args()


def discover_variants(mask_root: Path, requested: list[str] | None) -> list[str]:
    available = sorted(p.name for p in mask_root.iterdir() if (p.is_dir() and any(p.glob("*/masks"))))
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise RuntimeError(f"Missing variants under {mask_root}: {', '.join(missing)}")
    return requested


def run_variant(args: argparse.Namespace, variant: str) -> None:
    raw_root = args.result_root / variant / "propainter_raw"
    output_root = args.result_root / variant / "propainter_outputs"
    cmd = [
        sys.executable,
        "scripts/run_propainter_batch.py",
        "--probe-root",
        str(args.mask_root / variant),
        "--frames-root",
        str(args.frames_root),
        "--raw-output-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--gpu",
        args.gpu,
        "--max-clips",
        str(args.max_clips),
        "--subvideo-length",
        str(args.subvideo_length),
        "--neighbor-length",
        str(args.neighbor_length),
        "--ref-stride",
        str(args.ref_stride),
        "--raft-iter",
        str(args.raft_iter),
        "--mask-dilation",
        str(args.mask_dilation),
        "--conda-env",
        args.conda_env,
    ]
    if args.clips:
        cmd.extend(["--clips", *args.clips])
    if args.fp16:
        cmd.append("--fp16")
    if args.use_current_python:
        cmd.append("--use-current-python")
    if args.ascii_stage_root is not None:
        cmd.extend(["--ascii-stage-root", str(args.ascii_stage_root / variant)])

    print(f"\n=== Choice-A grid variant: {variant} ===", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    variants = discover_variants(args.mask_root, args.variants)
    print(f"Running {len(variants)} variants: {', '.join(variants)}", flush=True)
    for variant in variants:
        run_variant(args, variant)


if __name__ == "__main__":
    main()

