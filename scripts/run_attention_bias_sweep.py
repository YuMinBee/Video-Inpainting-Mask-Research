#!/usr/bin/env python3
"""Run ProPainter attention-bias lambda sweeps for selected mask variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "temporal_union": Path("experiments/aihub_subset_100/masks/temporal_union"),
    "ours_r10_t0p10": Path("experiments/aihub_subset_100/masks/ours_r10_t0p10"),
}


def lambda_tag(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, default=Path("experiments/aihub_subset_100_probe"))
    parser.add_argument("--confidence-root", type=Path, default=Path("experiments/aihub_subset_100/confidence_maps/occ_dist"))
    parser.add_argument("--result-root", type=Path, default=Path("results/4090_rerun/aihub_subset_100/attention_bias"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--query-update-lambda", type=float, default=0.0)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--conda-env", default="ai")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--subvideo-length", type=int, default=40)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument("--ascii-stage-root", type=Path, default=Path("C:/Temp/propainter_attention_bias_stage"))
    return parser.parse_args()


def run_one(args: argparse.Namespace, variant: str, lam: float) -> None:
    tag = lambda_tag(lam)
    if args.query_update_lambda != 0:
        tag = f"{tag}_q{lambda_tag(args.query_update_lambda).replace('lambda_', '')}"
    raw_root = args.result_root / variant / tag / "raw"
    output_root = args.result_root / variant / tag / "propainter_outputs"
    cmd = [
        sys.executable,
        "scripts/run_propainter_batch.py",
        "--probe-root",
        str(VARIANTS[variant]),
        "--frames-root",
        str(args.frames_root),
        "--raw-output-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--confidence-root",
        str(args.confidence_root),
        "--key-reliability-lambda",
        str(lam),
        "--query-update-lambda",
        str(args.query_update_lambda),
        "--conda-env",
        args.conda_env,
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
    ]
    if args.clips:
        cmd.extend(["--clips", *args.clips])
    if args.fp16:
        cmd.append("--fp16")
    if args.use_current_python:
        cmd.append("--use-current-python")
    if args.ascii_stage_root is not None:
        cmd.extend(["--ascii-stage-root", str(args.ascii_stage_root / variant / tag)])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    for variant in args.variants:
        for lam in args.lambdas:
            run_one(args, variant, lam)


if __name__ == "__main__":
    main()

