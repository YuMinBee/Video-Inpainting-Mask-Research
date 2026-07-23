#!/usr/bin/env python3
"""Run ProPainter over probe clips and normalize frame filenames."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path("experiments/aihub_subset_20_probe"))
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Optional root containing <clip>/frames when probe-root only contains variant masks.",
    )
    parser.add_argument("--raw-output-root", type=Path, default=Path("results/aihub_subset_20/propainter_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("results/aihub_subset_20/propainter_outputs"))
    parser.add_argument("--conda-env", default="ai")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--subvideo-length", type=int, default=40)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--confidence-root", type=Path, default=None, help="Optional root containing <clip>/maps confidence PNGs.")
    parser.add_argument("--key-reliability-lambda", type=float, default=0.0)
    parser.add_argument("--query-update-lambda", type=float, default=0.0)
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument(
        "--ascii-stage-root",
        type=Path,
        default=None,
        help="Stage each clip under an ASCII-only path before ProPainter inference.",
    )
    return parser.parse_args()


def frame_dir_for(clip: Path, frames_root: Path | None) -> Path:
    if frames_root is None:
        return clip / "frames"
    return frames_root / clip.name / "frames"


def confidence_dir_for(clip: Path, confidence_root: Path | None) -> Path | None:
    if confidence_root is None:
        return None
    direct = confidence_root / clip.name
    maps = direct / "maps"
    return maps if maps.is_dir() else direct


def clip_dirs(probe_root: Path, frames_root: Path | None) -> list[Path]:
    clips = []
    for p in sorted(probe_root.iterdir()):
        if frame_dir_for(p, frames_root).is_dir() and (p / "masks").is_dir():
            clips.append(p)
    return clips


def normalize_frames(source_frames: Path, propainter_frames: Path, output_dir: Path) -> int:
    src = sorted(source_frames.glob("*.png"))
    pred = sorted(propainter_frames.glob("*.png"))
    if len(src) != len(pred):
        raise RuntimeError(
            f"Frame count mismatch for {source_frames.parent.name}: "
            f"{len(src)} source vs {len(pred)} ProPainter"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for src_path, pred_path in zip(src, pred):
        shutil.copy2(pred_path, output_dir / src_path.name)
    return len(src)


def copy_fresh_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    propainter_root = Path("external/ProPainter").resolve()
    if not (propainter_root / "inference_propainter.py").exists():
        raise RuntimeError(f"Missing ProPainter checkout: {propainter_root}")

    clips = clip_dirs(args.probe_root, args.frames_root)
    if args.clips:
        wanted = set(args.clips)
        clips = [p for p in clips if p.name in wanted]
    clips = clips[: args.max_clips]
    if not clips:
        raise RuntimeError(f"No probe clips found under {args.probe_root}")

    args.raw_output_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

    for clip in clips:
        source_frames = frame_dir_for(clip, args.frames_root)
        frame_count = len(list(source_frames.glob("*.png")))
        if frame_count == 0:
            raise RuntimeError(f"No source frames found for {clip.name}: {source_frames}")

        normalized_dir = args.output_root / clip.name
        if len(list(normalized_dir.glob("*.png"))) == frame_count:
            print(f"Skipping {clip.name}: normalized frames already exist", flush=True)
            continue

        confidence_dir = confidence_dir_for(clip, args.confidence_root)
        if (args.key_reliability_lambda != 0 or args.query_update_lambda != 0) and confidence_dir is None:
            raise RuntimeError("--confidence-root is required when --key-reliability-lambda is nonzero")
        if confidence_dir is not None and not confidence_dir.is_dir():
            raise RuntimeError(f"Missing confidence maps for {clip.name}: {confidence_dir}")

        stage_clip = None
        video_dir = source_frames.resolve()
        mask_dir = (clip / "masks").resolve()
        confidence_arg = confidence_dir.resolve() if confidence_dir is not None else None
        infer_output_dir = (args.raw_output_root / clip.name).resolve()
        if args.ascii_stage_root is not None:
            stage_root = args.ascii_stage_root.resolve()
            stage_clip = stage_root / clip.name
            video_dir = stage_clip / "frames"
            mask_dir = stage_clip / "masks"
            infer_output_dir = stage_clip / "propainter_raw"
            copy_fresh_tree(source_frames, video_dir)
            copy_fresh_tree(clip / "masks", mask_dir)
            if confidence_dir is not None:
                confidence_arg = stage_clip / "confidence"
                copy_fresh_tree(confidence_dir, confidence_arg)

        python_cmd = [sys.executable] if args.use_current_python else ["conda", "run", "-n", args.conda_env, "python"]
        cmd = [
            *python_cmd,
            "inference_propainter.py",
            "--video",
            str(video_dir),
            "--mask",
            str(mask_dir),
            "--output",
            str(infer_output_dir),
            "--save_frames",
            "--subvideo_length",
            str(args.subvideo_length),
            "--neighbor_length",
            str(args.neighbor_length),
            "--ref_stride",
            str(args.ref_stride),
            "--raft_iter",
            str(args.raft_iter),
            "--mask_dilation",
            str(args.mask_dilation),
        ]
        if args.fp16:
            cmd.append("--fp16")
        if confidence_arg is not None:
            cmd.extend(["--confidence", str(confidence_arg)])
        if args.key_reliability_lambda != 0:
            cmd.extend(["--key_reliability_lambda", str(args.key_reliability_lambda)])
        if args.query_update_lambda != 0:
            cmd.extend(["--query_update_lambda", str(args.query_update_lambda)])

        print(f"Running ProPainter: {clip.name}", flush=True)
        subprocess.run(cmd, cwd=propainter_root, env=env, check=True)

        raw_clip_dir = args.raw_output_root / clip.name
        if stage_clip is not None:
            raw_clip_dir.parent.mkdir(parents=True, exist_ok=True)
            copy_fresh_tree(infer_output_dir, raw_clip_dir)

        propainter_frames = raw_clip_dir / "frames" / "frames"
        if not propainter_frames.exists():
            propainter_frames = raw_clip_dir / clip.name / "frames"
        count = normalize_frames(source_frames, propainter_frames, normalized_dir)
        print(f"Normalized {count} frames to {normalized_dir}", flush=True)


if __name__ == "__main__":
    main()

