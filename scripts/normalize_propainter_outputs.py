#!/usr/bin/env python3
"""Rename ProPainter sequential frame outputs to the source frame names."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--propainter-frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_frames = sorted(args.source_frames.glob("*.png"))
    propainter_frames = sorted(args.propainter_frames.glob("*.png"))
    if len(source_frames) != len(propainter_frames):
        raise RuntimeError(
            f"Frame count mismatch: {len(source_frames)} source vs "
            f"{len(propainter_frames)} ProPainter"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for src, pred in zip(source_frames, propainter_frames):
        shutil.copy2(pred, args.output_dir / src.name)
    print(f"Copied {len(source_frames)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
