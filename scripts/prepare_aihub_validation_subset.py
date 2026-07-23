#!/usr/bin/env python3
"""Extract a small AI-Hub validation subset without unpacking all category zips."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


CATEGORY_RE = re.compile(r"^[VS|VL]+(\d+)")
FRAME_RE = re.compile(r"_F(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aihub-root",
        type=Path,
        default=Path("dataset/040.Inpainting_자동화를_위한_영상_데이터/01.데이터/2.Validation"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/aihub_subset_20/dataset"),
    )
    parser.add_argument("--source-subdir", default="원천데이터/jpg")
    parser.add_argument("--mask-json-subdir", default="라벨링데이터/masking")
    parser.add_argument("--mask-png-subdir", default="라벨링데이터/png")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def category_id(path: Path) -> int:
    match = CATEGORY_RE.match(path.stem)
    if not match:
        return 9999
    return int(match.group(1))


def frame_number(name: str) -> int:
    match = FRAME_RE.search(Path(name).stem)
    if not match:
        return -1
    return int(match.group(1))


def is_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def first_component(name: str) -> str | None:
    parts = Path(name).parts
    if len(parts) < 2:
        return None
    return parts[0]


def category_zip_map(root: Path, subdir: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in sorted((root / subdir).glob("*.zip"), key=category_id):
        output[category_id(path)] = path
    return output


def list_clip_entries(zip_path: Path) -> dict[str, list[str]]:
    clips: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            clip = first_component(name)
            if clip is None or not is_image(name):
                continue
            clips[clip].append(name)
    return {clip: sorted(names, key=frame_number) for clip, names in clips.items()}


def select_clips(source_zips: dict[int, Path], max_clips: int) -> list[tuple[int, Path, str, list[str]]]:
    by_category: list[tuple[int, Path, list[tuple[str, list[str]]]]] = []
    for cat_id, zip_path in sorted(source_zips.items()):
        clips = sorted(list_clip_entries(zip_path).items())
        if clips:
            by_category.append((cat_id, zip_path, clips))

    selected: list[tuple[int, Path, str, list[str]]] = []
    round_idx = 0
    while len(selected) < max_clips:
        added = False
        for cat_id, zip_path, clips in by_category:
            if round_idx >= len(clips):
                continue
            clip_id, entries = clips[round_idx]
            selected.append((cat_id, zip_path, clip_id, entries))
            added = True
            if len(selected) >= max_clips:
                break
        if not added:
            break
        round_idx += 1
    return selected


def write_member(zf: zipfile.ZipFile, member: str, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(zf.read(member))
    return True


def extract_matching_members(
    zip_path: Path | None,
    clip_id: str,
    frame_numbers: set[int],
    dst_dir: Path,
    overwrite: bool,
    suffix_filter: tuple[str, ...],
) -> int:
    if zip_path is None or not zip_path.exists():
        return 0

    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.startswith(f"{clip_id}/")
            and name.lower().endswith(suffix_filter)
            and frame_number(name) in frame_numbers
        ]
        for name in sorted(names, key=frame_number):
            dst = dst_dir / Path(name).name
            if write_member(zf, name, dst, overwrite):
                written += 1
    return written


def main() -> None:
    args = parse_args()
    source_zips = category_zip_map(args.aihub_root, args.source_subdir)
    mask_json_zips = category_zip_map(args.aihub_root, args.mask_json_subdir)
    mask_png_zips = category_zip_map(args.aihub_root, args.mask_png_subdir)

    selected = select_clips(source_zips, args.max_clips)
    if not selected:
        raise RuntimeError(f"No clips found under {args.aihub_root}")

    rows = []
    for cat_id, source_zip, clip_id, frame_entries in selected:
        category_name = source_zip.stem
        chosen_frames = frame_entries[: args.max_frames]
        chosen_numbers = {frame_number(name) for name in chosen_frames}

        frame_out = args.output_root / args.source_subdir / category_name / clip_id
        json_out = args.output_root / args.mask_json_subdir / category_name / clip_id
        png_out = args.output_root / args.mask_png_subdir / category_name / clip_id

        frame_count = 0
        with zipfile.ZipFile(source_zip) as zf:
            for name in chosen_frames:
                dst = frame_out / Path(name).name
                if write_member(zf, name, dst, args.overwrite):
                    frame_count += 1

        json_count = extract_matching_members(
            mask_json_zips.get(cat_id),
            clip_id,
            chosen_numbers,
            json_out,
            args.overwrite,
            (".json",),
        )
        png_count = extract_matching_members(
            mask_png_zips.get(cat_id),
            clip_id,
            chosen_numbers,
            png_out,
            args.overwrite,
            (".png",),
        )

        rows.append(
            {
                "clip_id": clip_id,
                "category_id": cat_id,
                "category": category_name,
                "source_zip": str(source_zip),
                "frames": len(chosen_frames),
                "frames_written": frame_count,
                "mask_json_written": json_count,
                "mask_png_written": png_count,
                "frame_dir": str(frame_out),
                "mask_json_dir": str(json_out),
                "mask_png_dir": str(png_out),
            }
        )
        print(f"{clip_id}: frames={len(chosen_frames)} json={json_count} png={png_count}")

    manifest = {
        "aihub_root": str(args.aihub_root),
        "output_root": str(args.output_root),
        "max_clips": args.max_clips,
        "max_frames": args.max_frames,
        "clips": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
