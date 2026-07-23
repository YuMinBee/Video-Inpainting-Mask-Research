import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


BINS = [
    ("0-5 px", 0.0, 5.0),
    ("5-10 px", 5.0, 10.0),
    ("10-20 px", 10.0, 20.0),
    ("20+ px", 20.0, float("inf")),
]


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def mask_files(mask_dir: Path) -> list[Path]:
    files = sorted(mask_dir.glob("*.png"))
    if not files:
        files = sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg"))
    return files


def distance_to_raw_mask(raw: np.ndarray) -> np.ndarray:
    # Missing pixels lie outside raw. Distance transform on the inverse raw mask
    # gives their Euclidean distance to the nearest raw foreground boundary.
    inv_raw = (~raw).astype(np.uint8)
    return cv2.distanceTransform(inv_raw, cv2.DIST_L2, 5)


def add_bin_counts(distances: np.ndarray, counts: dict[str, int]) -> None:
    for label, lo, hi in BINS:
        if np.isinf(hi):
            counts[label] += int(np.sum(distances >= lo))
        else:
            counts[label] += int(np.sum((distances >= lo) & (distances < hi)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    clip_rows = []
    total_counts = {label: 0 for label, _, _ in BINS}
    total_missing = 0

    clips = [p for p in sorted(args.gt_root.iterdir()) if p.is_dir()]
    for clip_dir in clips:
        clip = clip_dir.name
        gt_mask_dir = clip_dir / "masks"
        raw_mask_dir = args.raw_root / clip / "masks"
        if not raw_mask_dir.exists():
            print(f"[skip] raw mask dir not found: {raw_mask_dir}")
            continue

        clip_counts = {label: 0 for label, _, _ in BINS}
        clip_missing = 0
        for gt_path in mask_files(gt_mask_dir):
            raw_path = raw_mask_dir / gt_path.name
            if not raw_path.exists():
                print(f"[skip] raw mask not found: {raw_path}")
                continue

            gt = read_mask(gt_path)
            raw = read_mask(raw_path)
            missing = gt & ~raw
            missing_count = int(missing.sum())
            dist = distance_to_raw_mask(raw)
            missing_dist = dist[missing]

            frame_counts = {label: 0 for label, _, _ in BINS}
            if missing_count:
                add_bin_counts(missing_dist, frame_counts)
                add_bin_counts(missing_dist, clip_counts)
                add_bin_counts(missing_dist, total_counts)

            clip_missing += missing_count
            total_missing += missing_count
            row = {
                "clip": clip,
                "frame": gt_path.name,
                "missing_pixels": missing_count,
            }
            for label, _, _ in BINS:
                row[label] = frame_counts[label]
                row[f"{label} ratio"] = frame_counts[label] / missing_count if missing_count else 0.0
            frame_rows.append(row)

        clip_row = {"clip": clip, "missing_pixels": clip_missing}
        for label, _, _ in BINS:
            clip_row[label] = clip_counts[label]
            clip_row[f"{label} ratio"] = clip_counts[label] / clip_missing if clip_missing else 0.0
        clip_rows.append(clip_row)

    summary_rows = []
    for label, _, _ in BINS:
        summary_rows.append(
            {
                "distance_from_raw_mask": label,
                "missing_pixels": total_counts[label],
                "missing_pixels_ratio": total_counts[label] / total_missing if total_missing else 0.0,
            }
        )

    for name, rows in [
        ("frame_missing_distance.csv", frame_rows),
        ("clip_missing_distance.csv", clip_rows),
        ("summary_missing_distance.csv", summary_rows),
    ]:
        path = args.out_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    md = args.out_dir / "summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Missing Distance from SAM Raw Mask\n\n")
        f.write("| Distance from raw mask | Missing pixels ratio |\n")
        f.write("|---|---:|\n")
        for row in summary_rows:
            f.write(f"| {row['distance_from_raw_mask']} | {row['missing_pixels_ratio']:.4f} |\n")
        near_10 = sum(total_counts[label] for label in ["0-5 px", "5-10 px"])
        near_10_ratio = near_10 / total_missing if total_missing else 0.0
        f.write(f"\nMissing pixels within 10 px: {near_10_ratio:.4f}\n")

    print(f"Wrote missing-distance analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
