import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def read_float_map(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def mask_files(mask_dir: Path) -> list[Path]:
    return sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg"))


def top_budget(score: np.ndarray, allowed: np.ndarray, budget: int) -> np.ndarray:
    out = np.zeros(score.shape, dtype=bool)
    if budget <= 0 or not np.any(allowed):
        return out
    ys, xs = np.where(allowed)
    values = score[ys, xs]
    k = min(int(budget), len(values))
    if k <= 0:
        return out
    idx = np.argpartition(values, -k)[-k:]
    out[ys[idx], xs[idx]] = True
    return out


def metrics(add: np.ndarray, missing: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    hit = add & missing
    add_count = int(add.sum())
    missing_count = int(missing.sum())
    gt_area = max(int(gt.sum()), 1)
    return {
        "recovery": float(hit.sum() / max(missing_count, 1)),
        "precision": float(hit.sum() / max(add_count, 1)),
        "false_add": float((add & ~gt).sum() / gt_area),
        "extra": float(add_count / gt_area),
        "added_pixels": add_count,
        "missing_pixels": missing_count,
    }


def evaluate_method(gt_root: Path, raw_root: Path, method_root: Path, ensemble_root: Path | None, method: str, score_name: str) -> list[dict]:
    rows = []
    for clip_dir in sorted(gt_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip = clip_dir.name
        gt_mask_dir = clip_dir / "masks"
        raw_mask_dir = raw_root / clip / "masks"
        method_mask_dir = method_root / clip / "masks"
        for gt_path in mask_files(gt_mask_dir):
            name = gt_path.name
            gt = read_mask(gt_path)
            raw = read_mask(raw_mask_dir / name)
            missing = gt & ~raw
            if method == "uncertainty":
                assert ensemble_root is not None
                score = read_float_map(ensemble_root / clip / score_name / name)
                ref = read_mask(method_mask_dir / name)
                budget = int((ref & ~raw).sum())
                add = top_budget(score, ~raw, budget)
            else:
                refined = read_mask(method_mask_dir / name)
                add = refined & ~raw
            row = {"clip": clip, "frame": gt_path.stem, "method": method, "score": score_name}
            row.update(metrics(add, missing, gt))
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["recovery", "precision", "false_add", "extra", "added_pixels", "missing_pixels"]
    out = []
    groups = sorted({(r["method"], r["score"]) for r in rows})
    for method, score in groups:
        subset = [r for r in rows if r["method"] == method and r["score"] == score]
        item = {"method": method, "score": score, "frames": len(subset)}
        for key in keys:
            item[key] = float(np.mean([r[key] for r in subset]))
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--area-dilation-root", required=True, type=Path)
    parser.add_argument("--distance-only-root", required=True, type=Path)
    parser.add_argument("--ours-root", required=True, type=Path)
    parser.add_argument("--ensemble-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += evaluate_method(args.gt_root, args.raw_root, args.area_dilation_root, None, "area_matched_dilation", "mask")
    rows += evaluate_method(args.gt_root, args.raw_root, args.distance_only_root, None, "area_matched_distance_only", "mask")
    rows += evaluate_method(args.gt_root, args.raw_root, args.ours_root, None, "ours_balanced", "mask")
    rows += evaluate_method(args.gt_root, args.raw_root, args.area_dilation_root, args.ensemble_root, "uncertainty", "U")
    rows += evaluate_method(args.gt_root, args.raw_root, args.area_dilation_root, args.ensemble_root, "uncertainty", "P")

    frame_rows = sorted(rows, key=lambda r: (r["method"], r["score"], r["clip"], r["frame"]))
    summary = summarize(frame_rows)
    write_csv(args.out_dir / "frame_uncertainty_baseline.csv", frame_rows)
    write_csv(args.out_dir / "summary_uncertainty_baseline.csv", summary)

    with (args.out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SAM Ensemble Uncertainty Baseline\n\n")
        f.write("Uncertainty/P baselines select top-score pixels outside the raw SAM mask using the same per-frame added-pixel budget as area-matched dilation.\n\n")
        f.write("| Method | Score | Recovery | Precision | FalseAdd | Extra |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['method']} | {row['score']} | {row['recovery']:.4f} | "
                f"{row['precision']:.4f} | {row['false_add']:.4f} | {row['extra']:.4f} |\n"
            )

    print(args.out_dir)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
