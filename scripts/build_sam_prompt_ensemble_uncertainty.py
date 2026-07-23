import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def write_u8(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Failed to write {path}")


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty GT mask")
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def jitter_box(box: np.ndarray, shape: tuple[int, int], jitter_frac: float, expand_frac: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = box.astype(np.float32)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    x0 -= bw * expand_frac
    x1 += bw * expand_frac
    y0 -= bh * expand_frac
    y1 += bh * expand_frac
    jitter = np.array(
        [
            rng.uniform(-jitter_frac, jitter_frac) * bw,
            rng.uniform(-jitter_frac, jitter_frac) * bh,
            rng.uniform(-jitter_frac, jitter_frac) * bw,
            rng.uniform(-jitter_frac, jitter_frac) * bh,
        ],
        dtype=np.float32,
    )
    x0, y0, x1, y1 = np.array([x0, y0, x1, y1], dtype=np.float32) + jitter
    x0 = float(np.clip(x0, 0, w - 2))
    y0 = float(np.clip(y0, 0, h - 2))
    x1 = float(np.clip(x1, x0 + 1, w - 1))
    y1 = float(np.clip(y1, y0 + 1, h - 1))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--jitter-frac", type=float, default=0.10)
    parser.add_argument("--box-expand-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    clips = args.clips or sorted(p.name for p in args.probe_root.iterdir() if (p / "frames").is_dir())
    rng = np.random.default_rng(args.seed)
    rows = []
    args.out_root.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        clip_dir = args.probe_root / clip
        for frame_path in sorted((clip_dir / "frames").glob("*.png")):
            name = frame_path.name
            frame = read_color(frame_path)
            gt = read_mask(clip_dir / "masks" / name)
            gt_box = bbox_from_mask(gt)
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            stack = []
            scores = []
            for _ in range(args.samples):
                box = jitter_box(gt_box, gt.shape, args.jitter_frac, args.box_expand_frac, rng)
                masks, sam_scores, _ = predictor.predict(box=box, multimask_output=True)
                index = int(np.argmax(sam_scores))
                stack.append(masks[index].astype(np.float32))
                scores.append(float(sam_scores[index]))

            probs = np.mean(np.stack(stack, axis=0), axis=0)
            uncertainty = 4.0 * probs * (1.0 - probs)
            write_u8(args.out_root / clip / "P" / name, probs * 255.0)
            write_u8(args.out_root / clip / "U" / name, uncertainty * 255.0)
            write_u8(args.out_root / clip / "mean_mask" / name, (probs >= 0.5).astype(np.uint8) * 255)

            rows.append(
                {
                    "clip": clip,
                    "frame": frame_path.stem,
                    "samples": args.samples,
                    "jitter_frac": args.jitter_frac,
                    "box_expand_frac": args.box_expand_frac,
                    "sam_score_mean": float(np.mean(scores)),
                    "uncertainty_mean": float(np.mean(uncertainty)),
                    "uncertainty_p95": float(np.quantile(uncertainty, 0.95)),
                }
            )

    with (args.out_root / "ensemble_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out_root)


if __name__ == "__main__":
    main()
