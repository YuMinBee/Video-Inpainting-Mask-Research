#!/usr/bin/env python3
"""Create paper-ready qualitative panels for the selective expansion study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    probe_root: Path
    metrics: Path
    method_metric_names: dict[str, str]
    output_roots: dict[str, Path]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    dataset: str
    clip: str
    frame: str
    title: str
    note: str


DATASETS = {
    "driving": DatasetSpec(
        name="Driving clips",
        probe_root=Path("experiments/aihub_subset_20_real"),
        metrics=Path("experiments/selective_expansion/full20_evaluation/frame_metrics.csv"),
        method_metric_names={
            "Boundary-only": "Boundary-only",
            "Temporal union": "Temporal union",
            "Ours r10 t0.10": "Aggressive pixel r10 t0.10",
            "Ours r10 t0.15": "Aggressive pixel r10 t0.15",
        },
        output_roots={
            "Boundary-only": Path("results/selective_expansion/basic/boundary_only_mask/propainter_outputs"),
            "Temporal union": Path("results/mask_union/temporal_union/propainter_outputs"),
            "Ours r10 t0.10": Path("results/selective_expansion/aggressive/pixel_r10_t0p10/propainter_outputs"),
            "Ours r10 t0.15": Path("results/selective_expansion/aggressive/pixel_r10_t0p15/propainter_outputs"),
        },
    ),
    "davis": DatasetSpec(
        name="DAVIS 2017 val",
        probe_root=Path("experiments/davis2017_val/probe"),
        metrics=Path("experiments/davis2017_val/evaluation/frame_metrics.csv"),
        method_metric_names={
            "Boundary-only": "Boundary-only",
            "Temporal union": "Temporal union",
            "Ours r10 t0.10": "Ours r10 t0.10",
            "Ours r10 t0.15": "Ours r10 t0.15",
        },
        output_roots={
            "Boundary-only": Path("results/davis2017_val/boundary_only/propainter_outputs"),
            "Temporal union": Path("results/davis2017_val/temporal_union/propainter_outputs"),
            "Ours r10 t0.10": Path("results/davis2017_val/ours_r10_t0p10/propainter_outputs"),
            "Ours r10 t0.15": Path("results/davis2017_val/ours_r10_t0p15/propainter_outputs"),
        },
    ),
}


CASES = [
    CaseSpec(
        case_id="01_success_driving",
        dataset="driving",
        clip="I-210715_I03012_W06",
        frame="frame_0016",
        title="Success - driving",
        note="Boundary-only leaves residue, temporal union edits wider background, ours balances both.",
    ),
    CaseSpec(
        case_id="02_success_davis",
        dataset="davis",
        clip="motocross-jump",
        frame="00028",
        title="Success - DAVIS",
        note="DAVIS shows the same trade-off trend: ours suppresses union over-change while retaining residue gains.",
    ),
    CaseSpec(
        case_id="03_temporal_union_failure",
        dataset="driving",
        clip="I-210618_I01001_W01",
        frame="frame_0011",
        title="Temporal union failure",
        note="Temporal union expands too broadly and changes background; selective expansion suppresses this.",
    ),
    CaseSpec(
        case_id="04_ours_failure",
        dataset="driving",
        clip="I-210720_O12052_T04",
        frame="frame_0001",
        title="Ours failure / limitation",
        note="Selected extra region is too broad, so ours still increases outside changes.",
    ),
]


METHOD_ORDER = ["Boundary-only", "Temporal union", "Ours r10 t0.10", "Ours r10 t0.15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/paper_visualizations/case_panels"))
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    return parser.parse_args()


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics[(row["clip"], row["frame"], row["method"])] = {
                "boundary_te": float(row["boundary_te"]),
                "outside": float(row["outside_changed_fraction"]),
                "extra": float(row["extra_mask_ratio"]),
                "residue": float(row["residue_diff_le_10"]),
            }
    return metrics


def label(image: np.ndarray, text: str, scale: float = 0.62) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (11, 27), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def footer(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(out, text, (9, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    idx = mask > 0
    out[idx] = ((1.0 - alpha) * out[idx] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def crop_bounds(mask: np.ndarray, pad: int = 80) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    return max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0), min(int(xs.max()) + pad + 1, w), min(int(ys.max()) + pad + 1, h)


def crop(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2]


def resize_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def change_heat(output: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(output.astype(np.float32) - frame.astype(np.float32)), axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 5.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return draw_contour(heat, mask, (255, 255, 255))


def metric_footer(metrics: dict[tuple[str, str, str], dict[str, float]], dataset: DatasetSpec, case: CaseSpec, method: str) -> str:
    metric_name = dataset.method_metric_names[method]
    row = metrics[(case.clip, case.frame, metric_name)]
    return f"BTE {row['boundary_te']:.3f} | out {row['outside']:.3f} | res {row['residue']:.3f}"


def make_panel(args: argparse.Namespace, case: CaseSpec, cropped: bool) -> Path:
    dataset = DATASETS[case.dataset]
    metrics = load_metrics(dataset.metrics)
    frame_name = f"{case.frame}.png"
    frame = read_color(dataset.probe_root / case.clip / "frames" / frame_name)
    mask = read_gray(dataset.probe_root / case.clip / "masks" / frame_name)
    bounds = crop_bounds(mask) if cropped else (0, 0, frame.shape[1], frame.shape[0])

    input_vis = draw_contour(overlay_mask(frame, mask, (0, 255, 255), 0.32), mask)
    top_tiles = [
        footer(
            label(resize_tile(crop(input_vis, bounds), args.tile_width, args.tile_height), "Input + mask"),
            f"{dataset.name} | {case.clip}",
        )
    ]
    heat_tiles = [
        footer(
            label(resize_tile(crop(draw_contour(frame, mask), bounds), args.tile_width, args.tile_height), "Mask contour"),
            case.title,
        )
    ]

    for method in METHOD_ORDER:
        output = read_color(dataset.output_roots[method] / case.clip / frame_name)
        top = draw_contour(output, mask)
        heat = change_heat(output, frame, mask)
        top_tiles.append(
            footer(
                label(resize_tile(crop(top, bounds), args.tile_width, args.tile_height), method),
                metric_footer(metrics, dataset, case, method),
            )
        )
        heat_tiles.append(label(resize_tile(crop(heat, bounds), args.tile_width, args.tile_height), f"Change heat: {method}", 0.52))

    spacer = np.full((16, args.tile_width * len(top_tiles), 3), 255, dtype=np.uint8)
    panel = np.vstack([np.hstack(top_tiles), spacer, np.hstack(heat_tiles)])
    suffix = "crop" if cropped else "full"
    out_path = args.output_root / f"{case.case_id}_{case.dataset}_{case.clip}_{case.frame}_{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), panel):
        raise RuntimeError(f"Failed to write panel: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    written = []
    for case in CASES:
        written.append((case, make_panel(args, case, cropped=False)))
        written.append((case, make_panel(args, case, cropped=True)))

    lines = [
        "# Paper Case Visualizations",
        "",
        "Each case includes a full-frame panel and a crop around the mask. The top row compares outputs; the bottom row shows change heatmaps against the input frame.",
        "",
    ]
    seen = set()
    for case, path in written:
        if case.case_id not in seen:
            lines.extend([f"## {case.title}", "", case.note, ""])
            seen.add(case.case_id)
        rel = path.relative_to(args.output_root)
        lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")
    lines.extend(
        [
            "",
            "## Suggested Caption",
            "",
            "Qualitative comparison of mask expansion strategies. Boundary-only expansion is conservative but may leave object residue. Temporal union removes more residue but can over-edit background regions. The proposed selective expansion provides a middle ground, while the failure case shows that it can still select overly broad regions when the temporal support is ambiguous.",
        ]
    )
    (args.output_root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for _, path in written:
        print(path)


if __name__ == "__main__":
    main()
