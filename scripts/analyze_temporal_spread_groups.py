import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


METRICS = {
    "BTE": "boundary_te",
    "Outside": "outside_changed_fraction",
    "Extra": "extra_mask_ratio",
    "ResProxy": "residue_diff_le_10",
}


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def compute_spread(mask_dir: Path) -> dict[str, float]:
    files = sorted(mask_dir.glob("*.png"))
    if not files:
        files = sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg"))
    if not files:
        raise FileNotFoundError(f"No masks in {mask_dir}")

    union = None
    areas = []
    for path in files:
        mask = read_mask(path)
        areas.append(float(mask.sum()))
        union = mask.copy() if union is None else (union | mask)

    mean_area = float(np.mean(areas))
    union_area = float(union.sum()) if union is not None else 0.0
    spread = union_area / mean_area if mean_area > 0 else 0.0
    occupancy_ratio = mean_area / union_area if union_area > 0 else 0.0
    return {
        "frames": len(files),
        "mean_mask_area": mean_area,
        "union_area": union_area,
        "temporal_spread": spread,
        "temporal_occupancy_ratio": occupancy_ratio,
    }


def assign_tertiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("temporal_spread").reset_index(drop=True)
    n = len(out)
    labels = []
    for idx in range(n):
        q = idx / n
        if q < 1 / 3:
            labels.append("low")
        elif q < 2 / 3:
            labels.append("mid")
        else:
            labels.append("high")
    out["spread_group"] = labels
    return out


def read_metrics(path: Path, keep_methods: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if keep_methods:
        df = df[df["method"].isin(keep_methods)].copy()
    return df


def win_counts(group_df: pd.DataFrame, ours_name: str, control_name: str) -> dict[str, int | str]:
    ours = group_df[group_df["method"] == ours_name].set_index("clip")
    ctrl = group_df[group_df["method"] == control_name].set_index("clip")
    common = sorted(set(ours.index) & set(ctrl.index))
    row: dict[str, int | str] = {
        "group": str(group_df["spread_group"].iloc[0]),
        "comparison": f"{ours_name} vs {control_name}",
        "clips": len(common),
    }
    if not common:
        for name in list(METRICS) + ["Outside+ResProxy"]:
            row[f"{name} win"] = 0
        return row

    o = ours.loc[common]
    c = ctrl.loc[common]
    for name, col in METRICS.items():
        row[f"{name} win"] = int((o[col] < c[col]).sum())
    row["Outside+ResProxy win"] = int(
        ((o[METRICS["Outside"]] < c[METRICS["Outside"]]) & (o[METRICS["ResProxy"]] < c[METRICS["ResProxy"]])).sum()
    )
    return row

def write_markdown_table(f, df: pd.DataFrame, float_fmt: str = ".6f") -> None:
    cols = list(df.columns)
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(format(value, float_fmt))
            else:
                values.append(str(value))
        f.write("| " + " | ".join(values) + " |\n")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--choice-a-csv", required=True, type=Path)
    parser.add_argument("--area-dilation-csv", required=True, type=Path)
    parser.add_argument("--distance-only-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    spread_rows = []
    for clip_dir in sorted(args.probe_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        mask_dir = clip_dir / "masks"
        if not mask_dir.exists():
            continue
        row = {"clip": clip_dir.name}
        row.update(compute_spread(mask_dir))
        spread_rows.append(row)

    spread_df = assign_tertiles(pd.DataFrame(spread_rows))
    spread_df.to_csv(args.out_dir / "clip_spread.csv", index=False)

    choice = read_metrics(
        args.choice_a_csv,
        ["Boundary-only", "Temporal union", "Ours-Conservative", "Ours-Balanced"],
    )
    dilation = read_metrics(args.area_dilation_csv)
    distance = read_metrics(args.distance_only_csv)
    metrics = pd.concat([choice, dilation, distance], ignore_index=True)
    metrics = metrics.merge(spread_df[["clip", "temporal_spread", "temporal_occupancy_ratio", "spread_group"]], on="clip")
    metrics.to_csv(args.out_dir / "clip_metrics_with_spread.csv", index=False)

    summary = (
        metrics.groupby(["spread_group", "method"], sort=False)
        .agg(
            clips=("clip", "nunique"),
            spread_mean=("temporal_spread", "mean"),
            BTE=("boundary_te", "mean"),
            Outside=("outside_changed_fraction", "mean"),
            Extra=("extra_mask_ratio", "mean"),
            ResProxy=("residue_diff_le_10", "mean"),
        )
        .reset_index()
    )
    order = {"low": 0, "mid": 1, "high": 2}
    summary["group_order"] = summary["spread_group"].map(order)
    summary = summary.sort_values(["group_order", "method"]).drop(columns=["group_order"])
    summary.to_csv(args.out_dir / "spread_group_summary.csv", index=False)

    controls = [
        "Boundary-only",
        "Ours-Conservative",
        "Temporal union",
        "Area-matched dilation for Balanced",
        "Area-matched distance-only for Balanced",
    ]
    win_rows = []
    for group in ["low", "mid", "high"]:
        group_df = metrics[metrics["spread_group"] == group]
        for control in controls:
            win_rows.append(win_counts(group_df, "Ours-Balanced", control))
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(args.out_dir / "spread_group_win_counts.csv", index=False)

    with (args.out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Temporal Spread Group Analysis\n\n")
        f.write("Temporal spread is computed as union mask area divided by mean per-frame mask area.\n\n")
        f.write("## Group Means\n\n")
        write_markdown_table(f, summary)
        f.write("\n## Ours-Balanced Clip-Level Win Counts\n\n")
        write_markdown_table(f, win_df, float_fmt=".0f")

    print(f"Wrote temporal-spread analysis to {args.out_dir}")


if __name__ == "__main__":
    main()


