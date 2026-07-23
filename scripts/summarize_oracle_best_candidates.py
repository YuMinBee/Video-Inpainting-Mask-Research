import argparse
from pathlib import Path

import pandas as pd


HIGHER_IS_BETTER = {"masked_psnr", "masked_ssim"}
LOWER_IS_BETTER = {"masked_mae", "outside_mae", "boundary_te", "extra_mask_ratio"}


def best_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric in LOWER_IS_BETTER
    idx = df.groupby("clip")[metric].idxmin() if ascending else df.groupby("clip")[metric].idxmax()
    out = df.loc[idx, ["clip", "method", metric]].copy()
    out = out.rename(columns={"method": f"best_method_by_{metric}", metric: f"oracle_{metric}"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clip_summary)
    metrics = ["masked_psnr", "masked_ssim", "masked_mae", "outside_mae", "boundary_te", "extra_mask_ratio"]

    oracle = None
    for metric in metrics:
        rows = best_rows(df, metric)
        oracle = rows if oracle is None else oracle.merge(rows, on="clip")
    oracle.to_csv(args.out_dir / "oracle_best_by_clip.csv", index=False)

    summary_rows = []
    for metric in metrics:
        method_col = f"best_method_by_{metric}"
        counts = oracle[method_col].value_counts().to_dict()
        for method, count in counts.items():
            summary_rows.append({"metric": metric, "best_method": method, "clips": int(count)})
    summary = pd.DataFrame(summary_rows).sort_values(["metric", "clips"], ascending=[True, False])
    summary.to_csv(args.out_dir / "oracle_best_method_counts.csv", index=False)

    primary = best_rows(df, "masked_mae").rename(
        columns={"best_method_by_masked_mae": "oracle_primary_method", "oracle_masked_mae": "oracle_masked_mae"}
    )
    baseline_rows = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method][["clip", "masked_mae", "masked_psnr", "masked_ssim", "outside_mae", "boundary_te"]]
        sub = primary.merge(sub, on="clip", how="left")
        baseline_rows.append(
            {
                "method": method,
                "clips": len(sub),
                "masked_mae_gap_to_oracle": float((sub["masked_mae"] - sub["oracle_masked_mae"]).mean()),
                "oracle_primary_win_clips": int((sub["oracle_primary_method"] == method).sum()),
                "masked_mae": float(sub["masked_mae"].mean()),
                "masked_psnr": float(sub["masked_psnr"].mean()),
                "masked_ssim": float(sub["masked_ssim"].mean()),
                "outside_mae": float(sub["outside_mae"].mean()),
                "boundary_te": float(sub["boundary_te"].mean()),
            }
        )
    gap = pd.DataFrame(baseline_rows)
    gap.to_csv(args.out_dir / "oracle_masked_mae_gap_summary.csv", index=False)

    with (args.out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Oracle Best-of-Candidates on Existing Synthetic-SAM Outputs\n\n")
        f.write("The primary oracle chooses the lowest masked MAE per clip.\n\n")
        f.write("| Method | Oracle primary wins | Masked MAE gap to oracle |\n")
        f.write("|---|---:|---:|\n")
        for _, row in gap.sort_values("masked_mae_gap_to_oracle").iterrows():
            f.write(
                f"| {row['method']} | {int(row['oracle_primary_win_clips'])} | "
                f"{row['masked_mae_gap_to_oracle']:.6f} |\n"
            )
        f.write("\n## Best Method Counts by Metric\n\n")
        f.write("| Metric | Best method | Clips |\n")
        f.write("|---|---|---:|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['metric']} | {row['best_method']} | {int(row['clips'])} |\n")

    print(args.out_dir)
    print(gap.sort_values("masked_mae_gap_to_oracle").to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
