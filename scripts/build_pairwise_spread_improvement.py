import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OURS_NAME = "Ours-Balanced"
CONTROLS = [
    ("Area-matched dilation for Balanced", "area_matched_dilation"),
    ("Area-matched distance-only for Balanced", "area_matched_distance_only"),
]
METRIC_COLS = {
    "BTE": "boundary_te",
    "ResProxy": "residue_diff_le_10",
    "Outside": "outside_changed_fraction",
}


def paired_df(df: pd.DataFrame, control_name: str) -> pd.DataFrame:
    ours = df[df["method"] == OURS_NAME].copy().set_index("clip")
    ctrl = df[df["method"] == control_name].copy().set_index("clip")
    common = sorted(set(ours.index) & set(ctrl.index))
    rows = []
    for clip in common:
        o = ours.loc[clip]
        c = ctrl.loc[clip]
        row = {
            "clip": clip,
            "spread_group": o["spread_group"],
            "temporal_spread": o["temporal_spread"],
        }
        for metric, col in METRIC_COLS.items():
            row[f"{metric}_ours"] = o[col]
            row[f"{metric}_baseline"] = c[col]
            row[f"{metric}_improvement"] = c[col] - o[col]
            row[f"{metric}_win"] = int(o[col] < c[col])
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_md(path: Path, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    cols = [
        "group",
        "comparison",
        "clips",
        "BTE win",
        "ResProxy win",
        "Outside win",
        "BTE improvement mean",
        "ResProxy improvement mean",
        "Outside improvement mean",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Pairwise Temporal Spread Improvement Analysis\n\n")
        f.write(
            "Improvement is computed as baseline metric minus Ours-Balanced metric; "
            "positive means Ours-Balanced is lower/better.\n\n"
        )
        f.write("## Low/Mid/High Win Summary\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in summary[summary["group"].isin(["low", "mid", "high"])].iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n## Correlations\n\n")
        f.write("| comparison | metric | pearson_r | spearman_r | improvement_mean |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for _, row in corr.iterrows():
            f.write(
                f"| {row['comparison']} | {row['metric']} | "
                f"{row['pearson_r']:.6f} | {row['spearman_r']:.6f} | "
                f"{row['improvement_mean']:.6f} |\n"
            )


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True})
    colors_by_group = {"low": "#4C78A8", "mid": "#F58518", "high": "#54A24B"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row_idx, (control_name, short_name) in enumerate(CONTROLS):
        pair = paired_df(df, control_name)
        for col_idx, metric in enumerate(["ResProxy", "BTE"]):
            ax = axes[row_idx, col_idx]
            ycol = f"{metric}_improvement"
            colors = pair["spread_group"].map(colors_by_group)
            ax.scatter(
                pair["temporal_spread"],
                pair[ycol],
                c=colors,
                s=28,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.25,
            )
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
            z = np.polyfit(pair["temporal_spread"], pair[ycol], 1)
            xs = np.linspace(pair["temporal_spread"].min(), pair["temporal_spread"].max(), 100)
            ax.plot(xs, z[0] * xs + z[1], color="#D62728", linewidth=1.5)
            r = pair["temporal_spread"].corr(pair[ycol], method="pearson")
            ax.set_title(f"{short_name}\n{metric}: baseline - ours, r={r:.3f}")
            ax.set_ylabel(f"{metric} improvement")
            if row_idx == 1:
                ax.set_xlabel("Temporal spread")
    fig.tight_layout()
    fig.savefig(out_dir / "spread_vs_improvement_scatter.png", dpi=220)
    plt.close(fig)

    for control_name, short_name in CONTROLS:
        pair = paired_df(df, control_name)
        for metric in ["ResProxy", "BTE"]:
            ycol = f"{metric}_improvement"
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            colors = pair["spread_group"].map(colors_by_group)
            ax.scatter(
                pair["temporal_spread"],
                pair[ycol],
                c=colors,
                s=32,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.25,
            )
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
            z = np.polyfit(pair["temporal_spread"], pair[ycol], 1)
            xs = np.linspace(pair["temporal_spread"].min(), pair["temporal_spread"].max(), 100)
            ax.plot(xs, z[0] * xs + z[1], color="#D62728", linewidth=1.5)
            r = pair["temporal_spread"].corr(pair[ycol], method="pearson")
            rho = pair["temporal_spread"].corr(pair[ycol], method="spearman")
            ax.set_title(f"Temporal spread vs {metric} improvement\nOurs-Balanced vs {control_name}")
            ax.set_xlabel("Temporal spread")
            ax.set_ylabel(f"{metric}_baseline - {metric}_ours")
            ax.text(
                0.02,
                0.98,
                f"Pearson r={r:.3f}\nSpearman rho={rho:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"spread_vs_{metric.lower()}_improvement_{short_name}.png", dpi=220)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    summary_rows = []
    corr_rows = []
    all_pairs = []
    for control_name, short_name in CONTROLS:
        pair = paired_df(df, control_name)
        comparison = f"Ours-Balanced vs {control_name}"
        pair.insert(1, "comparison", comparison)
        pair.to_csv(args.out_dir / f"clip_pairwise_{short_name}.csv", index=False)
        all_pairs.append(pair)

        for group in ["low", "mid", "high", "all"]:
            g = pair if group == "all" else pair[pair["spread_group"] == group]
            row = {"group": group, "comparison": comparison, "clips": len(g)}
            for metric in ["BTE", "ResProxy", "Outside"]:
                row[f"{metric} win"] = int(g[f"{metric}_win"].sum())
                row[f"{metric} win ratio"] = float(g[f"{metric}_win"].mean()) if len(g) else np.nan
                row[f"{metric} baseline mean"] = float(g[f"{metric}_baseline"].mean()) if len(g) else np.nan
                row[f"{metric} ours mean"] = float(g[f"{metric}_ours"].mean()) if len(g) else np.nan
                row[f"{metric} improvement mean"] = float(g[f"{metric}_improvement"].mean()) if len(g) else np.nan
            summary_rows.append(row)

        for metric in ["ResProxy", "BTE"]:
            x = pair["temporal_spread"]
            y = pair[f"{metric}_improvement"]
            corr_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "definition": f"{metric}_baseline - {metric}_ours",
                    "n": len(pair),
                    "pearson_r": float(x.corr(y, method="pearson")),
                    "spearman_r": float(x.corr(y, method="spearman")),
                    "improvement_mean": float(y.mean()),
                    "improvement_median": float(y.median()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    corr = pd.DataFrame(corr_rows)
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)

    summary.to_csv(args.out_dir / "low_mid_high_pairwise_win_summary.csv", index=False)
    corr.to_csv(args.out_dir / "spread_improvement_correlations.csv", index=False)
    all_pairs_df.to_csv(args.out_dir / "clip_pairwise_all_controls.csv", index=False)
    write_summary_md(args.out_dir / "summary.md", summary, corr)
    save_plots(df, args.out_dir)

    cols = [
        "group",
        "comparison",
        "clips",
        "BTE win",
        "ResProxy win",
        "Outside win",
        "BTE improvement mean",
        "ResProxy improvement mean",
        "Outside improvement mean",
    ]
    print(args.out_dir)
    print(summary[cols].to_string(index=False))
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
