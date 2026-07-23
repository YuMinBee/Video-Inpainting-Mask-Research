# Curated Results Snapshot (2026-06-28)

This directory preserves the small, research-critical outputs from the D: drive without committing model frames, masks, videos, checkpoints, or other large generated artifacts.

## Reading order

1. `aihub100/main_table.md` — principal 100-clip operating-point comparison.
2. `aihub100/clip_win_counts.md` — strict clip-level wins for Ours-Balanced.
3. `aihub100/spread_improvement_analysis.md` — test of whether temporal spread predicts improvement.
4. `davis30/results.md` — cross-dataset proxy-metric check.
5. `synthetic10/sam_jitter5_results.md` — clean-background reconstruction with imperfect SAM masks.
6. `synthetic10/missing_distance.md` and `synthetic10/mask_correction_efficiency.md` — error mechanism.
7. `synthetic10/oracle_analysis.md` and `synthetic10/sam_uncertainty_baseline.md` — candidate and uncertainty tests.

The CSV files beside each Markdown table are the machine-readable source summaries. Clip-level CSVs are retained where they are needed to audit aggregate claims. Frame-level CSVs are omitted because they are generated, substantially larger, and recoverable from the full D: drive output.

## Provenance

Files were copied without numerical modification from:

```text
D:\DCG-TR_experiment_results\experiments\4090_rerun\aihub_subset_100
D:\DCG-TR_experiment_results\experiments\4090_rerun\davis2017_val
D:\DCG-TR_experiment_results\experiments\synthetic_gt_*
```

`MANIFEST.sha256` records the content hash of every preserved result file. Paths in the manifest are relative to this snapshot directory.

## Scope

The snapshot is an evidence package, not a fully runnable experiment. Use `reproducibility/README.md` for environments and commands, and retain the D: drive folders for full reproduction from intermediate masks and model outputs.
