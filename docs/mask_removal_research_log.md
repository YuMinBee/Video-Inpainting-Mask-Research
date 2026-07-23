# Mask Quality and Object Removal Research Log

This note records the main experimental path and the current conclusion. The project did not converge to a strong paper claim for temporal occupancy alone, but it produced a useful and reproducible investigation of how mask quality affects video object removal.

## Research Question

The initial hypothesis was that temporal occupancy memory could improve video object removal by expanding incomplete masks in temporally consistent regions. During the experiments, the question became more specific:

> Is a segmentation mask the right mask for object removal, and can temporal occupancy or SAM uncertainty identify the missing removal-support regions?

The answer from the current experiments is cautious. Mask quality strongly matters for removal, but the dominant errors in the synthetic-SAM setup are mostly local boundary errors, where simple dilation is a very strong baseline.

## What Was Built

The repository now contains a pipeline for comparing object-removal masks under controlled synthetic ground truth and real AI-Hub style evaluation.

Key scripts:

- `scripts/build_synthetic_gt_probe.py`: builds synthetic videos with a clean background target.
- `scripts/build_sam_jitter_masks.py`: generates imperfect SAM masks from jittered box prompts.
- `scripts/build_synthetic_gt_mask_variants.py`: builds boundary-only, temporal union, Ours-Balanced, area-matched dilation, and area-matched distance-only masks.
- `scripts/evaluate_synthetic_gt_outputs.py`: evaluates ProPainter outputs against the clean background.
- `scripts/evaluate_mask_correction_efficiency.py`: measures missing-region recovery before running ProPainter.
- `scripts/analyze_missing_distance.py`: measures whether missing object pixels are near the raw-mask boundary.
- `scripts/analyze_temporal_spread_groups.py`: splits AI-Hub clips by temporal spread and compares operating points.
- `scripts/build_pairwise_spread_improvement.py`: computes spread-vs-improvement correlation and pairwise win counts.
- `scripts/build_sam_prompt_ensemble_uncertainty.py`: builds SAM prompt-ensemble probability and uncertainty maps.
- `scripts/evaluate_uncertainty_baseline.py`: tests whether high-uncertainty pixels recover missing object regions.
- `scripts/summarize_oracle_best_candidates.py`: computes oracle best-of-candidates from existing synthetic-SAM outputs.

## Main Experimental Findings

### 1. Mask quality directly affects object removal

The synthetic-GT setup compares ProPainter outputs against the clean background instead of relying only on residue proxy metrics. This confirmed that imperfect masks can lead to visible and measurable removal degradation.

This matters because residue proxy alone can be misleading: it is useful on real videos without ground truth, but it is not a replacement for clean-background evaluation.

### 2. SAM masks can be insufficient

Jittered SAM masks were not close to perfect masks in the synthetic setup. The measured mask errors showed substantial missing object regions.

Representative diagnostic:

- `sam_jitter_5` mask IoU was much lower than perfect masks.
- Missing object ratio was large enough to affect downstream removal.
- Boundary error and temporal jitter increased relative to the GT alpha mask.

This supports the motivation that automatic segmentation masks are not always valid removal masks.

### 3. The synthetic-SAM errors are mostly local boundary errors

Missing-distance analysis showed that a large fraction of missing object pixels were close to the raw SAM mask:

| Distance from raw mask | Missing pixels ratio |
|---|---:|
| 0-5 px | 0.3319 |
| 5-10 px | 0.2221 |
| 10-20 px | 0.2072 |
| 20+ px | 0.2388 |

Thus, about 55.4% of missing pixels were within 10 pixels, and about 76.1% were within 20 pixels. This explains why simple dilation performs strongly in this setting.

Conclusion:

> This synthetic-SAM setting is dominated by local boundary shrinkage, not by a temporal occupancy failure mode.

### 4. Temporal occupancy alone is not a strong enough solution

Ours-Balanced often improves residue-proxy and BTE compared with some controls, but it does not consistently beat area-matched dilation. The most important failure is that geometric baselines with the same mask budget remain very competitive.

In the SAM synthetic-GT evaluation, masked MAE oracle wins were:

| Method | Oracle win by masked MAE |
|---|---:|
| Area-matched dilation | 6/10 |
| Temporal union | 2/10 |
| Area-matched distance-only | 1/10 |
| Boundary-only | 1/10 |
| Ours-Balanced | 0/10 |

This makes it risky to present temporal occupancy as the main solution.

### 5. SAM prompt-ensemble uncertainty did not localize missing regions

SAM prompt ensembles were tested with probability and uncertainty maps:

- `P`: mean mask probability over jittered prompts
- `U = 4P(1-P)`: prompt-ensemble uncertainty

At the same extra-mask budget as area-matched dilation, uncertainty-based selection was much weaker:

| Method | Recovery | Precision | FalseAdd | Extra |
|---|---:|---:|---:|---:|
| Area-matched dilation | 0.7990 | 0.5074 | 0.3000 | 0.6031 |
| Area-matched distance-only | 0.7579 | 0.4850 | 0.3113 | 0.6045 |
| Ours-Balanced | 0.7361 | 0.4596 | 0.3244 | 0.6045 |
| Uncertainty top-P | 0.2833 | 0.1988 | 0.4756 | 0.6031 |
| Uncertainty top-U | 0.2830 | 0.1987 | 0.4757 | 0.6031 |

Conclusion:

> Box-jitter SAM uncertainty is not a reliable missing-region cue in this setup.

## Current Research Conclusion

The useful conclusion is not that temporal occupancy solves removal-mask refinement. The better conclusion is:

> Object removal needs removal-support masks, not just segmentation masks. However, in the current synthetic-SAM setting, the dominant mask error is local boundary under-coverage, so simple area-matched dilation is a strong and often superior baseline. Temporal occupancy alone is therefore not enough to support a strong paper claim.

This is a negative result, but it is informative. It prevents overclaiming and leaves a clearer future direction: a useful method needs stronger cues than temporal occupancy or prompt-ensemble uncertainty alone, especially cues that distinguish true missing object support from background over-removal.

## Important Result Locations

Large result folders are stored on the D drive:

```text
D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe
D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_masks
D:\DCG-TR_experiment_results\results\synthetic_gt_probe
D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_evaluation

D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_raw_masks
D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_sam_jitter5_raw
D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_sam_jitter5_masks
D:\DCG-TR_experiment_results\results\synthetic_gt_probe_sam_jitter5
D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_jitter5_evaluation

D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_jitter5_missing_distance
D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_jitter5_mask_correction
D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_jitter5_oracle_best
D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_prompt_ensemble_j10_s16
D:\DCG-TR_experiment_results\experiments\synthetic_gt_sam_uncertainty_baseline_j10_s16

D:\DCG-TR_experiment_results\experiments\4090_rerun\aihub_subset_100\spread_group_analysis
```

Original experiment/result backups:

```text
D:\DCG-TR_experiment_backup_20260627\workspace\experiments
D:\DCG-TR_experiment_backup_20260627\workspace\results
```

## If This Work Is Continued

The next version should not claim that temporal occupancy alone solves mask refinement. A stronger follow-up would need one of the following:

- a cue that separates missing object support from background expansion better than dilation;
- a benchmark where temporal mask spread, not local boundary shrinkage, is the dominant failure mode;
- a learning-based removal-mask predictor trained with synthetic GT and evaluated against strong area-matched dilation and distance-only controls.
