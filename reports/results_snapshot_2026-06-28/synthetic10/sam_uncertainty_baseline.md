# SAM Ensemble Uncertainty Baseline

Uncertainty/P baselines select top-score pixels outside the raw SAM mask using the same per-frame added-pixel budget as area-matched dilation.

| Method | Score | Recovery | Precision | FalseAdd | Extra |
|---|---|---:|---:|---:|---:|
| area_matched_dilation | mask | 0.7990 | 0.5074 | 0.3000 | 0.6031 |
| area_matched_distance_only | mask | 0.7579 | 0.4850 | 0.3113 | 0.6045 |
| ours_balanced | mask | 0.7361 | 0.4596 | 0.3244 | 0.6045 |
| uncertainty | P | 0.2833 | 0.1988 | 0.4756 | 0.6031 |
| uncertainty | U | 0.2830 | 0.1987 | 0.4757 | 0.6031 |
