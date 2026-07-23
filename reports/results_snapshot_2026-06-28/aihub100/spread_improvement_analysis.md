# Pairwise Temporal Spread Improvement Analysis

Improvement is computed as baseline metric minus Ours-Balanced metric; positive means Ours-Balanced is lower/better.

## Low/Mid/High Win Summary

| group | comparison | clips | BTE win | ResProxy win | Outside win | BTE improvement mean | ResProxy improvement mean | Outside improvement mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | Ours-Balanced vs Area-matched dilation for Balanced | 34 | 23 | 31 | 7 | 0.000136 | 0.006103 | -0.001473 |
| mid | Ours-Balanced vs Area-matched dilation for Balanced | 33 | 23 | 32 | 10 | 0.000065 | 0.006672 | -0.000820 |
| high | Ours-Balanced vs Area-matched dilation for Balanced | 33 | 29 | 30 | 10 | 0.000168 | 0.005546 | -0.000430 |
| low | Ours-Balanced vs Area-matched distance-only for Balanced | 34 | 20 | 24 | 9 | 0.000040 | 0.002889 | -0.000502 |
| mid | Ours-Balanced vs Area-matched distance-only for Balanced | 33 | 13 | 26 | 10 | -0.000036 | 0.003546 | -0.000705 |
| high | Ours-Balanced vs Area-matched distance-only for Balanced | 33 | 22 | 24 | 8 | 0.000045 | 0.002876 | -0.000409 |

## Correlations

| comparison | metric | pearson_r | spearman_r | improvement_mean |
|---|---|---:|---:|---:|
| Ours-Balanced vs Area-matched dilation for Balanced | ResProxy | -0.068643 | 0.003072 | 0.006107 |
| Ours-Balanced vs Area-matched dilation for Balanced | BTE | 0.012927 | 0.138302 | 0.000123 |
| Ours-Balanced vs Area-matched distance-only for Balanced | ResProxy | -0.129458 | -0.018050 | 0.003102 |
| Ours-Balanced vs Area-matched distance-only for Balanced | BTE | 0.036426 | 0.042076 | 0.000017 |
