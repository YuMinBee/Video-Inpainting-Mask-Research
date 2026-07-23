# Temporal Spread Group Analysis

Temporal spread is computed as union mask area divided by mean per-frame mask area.

## Group Means

| spread_group | method | clips | spread_mean | BTE | Outside | Extra | ResProxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | Area-matched dilation for Balanced | 34 | 1.514114 | 0.009693 | 0.077297 | 0.162917 | 0.370032 |
| low | Area-matched distance-only for Balanced | 34 | 1.514114 | 0.009597 | 0.078267 | 0.161238 | 0.366818 |
| low | Boundary-only | 34 | 1.514114 | 0.010688 | 0.069266 | 0.050562 | 0.382371 |
| low | Ours-Balanced | 34 | 1.514114 | 0.009557 | 0.078770 | 0.162948 | 0.363928 |
| low | Ours-Conservative | 34 | 1.514114 | 0.011134 | 0.173611 | 0.452409 | 0.168676 |
| low | Temporal union | 34 | 1.514114 | 0.011815 | 0.208967 | 0.585272 | 0.141217 |
| mid | Area-matched dilation for Balanced | 33 | 2.535590 | 0.014173 | 0.062881 | 0.307258 | 0.394965 |
| mid | Area-matched distance-only for Balanced | 33 | 2.535590 | 0.014072 | 0.062997 | 0.306786 | 0.391839 |
| mid | Boundary-only | 33 | 2.535590 | 0.015457 | 0.055929 | 0.084074 | 0.410743 |
| mid | Ours-Balanced | 33 | 2.535590 | 0.014107 | 0.063701 | 0.306943 | 0.388293 |
| mid | Ours-Conservative | 33 | 2.535590 | 0.014252 | 0.208038 | 1.401022 | 0.182547 |
| mid | Temporal union | 33 | 2.535590 | 0.016005 | 0.305442 | 2.032869 | 0.134919 |
| high | Area-matched dilation for Balanced | 33 | 6.837041 | 0.024310 | 0.071714 | 0.500701 | 0.444091 |
| high | Area-matched distance-only for Balanced | 33 | 6.837041 | 0.024187 | 0.071734 | 0.501181 | 0.441421 |
| high | Boundary-only | 33 | 6.837041 | 0.025344 | 0.068925 | 0.179093 | 0.453416 |
| high | Ours-Balanced | 33 | 6.837041 | 0.024142 | 0.072143 | 0.501185 | 0.438545 |
| high | Ours-Conservative | 33 | 6.837041 | 0.018091 | 0.157776 | 3.282239 | 0.275836 |
| high | Temporal union | 33 | 6.837041 | 0.018087 | 0.323489 | 8.307529 | 0.167481 |

## Ours-Balanced Clip-Level Win Counts

| group | comparison | clips | BTE win | Outside win | Extra win | ResProxy win | Outside+ResProxy win |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | Ours-Balanced vs Boundary-only | 34 | 33 | 1 | 0 | 34 | 1 |
| low | Ours-Balanced vs Ours-Conservative | 34 | 26 | 32 | 32 | 3 | 1 |
| low | Ours-Balanced vs Temporal union | 34 | 26 | 32 | 33 | 3 | 1 |
| low | Ours-Balanced vs Area-matched dilation for Balanced | 34 | 23 | 7 | 12 | 31 | 7 |
| low | Ours-Balanced vs Area-matched distance-only for Balanced | 34 | 20 | 9 | 0 | 24 | 5 |
| mid | Ours-Balanced vs Boundary-only | 33 | 33 | 0 | 0 | 33 | 0 |
| mid | Ours-Balanced vs Ours-Conservative | 33 | 16 | 33 | 33 | 0 | 0 |
| mid | Ours-Balanced vs Temporal union | 33 | 16 | 33 | 33 | 0 | 0 |
| mid | Ours-Balanced vs Area-matched dilation for Balanced | 33 | 23 | 10 | 18 | 32 | 9 |
| mid | Ours-Balanced vs Area-matched distance-only for Balanced | 33 | 13 | 10 | 0 | 26 | 7 |
| high | Ours-Balanced vs Boundary-only | 33 | 30 | 1 | 0 | 33 | 1 |
| high | Ours-Balanced vs Ours-Conservative | 33 | 2 | 32 | 33 | 0 | 0 |
| high | Ours-Balanced vs Temporal union | 33 | 9 | 33 | 33 | 0 | 0 |
| high | Ours-Balanced vs Area-matched dilation for Balanced | 33 | 29 | 10 | 16 | 30 | 8 |
| high | Ours-Balanced vs Area-matched distance-only for Balanced | 33 | 22 | 8 | 0 | 24 | 5 |
