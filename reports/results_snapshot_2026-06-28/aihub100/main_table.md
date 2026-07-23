# Full 100-Clip Main Table

| Method | Role | BTE | Outside | Extra | ResProxy |
|---|---|---:|---:|---:|---:|
| Boundary-only | Conservative baseline | 0.017147 | 0.061762 | 0.1041 | 0.4154 |
| Temporal union | Aggressive baseline | 0.015304 | 0.276760 | 3.6329 | 0.1465 |
| Ours-Conservative | rb=0 lower-budget ablation | 0.016681 | 0.063104 | 0.1347 | 0.4116 |
| **Ours-Balanced** | **rb=5, rs=30, tau=0.15** | **0.015917** | **0.068664** | **0.3232** | **0.3967** |
| Area-matched dilation (Balanced) | Matches Ours-Balanced Extra | 0.016041 | 0.067745 | 0.3232 | 0.4028 |
| Area-matched distance-only (Balanced) | No occupancy, matched Extra | 0.015933 | 0.068122 | 0.3226 | 0.3998 |
| Distance-only | Occupancy gate removed | 0.016598 | 0.063611 | 0.1999 | 0.4098 |
| Occupancy-only | Distance gate removed | 0.014486 | 0.176602 | 1.6430 | 0.2110 |

Notes:
- Ours-Balanced, Area-matched dilation (Balanced), and Area-matched distance-only (Balanced) are full 100-clip D-drive runs.
- Conservative and ablation baselines are from the hard-ablation full 100-clip summary.
- Lower is better for all four metrics shown here.
