# Mask Correction Efficiency

| Method | Missing recovery ↑ | Added precision ↑ | False added ↓ | Extra budget ↓ | Raw missing ↓ |
|---|---:|---:|---:|---:|---:|
| Boundary-only | 0.4919 | 0.7131 | 0.0648 | 0.2253 | 0.4217 |
| Temporal union | 0.8718 | 0.2335 | 1.2388 | 1.6073 | 0.4217 |
| Ours-Balanced | 0.7361 | 0.4596 | 0.3244 | 0.6045 | 0.4217 |
| Area-matched dilation | 0.7990 | 0.5074 | 0.3000 | 0.6031 | 0.4217 |
| Area-matched distance-only | 0.7579 | 0.4850 | 0.3113 | 0.6045 | 0.4217 |

Definitions:

- `H = M_gt \ M_raw` is the missed GT object region.
- `A = M_v \ M_raw` is the area newly added by a method.
- `Missing recovery = |A ∩ H| / |H|`.
- `Added precision = |A ∩ H| / |A|`.
- `False added = |A \ M_gt| / |M_gt|`.
- `Extra budget = |A| / |M_gt|`.
