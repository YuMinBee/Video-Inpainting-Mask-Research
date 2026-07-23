# Synthetic-GT Evaluation

Clips: `synthetic_001_I-210618_I01001_W01, synthetic_002_I-210618_I01006_W04, synthetic_003_I-210627_O01004_W04, synthetic_004_I-210627_O04018_W05, synthetic_005_I-210714_O01002_T04, synthetic_006_I-210714_O01002_W02, synthetic_007_I-210714_O01003_T03, synthetic_008_I-210715_I03012_W06, synthetic_009_I-210715_I06019_T02, synthetic_010_I-210715_I09026_W02`

| Method | mPSNR ↑ | mSSIM ↑ | mMAE ↓ | Outside MAE ↓ | BTE ↓ | Extra ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Boundary-only | 21.714 | 0.7482 | 0.04938 | 0.01967 | 0.017225 | 0.1347 |
| Temporal union | 14.628 | 0.4217 | 0.13908 | 0.03414 | 0.013165 | 1.2694 |
| Ours-Balanced | 20.163 | 0.6996 | 0.06064 | 0.02090 | 0.013357 | 0.4915 |
| Area-matched dilation | 21.232 | 0.7372 | 0.05314 | 0.02043 | 0.012811 | 0.4921 |
| Area-matched distance-only | 20.032 | 0.6965 | 0.06178 | 0.02094 | 0.013009 | 0.4915 |
