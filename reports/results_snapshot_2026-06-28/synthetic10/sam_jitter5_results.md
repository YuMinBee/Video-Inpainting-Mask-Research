# Synthetic-GT Evaluation

Clips: `synthetic_001_I-210618_I01001_W01, synthetic_002_I-210618_I01006_W04, synthetic_003_I-210627_O01004_W04, synthetic_004_I-210627_O04018_W05, synthetic_005_I-210714_O01002_T04, synthetic_006_I-210714_O01002_W02, synthetic_007_I-210714_O01003_T03, synthetic_008_I-210715_I03012_W06, synthetic_009_I-210715_I06019_T02, synthetic_010_I-210715_I09026_W02`

| Method | mPSNR ??| mSSIM ??| mMAE ??| Outside MAE ??| BTE ??| Extra ??|
|---|---:|---:|---:|---:|---:|---:|
| Boundary-only | 12.911 | 0.4281 | 0.17998 | 0.02032 | 0.038157 | 0.1468 |
| Temporal union | 12.776 | 0.3724 | 0.17591 | 0.03845 | 0.023862 | 1.3207 |
| Ours-Balanced | 14.446 | 0.5010 | 0.14421 | 0.02124 | 0.030402 | 0.4064 |
| Area-matched dilation | 15.439 | 0.5601 | 0.12903 | 0.02067 | 0.028230 | 0.3820 |
| Area-matched distance-only | 14.786 | 0.5232 | 0.13753 | 0.02085 | 0.029393 | 0.3933 |
