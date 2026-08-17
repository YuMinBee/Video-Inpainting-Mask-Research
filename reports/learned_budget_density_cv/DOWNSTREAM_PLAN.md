# Experiment 15 Conditional Downstream Plan

Frozen at 2026-08-15T15:58:59+09:00 after the locked 30-source mask gate
returned GO and before any Experiment 15 inpainting output existed.

## Comparison

The selected `density_absolute` hybrid-frontier mask is compared only with
exact-area distance expansion. They contain exactly the same number of added
pixels on every frame. Both registered defect settings (5% and 10% box jitter)
are evaluated on all 30 sealed source clips and 16 frames per clip.

Frame metrics are averaged within setting and source, then the two settings are
averaged within source. Bootstrap resampling and win counts use the resulting
30 paired source values.

## Frozen backends

- Official ProPainter: FP16, subvideo length 40, neighbor length 10, reference
  stride 10, RAFT iterations 20, and no backend mask dilation.
- Official E2FGVI-HQ: FP32, neighbor stride 5, reference step 10, all available
  references, and no backend mask dilation.

Clean synthetic background frames are used only for offline PSNR/SSIM/MAE and
temporal-error evaluation.

## Joint gate

Each backend must independently satisfy all of the following against exact-area
distance:

1. masked-PSNR gain at least +0.25 dB;
2. 95% source-bootstrap CI lower bound above zero;
3. at least 18/30 source wins;
4. negative masked-MAE delta; and
5. outside-GT MAE delta no greater than +0.001.

Both backends must pass for the full learned-method downstream claim. The mask
result remains valid even if this downstream gate fails.
