# Single-Component 30-Clip Replication: Results

## Outcome

The fixed-budget SAM2 mask-ranking effect replicated, but the pre-registered
downstream claim did not.

- Mask-level correction: **SUPPORTED**.
- ProPainter `+0.50 dB` replication: **NOT SUPPORTED**.
- E2FGVI-HQ `+0.50 dB` replication: **NOT SUPPORTED**.
- Backend-independent downstream claim: **NOT SUPPORTED**.

This is not a zero-effect result. Both backends have a positive clip-bootstrap
PSNR confidence interval, but the effects are smaller and less consistent than
the registered minimum effect.

## Integrity and controlled defect

- New, non-overlapping backgrounds: 30 clips / 1,200 frames.
- Resolution: 960x540, 40 frames per clip.
- Every donor and all 1,200 generated GT masks: exactly one 8-connected
  component.
- Raw masks: SAM ViT-B with independently jittered 5% boxes.
- Raw SAM IoU: `0.6150`.
- Missing mask area / GT: `0.3714`.
- Over-mask area / GT: `0.0330`.
- Prediction / GT area: `0.6616`.

The new data therefore represents predominantly boundary under-coverage rather
than the disconnected-object over-mask failure found in experiment 09.

## Fixed-budget mask result

Both corrections add exactly the same number of pixels on every frame.

| Method | Missing recovery | Added precision | False added / GT |
|---|---:|---:|---:|
| Radius-5 dilation | 0.4462 | 0.7131 | 0.0487 |
| SAM2 bidirectional mean | 0.4809 | 0.8097 | 0.0328 |

The paired clip recovery delta is `+0.0347`, 95% CI
`[+0.0156, +0.0535]`, with `23/30` clip wins. All 1,200 frame budgets match
exactly. The registered mask-level gate passes.

## Downstream result

The table reports SAM2-ranked correction minus equal-budget dilation.

| Backend | Masked PSNR delta | 95% clip CI | Wins | Masked MAE delta | Outside-MAE delta | Registered decision |
|---|---:|---:|---:|---:|---:|---|
| ProPainter | +0.366 dB | [+0.095, +0.687] | 18/30 | -0.00678 | +0.000022 | **NOT SUPPORTED** |
| E2FGVI-HQ | +0.321 dB | [+0.010, +0.727] | 17/30 | -0.00329 | -0.000054 | **NOT SUPPORTED** |

ProPainter fails only the registered `+0.50 dB` minimum magnitude. E2FGVI-HQ
fails that magnitude and the `18/30` win requirement. The outside-GT harm limit
passes for both backends.

## Robustness and failure analysis

- ProPainter median / 10% trimmed mean: `+0.172 / +0.227 dB`.
- E2FGVI-HQ median / 10% trimmed mean: `+0.097 / +0.132 dB`.
- Cross-backend clip-effect correlation: Pearson `r=0.862`,
  `p=9.83e-10`; Spearman `rho=0.635`, `p=0.000163`.
- Both backends improve on 15 clips and both worsen on 10 clips.
- Seven clips improve mask recovery but lose PSNR on both backends.
- Four clips lose mask recovery but improve PSNR on both backends.
- Recovery delta is only weakly related to downstream PSNR:
  `r=0.240` for ProPainter and `r=0.268` for E2FGVI.

Thus, the two inpainting models respond similarly to the corrected masks, but
aggregate missing-area recovery is not a sufficient per-clip utility surrogate.
Where the fixed pixel budget is placed along a high-contrast or motion-sensitive
boundary matters more than recovered area alone.

Representative cases:

- Clear success: `synthetic_027_I-211105_I06019_T06` has mask recovery
  `+0.149`, ProPainter `+3.176 dB`, and E2FGVI `+4.672 dB`. Its selected frame
  improves by `+8.06 / +8.91 dB`.
- Clear failure despite better recovery:
  `synthetic_022_I-211026_I08022_W07` has mask recovery `+0.038`, but
  ProPainter loses `0.635 dB` and E2FGVI loses `0.887 dB`. The selected frame
  loses `1.80 / 1.67 dB` because the equal budget is redistributed away from
  consequential residual boundary regions.

Panels and exact selected-frame values are stored in `cases/`.

## Why experiment 09 looked stronger

The post-hoc audit of experiment 09 found 23 single-component and seven
multi-component clips.

| Experiment-09 group | Clips | Raw IoU | Raw over-mask / GT | Pred / GT area | ProPainter PSNR | E2FGVI PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Single component | 23 | 0.664 | 0.043 | 0.730 | +0.273 dB | -0.091 dB |
| Multiple components | 7 | 0.212 | 2.306 | 2.633 | +1.431 dB | +1.100 dB |
| New single-component replication | 30 | 0.615 | 0.033 | 0.662 | +0.366 dB | +0.321 dB |

The seven disconnected-object cases contain a qualitatively different and much
more severe SAM box-prompt failure and disproportionately large downstream
gains. This explains why the original `+0.543 dB` ProPainter mean was too
optimistic for ordinary boundary under-coverage.

This is an explanatory audit, not a causal exclusion: experiment 10 also uses
new backgrounds and donors, and between-experiment effect-difference confidence
intervals include zero. The honest conclusion is that the large effect did not
survive the stricter composition control, while a smaller positive effect did.

## Research decision

Current evidence supports a project contribution and a careful negative-result
or diagnostic paper story:

1. fixed-budget SAM2 ranking reliably recovers more missing boundary pixels;
2. that mask-level advantage transfers as a modest average inpainting gain on
   two backends;
3. a strong `>=+0.50 dB` downstream claim does not replicate; and
4. synthetic disconnected-component prompts can substantially inflate the
   apparent benefit.

It does not yet support a strong standalone method paper. A stronger paper would
need a real-video boundary benchmark and a pre-registered utility-aware selector
that predicts when SAM2 redistribution helps, evaluated on a separate test set.

## Artifacts

- Protocol: `PREREGISTRATION.md`
- Mask result: `mask_level/RESULTS.md`
- Backend analyses: `propainter/PAIRED_RESULTS.md`,
  `e2fgvi_hq/PAIRED_RESULTS.md`
- Diagnostics: `diagnostics/POSTHOC_ANALYSIS.md`
- Experiment comparison: `replication_comparison/REPLICATION_COMPARISON.md`
- Full-resolution outputs:
  `D:/DCG-TR_experiment_results/experiments/10_single_component_confirmatory_30`
