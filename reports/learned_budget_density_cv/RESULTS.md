# Experiment 15: Learned Budget-Density Temporal Mask Correction

## Decision

**Overall status: GO.**

The preregistered density-based budget model passed source-disjoint 5-fold
cross-validation, was locked before sealed-test generation, passed the
30-source sealed mask gate, and passed the downstream gate on both ProPainter
and E2FGVI-HQ.

This is the first experiment in the project that supports an observable-input,
trained mask-correction method on a held-out test. It does not yet establish a
paper-level state of the art because the sealed test is synthetic and a recent
competitive refinement/robust-removal baseline has not been run.

## What changed after Experiment 14

Experiment 14 predicted one correction-budget scalar from pooled features. Its
validation log-ratio MAE was `0.5599`, so the test stayed sealed.

Experiment 15 replaced that bottleneck with a dense residual-support map. The
same per-pixel predictions serve two coupled roles:

1. summing the density estimates how many pixels should be added; and
2. combining the learned rank with a distance rank chooses where those pixels
   should be added through a connectivity-aware frontier.

The final input has 13 observable channels: RGB, raw-mask geometry, and frozen
SAM2 forward/backward rank, probability, and disagreement signals. No clean
background, GT mask, or inpainting output is used at inference.

## Protocol and isolation

- Development: 35 source groups, 1,920 source/setting/frame samples.
- Selection: five source-group folds, seven held-out sources per fold.
- Training schedule: fixed eight epochs; no held-out-fold early stopping or
  calibration.
- Sealed test: 30 untouched source groups, 16 frames, two SAM box-jitter
  severities, 960 samples.
- Test data and test features were absent when the model was locked.
- Final model: `BudgetDensityUNet`, 120,050 parameters.
- Locked checkpoint SHA-256:
  `5ffc11de19342cc58f20da1c33a580a0bae290667777079e66dacc7befdc07ef`.
- Large-output root:
  `D:/DCG-TR_experiment_results/experiments/15_learned_budget_density_cv`.

The preregistration, protocol clarification, and downstream plan were written
before their corresponding outputs:

- `PREREGISTRATION.md`
- `PROTOCOL_CLARIFICATION.md`
- `DOWNSTREAM_PLAN.md`

## 1. Source-disjoint OOF budget gate

| Candidate | Log-budget MAE | Spearman | Prediction std / target std | Gate |
|---|---:|---:|---:|---|
| Scalar, rank channels | 0.5867 | 0.1706 | 0.9812 | FAIL |
| Scalar, absolute channels | 0.4952 | 0.2328 | 0.7394 | FAIL |
| Ordinal, absolute channels | 0.5198 | 0.2420 | 0.7914 | FAIL |
| **Density, absolute channels** | **0.3300** | **0.6919** | **0.6519** | **PASS** |
| Density + temporal smoothing | 0.3547 | 0.6728 | 0.6516 | PASS |

Frozen thresholds were MAE `<=0.45`, Spearman `>=0.30`, and prediction
standard deviation at least 25% of the target standard deviation. The result
shows that the correction amount was observable; the pooled scalar head in
Experiment 14 was the main bottleneck.

## 2. Full-resolution OOF mask gate

The unsmoothed density model was selected by the frozen hierarchy.

| Reference at the exact predicted area | Recovery delta | 95% CI | Source wins | Precision delta |
|---|---:|---:|---:|---:|
| Distance frontier | **+0.0754** | **[+0.0634, +0.0870]** | **35/35** | **+0.0858** |
| Frozen SAM2 mean rank | **+0.1465** | **[+0.0994, +0.2025]** | **34/35** | **+0.1101** |

Every comparison uses the model's predicted added-pixel count on that frame.
The gain therefore cannot be explained by a larger mask than the distance
baseline.

## 3. One-shot sealed mask test

| Reference at the exact predicted area | Recovery delta | 95% CI | Source wins | Precision delta |
|---|---:|---:|---:|---:|
| Distance frontier | **+0.1020** | **[+0.0830, +0.1215]** | **29/30** | **+0.1020** |
| Frozen SAM2 mean rank | **+0.2691** | **[+0.1949, +0.3475]** | **29/30** | **+0.1935** |

The sealed budget log-ratio MAE was `0.3509`; all mask checks passed. All 960
candidate/reference frame pairs matched their intended exact-area budgets.

## 4. Downstream inpainting gate

The two jitter severities were first averaged within each source. Paired
bootstrap statistics then used the 30 source clips as independent units.

| Backend | Masked-PSNR delta | 95% CI | Wins | Masked-MAE delta | Outside-MAE delta | Gate |
|---|---:|---:|---:|---:|---:|---|
| **ProPainter** | **+1.205 dB** | **[+0.400, +2.335]** | **26/30** | **-0.022997** | **-0.000084** | **PASS** |
| **E2FGVI-HQ** | **+1.101 dB** | **[+0.339, +2.215]** | **23/30** | **-0.020189** | **-0.000120** | **PASS** |

The frozen backend gate required at least `+0.25 dB`, a positive CI lower
bound, at least 18/30 wins, lower masked MAE, and outside-MAE harm no greater
than `0.001`. Both backends passed every check. The result covers 3,840 final
inpainting frames: two mask methods, two severities, 30 clips, 16 frames, and
two backends.

## 5. Robustness and outlier audit

One source, `I-211228_O09041_W09`, produced a very large gain because the
distance frontier spent its equal budget around the exterior while leaving
person-shaped holes inside the mask. The learned frontier filled those holes,
raising PSNR by about 14 dB on both backends.

This case increases the mean, but it does not create the conclusion:

| Post-hoc robust check | ProPainter | E2FGVI-HQ |
|---|---:|---:|
| Mean after removing the single largest gain | +0.753 dB | +0.632 dB |
| 95% bootstrap CI after removal | [+0.297, +1.292] | [+0.253, +1.068] |
| Median source delta | +0.333 dB | +0.261 dB |
| Median bootstrap CI | [+0.178, +0.555] | [+0.124, +0.388] |

These checks were not part of the registered decision gate and are reported
only as an outlier sensitivity analysis.

## 6. Failure cases

- ProPainter lost on 4/30 sources; E2FGVI-HQ lost on 7/30.
- Only one source lost on both backends: `I-211208_O05025_T05`.
- On that common failure, mask recovery still improved by `+0.0508`, but PSNR
  changed by `-1.304 dB` on ProPainter and `-0.604 dB` on E2FGVI-HQ.
- Visual inspection showed that the learned mask covered more residual object
  support, yet both inpainters synthesized a large incorrect brown region.
  Thus better support coverage can expose a harder reconstruction hole.
- Recovery delta had a nonlinear relationship with downstream utility:
  Spearman correlation with the two-backend mean PSNR delta was `0.521`, while
  Pearson correlation was only `0.173`.
- The two backends nevertheless agreed strongly on source difficulty
  (Spearman `0.773`; Pearson `0.985`).

The remaining technical problem is therefore not only boundary detection.
Future training should include an inpainting-utility or temporal-output term so
that the allocator learns which correct support pixels are also reconstructable
and temporally stable.

## Supported claim

> A small density-based head can jointly predict variable correction amount and
> temporally informed support placement from observable RGB, mask geometry, and
> frozen SAM2 evidence. On a locked 30-source synthetic test, it improves both
> exact-area mask recovery and downstream removal quality over a distance-only
> frontier on ProPainter and E2FGVI-HQ.

## Claim boundary

This result upgrades the work from a diagnostic-only project to a validated
method prototype. A strong paper claim still needs:

1. a recent competitive imperfect-mask/refinement baseline under a fair
   interface;
2. naturally occurring imperfect masks or an external real-video defect set;
3. ablations of absolute SAM2 evidence, the density budget, the distance
   mixture, and temporal/connectivity constraints; and
4. failure-aware supervision aligned with downstream reconstruction utility.

## Compact result files

- `oof_budget_summary.csv`
- `oof_mask_comparisons.csv`
- `sealed_test_comparisons.csv`
- `downstream_paired_metrics.csv`
- `posthoc_robustness.csv`
- `downstream_failure_sources.csv`
