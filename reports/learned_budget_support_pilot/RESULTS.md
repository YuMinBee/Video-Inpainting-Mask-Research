# Experiment 14: Learned Budget-Support Pilot

## Decision

**VALIDATION STOP — the sealed test was not generated or opened.**

The learned spatial score became useful when combined with a geometric
distance prior, but the learned budget head missed the frozen validation
accuracy requirement. Under the validation-only stopping rule, no model lock,
sealed-test evaluation, or downstream inpainting run is authorized.

## Setup

- Train: 25 source groups, two independent regenerations, 1,600 frame/defect
  samples.
- Validation: 10 disjoint source groups, 320 samples.
- Sealed test: 30 further source groups fixed in advance but never generated.
- Inputs: RGB, raw-mask geometry, and frozen bidirectional SAM2 ranking features.
- Model: 121,570-parameter U-Net with pixel-support and budget heads.
- Training: best epoch 3, early stop at epoch 9, 67.8 seconds on RTX 4090.
- Checkpoint SHA-256:
  `cea71d02ef4ba57b2fb37834c29fdb21b2e18f5b12af80226781a7588b79fa6f`.

## First validation diagnostic

The uncalibrated learned selector beat frozen SAM2-mean ranking but not the
strong exact predicted-area distance baseline.

| Comparison | Recovery delta |
|---|---:|
| Learned frontier vs exact-area distance | `-0.0083` |
| Learned frontier vs frozen SAM2 mean | `+0.1540` |
| Budget log-ratio MAE | `0.5998` |

This motivated one explicitly recorded validation-only calibration. Before
running it, [CALIBRATION_PLAN.md](CALIBRATION_PLAN.md) froze a scalar median
budget bias and four distance-prior weights. The sealed test remained absent.

## Frozen calibration sweep

The scalar budget log bias was `-0.300969`. Every spatial comparison below
uses exactly the same calibrated predicted area for the learned frontier and
its distance/SAM2 controls.

| Distance-prior weight | Recovery vs distance | 95% CI | Wins | Precision delta | Recovery vs SAM2 | Wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | `-0.0686` | `[-0.1670, +0.0156]` | 5/10 | `+0.0022` | `+0.1073` | 8/10 |
| 0.25 | `+0.0003` | `[-0.0614, +0.0512]` | 6/10 | `+0.0466` | `+0.1763` | 9/10 |
| **0.50** | **`+0.0370`** | **`[+0.0117, +0.0597]`** | **9/10** | **`+0.0599`** | **`+0.2129`** | **9/10** |
| 0.75 | `+0.0286` | `[+0.0198, +0.0379]` | 10/10 | `+0.0374` | `+0.2046` | 8/10 |

The frozen hierarchy selected weight `0.50`. It passed all four spatial
validation checks: positive mean improvement and at least 6/10 wins against
both exact-area distance and frozen SAM2 mean. Its improvement over distance
also had a positive clip-bootstrap CI.

## Blocking failure

Median bias calibration reduced budget log-ratio MAE only from `0.5998` to
`0.5599`; the frozen authorization threshold was `<= 0.45`. This check is
independent of the spatial mixture and therefore failed for every candidate.

The defensible interpretation is:

1. **Spatial signal:** promising on validation only. Learned temporal evidence
   complements, but does not replace, the strong distance prior.
2. **Budget signal:** not accurate enough. The current pooled regression head
   is the bottleneck.
3. **Method claim:** unsupported. No sealed-test or downstream result exists,
   so the positive validation number cannot be presented as generalization.

## Artifact isolation

Large outputs are stored under
`D:/DCG-TR_experiment_results/experiments/14_learned_budget_support_pilot`.
At the stopping decision, `model_lock.json`, `data/sealed_test`, and
`feature_cache/test_manifest.csv` were all absent. Experiments 10--13 and the
old failed LaTeX manuscript were not modified or used for model selection.
