# Research Flow and Evidence Map

## 1. Research question

Video object removal is usually driven by a segmentation mask, but a mask that separates an object from the background is not necessarily large or stable enough to remove the object's complete visual support. This work asks:

> Can temporal occupancy refine an imperfect segmentation mask into a better removal-support mask without causing excessive background modification?

The analysis separates three questions:

1. Does mask expansion improve removal-related metrics?
2. Is temporal occupancy better than simple geometry at the same mask budget?
3. Do proxy-metric improvements predict reconstruction quality when clean ground truth is available?

## 2. Experimental sequence

### Stage A — Initial real-video evaluation

AI-Hub driving clips and DAVIS 2017 validation clips were evaluated with four lower-is-better proxy metrics:

- **BTE**: boundary temporal error;
- **Outside**: change outside the reference mask;
- **Extra**: added-mask area relative to the reference;
- **ResProxy**: residual-object proxy.

The initial variants established a useful trade-off: aggressive temporal union reduces boundary error but over-expands heavily, while boundary-only processing is conservative but leaves more residue.

### Stage B — Budget-matched controls

Area-matched dilation and area-matched distance-only controls were introduced to prevent mask size from being mistaken for method quality. All three methods add almost the same number of pixels, so their differences more directly test spatial selection.

On the 100-clip AI-Hub run:

| Method | BTE | Outside | Extra | ResProxy |
|---|---:|---:|---:|---:|
| Ours-Balanced | 0.015917 | 0.068664 | 0.3232 | 0.3967 |
| Area-matched dilation | 0.016041 | 0.067745 | 0.3232 | 0.4028 |
| Area-matched distance-only | 0.015933 | 0.068122 | 0.3226 | 0.3998 |

Ours-Balanced has the best mean BTE and ResProxy of these three, but the margins are small and Outside is worse. At clip level, it beats area-matched dilation on BTE in 75/100 clips and ResProxy in 93/100, but only 24/100 clips on Outside and ResProxy simultaneously. Against area-matched distance-only, the joint win count is 17/100.

**Interpretation:** the method finds a useful operating point, but the evidence does not isolate a large or consistently dominant benefit from occupancy.

### Stage C — Cross-dataset check

On 30 DAVIS clips, Ours-Balanced lowers BTE from 0.041052 (Boundary-only) to 0.038699, while Outside rises from 0.014502 to 0.030683 and Extra rises from 0.3402 to 0.8370. The same removal-versus-collateral-change trade-off therefore appears outside the driving set.

### Stage D — Synthetic clean-background ground truth

A 10-clip synthetic benchmark was created so that ProPainter output could be compared directly with the clean background. Under jittered SAM masks:

| Method | masked MAE | masked PSNR | BTE | Extra |
|---|---:|---:|---:|---:|
| Ours-Balanced | 0.14421 | 14.446 | 0.030402 | 0.4064 |
| Area-matched dilation | **0.12903** | **15.439** | **0.028230** | 0.3820 |
| Area-matched distance-only | 0.13753 | 14.786 | 0.029393 | 0.3933 |

Area-matched dilation is the lowest-MAE candidate in 6/10 clips; Ours-Balanced is lowest in 0/10. This is the strongest evidence against claiming that temporal occupancy is the main solution in the current setup.

### Stage E — Why dilation is strong

The distance-to-mask diagnostic locates 55.4% of missed SAM object pixels within 10 px of the raw mask and 76.1% within 20 px. The mask-correction test also gives area-matched dilation higher recovery (0.7990) and precision (0.5074) than Ours-Balanced (0.7361 and 0.4596).

**Interpretation:** the benchmark is dominated by local boundary under-coverage. A local geometric expansion is well matched to that error.

### Stage F — SAM prompt-ensemble uncertainty

Probability and uncertainty from jittered SAM box prompts were evaluated using the same added-pixel budget as dilation. Both recover only about 0.283 of the missing region, versus 0.799 for dilation, and add more false background.

**Interpretation:** box-jitter disagreement does not identify missing removal-support pixels reliably in this setting.

## 3. Supported and unsupported claims

### Supported

- Imperfect segmentation masks measurably degrade video object removal.
- Removal quality depends on a trade-off between missing support and collateral background modification.
- Simple, budget-matched geometric baselines are essential controls.
- The current synthetic-SAM error distribution is mostly local to the mask boundary.
- Temporal occupancy and box-jitter SAM uncertainty are insufficient as standalone missing-support cues in the tested setting.

### Not supported by the current evidence

- Temporal occupancy consistently outperforms area-matched dilation.
- Better residue proxies necessarily imply better clean-background reconstruction.
- Temporal spread predicts the gain over geometry: the measured Pearson and Spearman correlations are close to zero.
- SAM prompt uncertainty is a reliable removal-mask confidence signal.

## 4. Main limitations

- Synthetic-GT contains 10 clips and its SAM errors are biased toward local boundary shrinkage.
- Real datasets lack clean background ground truth, so their removal quality is assessed using proxies.
- Some metrics reward opposite behaviors; no single scalar captures the full removal/collateral-change trade-off.
- The study evaluates preprocessing around ProPainter, not a learned removal-support predictor.

## 5. Recommended next experiment

The next study should construct or collect failures with genuine nonlocal and temporally intermittent under-coverage, then compare:

1. area-matched dilation;
2. distance-only selection;
3. temporal occupancy;
4. a cue that distinguishes object support from background, such as appearance/depth/flow consistency or a learned support predictor.

Evaluation should report both clean-background reconstruction and budget-matched mask correction, with pre-registered primary metrics. This would test the intended temporal failure mode instead of a boundary-dilation regime.

## 6. Evidence locations

- Curated, Git-sized tables: `reports/results_snapshot_2026-06-28/`
- Full experimental narrative: `docs/mask_removal_research_log.md`
- Commands and expected values: `reproducibility/README.md`
- Large outputs: `D:\DCG-TR_experiment_results`
- Pre-reorganization backup: `D:\DCG-TR_experiment_backup_20260627`
