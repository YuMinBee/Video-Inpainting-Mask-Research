# Project Structure

This repository is organized so that raw data, third-party code, model outputs,
and experiment analysis do not get mixed together.

## Top-Level Layout

```text
dataset/
  Raw and sample datasets. Do not write experiment outputs here.

external/
  Third-party repositories used as executable dependencies.
  Current contents:
    ProPainter/
    Depth-Anything-3/

scripts/
  Our lightweight pipeline and analysis scripts.

results/
  Model outputs and reusable intermediate artifacts.

experiments/
  Experiment-specific visualizations, metrics, summaries, and figures.

configs/
  Shared experiment config files. Keep command presets here when they become stable.

docs/
  Notes, plans, and paper-facing summaries.
  Current research log:
    mask_removal_research_log.md

45EDC5BA43BD4CCC93F1F737EEA150AF/
  KAIA LaTeX template and paper draft files.
```

## Keep `external/` As-Is

Leave `external/` as a vendor-code area. It is better not to mix our scripts or
outputs into the third-party repositories, except for unavoidable model caches
such as `external/ProPainter/weights/`.

Recommended rule:

```text
external/ = code we did not author
scripts/  = code we author
results/  = model outputs
experiments/ = analysis of outputs
```

This makes it easy to update or replace ProPainter/DA3 later without losing our
own experiment logic.

## Current Experiment Folders
The active research direction moved from depth-based refinement to mask quality for object removal. The most up-to-date narrative is:

```text
docs/mask_removal_research_log.md
```

Large synthetic-GT, SAM, ProPainter, and AI-Hub analysis outputs are kept on the D drive rather than inside this repository. The scripts that reproduce or summarize those outputs live in `scripts/`.

```text
experiments/depth_temporal_probe/
  First smoke-test probe.
  Uses OpenCV Telea baseline and proxy depth unless external outputs are passed.

experiments/depth_temporal_probe_real/
  Real probe outputs using ProPainter baseline and DA3 depth.
  Includes depth edge maps, temporal error maps, visual panels, and overlap analysis.

experiments/depth_temporal_probe_real/overlap_analysis/
  Clean overlap report:
    RESULTS.md
    frame_metrics.csv
    clip_summary.csv
    summary.json
    maps/

experiments/02_boundary_refinement/
  Reserved for the next main experiment:
    depth-aware boundary refinement and ablations.
```

## Current Result Folders

```text
results/propainter_outputs/
  Normalized ProPainter frame outputs.
  These are the baseline frames to use in later scripts.

results/propainter_raw/
  Raw ProPainter output layout.
  Useful for debugging, but not the canonical baseline location.

results/da3_depths/
  DA3 depth outputs as `.png` and `.npy`.

results/raft_flows/
  Reserved for RAFT forward/backward flow outputs.

results/refined_outputs/
  Reserved for depth-aware temporal refinement outputs.

results/metrics/
  Reserved for global metric CSV/JSON files.

results/figures/
  Reserved for paper-ready figures copied from experiment folders.
```

## Canonical Inputs For Main Experiment

Use these as the default sources:

```text
Frames:
  experiments/depth_temporal_probe/<clip>/frames/

Masks:
  experiments/depth_temporal_probe/<clip>/masks/

ProPainter baseline:
  results/propainter_outputs/<clip>/

DA3 depth:
  results/da3_depths/<clip>/

Overlap analysis:
  experiments/depth_temporal_probe_real/overlap_analysis/
```

## Next Main Experiment

The next stage should write to:

```text
experiments/02_boundary_refinement/
results/raft_flows/
results/refined_outputs/
results/metrics/
```

Do not overwrite the probe folders. Treat them as the baseline observation used
to motivate the paper.



