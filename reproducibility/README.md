# Reproducibility Guide

This guide documents how to reproduce the selective temporal mask expansion experiments after moving to a new machine.

## 1. What To Copy

For the fastest continuation, copy these directories from the current machine:

```bash
DCG-TR/
  scripts/
  external/
  experiments/
  results/
  dataset/davis/
  dataset/040.Inpainting_자동화를_위한_영상_데이터/
  reproducibility/
```

Approximate current sizes:

```text
dataset      542G
experiments   35G
results       27G
external     351M
```

If storage is limited, keep at least:

```text
external/ProPainter
external/ProPainter/weights
external/Depth-Anything-3
scripts
experiments/selective_expansion/full20_evaluation
experiments/davis2017_val/evaluation
experiments/paper_tables
experiments/paper_visualizations
```

This minimal copy preserves the result tables and visualizations, but it does not allow rerunning ProPainter from raw data unless the datasets and intermediate masks are also copied.

## 2. Environments

The experiment used the following conda environment:

- `convnext`: ProPainter, OpenCV, PyTorch, and experiment scripts.

Environment exports are saved here:

```bash
reproducibility/convnext_env.yml
```

Create them on the new machine:

```bash
conda env create -f reproducibility/convnext_env.yml
```

If the exact CUDA package versions do not solve on the new machine, create a fresh PyTorch environment matching the new CUDA driver, then install the dependencies needed by `external/ProPainter/requirements.txt` and the local scripts. The code was run successfully with two RTX 3090 GPUs.

## 3. External Repositories And Weights

Required local repositories:

```text
external/ProPainter
external/Depth-Anything-3
```

Required ProPainter weights:

```text
external/ProPainter/weights/ProPainter.pth
external/ProPainter/weights/raft-things.pth
external/ProPainter/weights/recurrent_flow_completion.pth
```

Current weight directory size is about `191M`.

## 4. Current Main Outputs

Research tables and visualizations:

```text
experiments/paper_tables/SELECTIVE_EXPANSION_TABLES.md
experiments/paper_visualizations/case_panels/INDEX.md
```

Driving results:

```text
experiments/selective_expansion/full20_evaluation/MAIN_TABLE.md
experiments/selective_expansion/full20_evaluation/WIN_COUNTS.md
experiments/selective_expansion/full20_evaluation/CASE_STUDIES.md
experiments/selective_expansion/full20_evaluation/frame_metrics.csv
experiments/selective_expansion/full20_evaluation/clip_summary.csv
```

DAVIS results:

```text
experiments/davis2017_val/evaluation/RESULTS.md
experiments/davis2017_val/evaluation/WIN_COUNTS.md
experiments/davis2017_val/evaluation/frame_metrics.csv
experiments/davis2017_val/evaluation/clip_summary.csv
```

## 5. Rebuild Tables And Figures From Existing Metrics

If `experiments/` and `results/` are copied, the research assets can be regenerated without rerunning ProPainter:

```bash
python scripts/build_paper_result_tables.py
python scripts/make_paper_case_visualizations.py
```

## 6. Full Reproduction: Driving Clips

Prepare the AI-Hub subset:

```bash
python scripts/prepare_aihub_validation_subset.py \
  --aihub-root "dataset/040.Inpainting_자동화를_위한_영상_데이터" \
  --output-root experiments/aihub_subset_20_real \
  --max-clips 20
```

Build mask variants:

```bash
python scripts/build_temporal_union_masks.py \
  --probe-root experiments/aihub_subset_20_real \
  --output-root experiments/mask_union

python scripts/build_selective_expansion_masks.py \
  --probe-root experiments/aihub_subset_20_real \
  --output-root experiments/selective_expansion/basic \
  --radius 5 \
  --occupancy-threshold 0.4

python scripts/build_aggressive_selective_masks.py \
  --probe-root experiments/aihub_subset_20_real \
  --output-root experiments/selective_expansion/aggressive/pixel_r10_t0p10 \
  --radius 10 \
  --occupancy-threshold 0.10 \
  --mode pixel

python scripts/build_aggressive_selective_masks.py \
  --probe-root experiments/aihub_subset_20_real \
  --output-root experiments/selective_expansion/aggressive/pixel_r10_t0p15 \
  --radius 10 \
  --occupancy-threshold 0.15 \
  --mode pixel
```

Run ProPainter for the five main methods:

```bash
python scripts/run_propainter_batch.py \
  --probe-root experiments/aihub_subset_20_real \
  --raw-output-root results/aihub_subset_20/propainter_raw \
  --output-root results/aihub_subset_20/propainter_outputs \
  --gpu 0 --max-clips 20 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/selective_expansion/basic/boundary_only_mask \
  --raw-output-root results/selective_expansion/basic/boundary_only_mask/propainter_raw \
  --output-root results/selective_expansion/basic/boundary_only_mask/propainter_outputs \
  --gpu 0 --max-clips 20 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/mask_union/temporal_union \
  --raw-output-root results/mask_union/temporal_union/propainter_raw \
  --output-root results/mask_union/temporal_union/propainter_outputs \
  --gpu 0 --max-clips 20 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/selective_expansion/aggressive/pixel_r10_t0p10 \
  --raw-output-root results/selective_expansion/aggressive/pixel_r10_t0p10/propainter_raw \
  --output-root results/selective_expansion/aggressive/pixel_r10_t0p10/propainter_outputs \
  --gpu 0 --max-clips 20 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/selective_expansion/aggressive/pixel_r10_t0p15 \
  --raw-output-root results/selective_expansion/aggressive/pixel_r10_t0p15/propainter_raw \
  --output-root results/selective_expansion/aggressive/pixel_r10_t0p15/propainter_outputs \
  --gpu 0 --max-clips 20 --subvideo-length 40 --fp16 --use-current-python
```

Evaluate:

```bash
python scripts/evaluate_selective_expansion.py \
  --probe-root experiments/aihub_subset_20_real \
  --output-root experiments/selective_expansion/full20_evaluation \
  --methods Original Boundary-only "Temporal union" "Aggressive pixel r10 t0.10" "Aggressive pixel r10 t0.15"

python scripts/build_paper_result_tables.py
python scripts/make_paper_case_visualizations.py
```

## 7. Full Reproduction: DAVIS 2017 Val

Prepare DAVIS:

```bash
python scripts/prepare_davis_probe.py \
  --davis-root dataset/davis/DAVIS \
  --split-file dataset/davis/DAVIS/ImageSets/2017/val.txt \
  --output-root experiments/davis2017_val/probe \
  --max-frames 0
```

Build masks:

```bash
python scripts/build_temporal_union_simple.py \
  --probe-root experiments/davis2017_val/probe \
  --output-root experiments/davis2017_val/masks/temporal_union

python scripts/build_selective_expansion_masks.py \
  --probe-root experiments/davis2017_val/probe \
  --output-root experiments/davis2017_val/masks/basic \
  --radius 5 \
  --occupancy-threshold 0.4

python scripts/build_aggressive_selective_masks.py \
  --probe-root experiments/davis2017_val/probe \
  --output-root experiments/davis2017_val/masks/ours_r10_t0p10 \
  --radius 10 \
  --occupancy-threshold 0.10 \
  --mode pixel

python scripts/build_aggressive_selective_masks.py \
  --probe-root experiments/davis2017_val/probe \
  --output-root experiments/davis2017_val/masks/ours_r10_t0p15 \
  --radius 10 \
  --occupancy-threshold 0.15 \
  --mode pixel
```

Run ProPainter. With two GPUs, split the clips into two lists and run the commands in parallel. The existing scripts support `--clips`.

Example single-GPU commands:

```bash
python scripts/run_propainter_batch.py \
  --probe-root experiments/davis2017_val/probe \
  --raw-output-root results/davis2017_val/original/propainter_raw \
  --output-root results/davis2017_val/original/propainter_outputs \
  --gpu 0 --max-clips 99 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/davis2017_val/masks/basic/boundary_only_mask \
  --raw-output-root results/davis2017_val/boundary_only/propainter_raw \
  --output-root results/davis2017_val/boundary_only/propainter_outputs \
  --gpu 0 --max-clips 99 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/davis2017_val/masks/temporal_union \
  --raw-output-root results/davis2017_val/temporal_union/propainter_raw \
  --output-root results/davis2017_val/temporal_union/propainter_outputs \
  --gpu 0 --max-clips 99 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/davis2017_val/masks/ours_r10_t0p10 \
  --raw-output-root results/davis2017_val/ours_r10_t0p10/propainter_raw \
  --output-root results/davis2017_val/ours_r10_t0p10/propainter_outputs \
  --gpu 0 --max-clips 99 --subvideo-length 40 --fp16 --use-current-python

python scripts/run_propainter_batch.py \
  --probe-root experiments/davis2017_val/masks/ours_r10_t0p15 \
  --raw-output-root results/davis2017_val/ours_r10_t0p15/propainter_raw \
  --output-root results/davis2017_val/ours_r10_t0p15/propainter_outputs \
  --gpu 0 --max-clips 99 --subvideo-length 40 --fp16 --use-current-python
```

Evaluate:

```bash
python scripts/evaluate_davis_selective.py
python scripts/build_paper_result_tables.py
python scripts/make_paper_case_visualizations.py
```

Expected DAVIS frame count for each method:

```text
1999 png frames
```

Check with:

```bash
for m in original boundary_only temporal_union ours_r10_t0p10 ours_r10_t0p15; do
  printf '%s ' "$m"
  find "results/davis2017_val/$m/propainter_outputs" -path '*/[0-9]*.png' | wc -l
done
```

## 8. Expected Key Numbers

Driving clips:

```text
Ours r10 t0.10: residue < boundary-only 19/20, outside < temporal union 19/20
Ours r10 t0.15: residue < boundary-only 19/20, outside < temporal union 20/20
```

DAVIS 2017 val:

```text
Ours r10 t0.10: residue < boundary-only 21/30, outside < temporal union 30/30
Ours r10 t0.15: residue < boundary-only 20/30, outside < temporal union 30/30
```

## 9. Notes

- `run_propainter_batch.py` skips completed clips if normalized output frames already exist, so interrupted runs can be resumed.
- The ProPainter video writer may warn about macro-block resizing. The evaluation uses normalized PNG frames, so this warning is not fatal.
- The evaluation uses Farneback optical flow for temporal-error metrics. It is CPU-bound and can take several minutes for DAVIS.
- The exact AI-Hub subset depends on the deterministic order used by `prepare_aihub_validation_subset.py`. To preserve the exact reported numbers, copy `experiments/aihub_subset_20_real` and the generated masks/results rather than reselecting from raw data.
