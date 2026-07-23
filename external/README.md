# External Dependencies

This directory stores third-party repositories used by the experiment pipeline.

Current repositories:

```text
ProPainter/
Depth-Anything-3/
```

Policy:

- Keep third-party code here.
- Keep our scripts in `scripts/`.
- Keep generated outputs in `results/` or `experiments/`.
- Avoid editing files inside third-party repositories unless a local compatibility patch is absolutely necessary.

Known runtime environments:

```text
ProPainter:
  conda env: convnext
  GPU command prefix: CUDA_VISIBLE_DEVICES=1 conda run -n convnext ...

Depth Anything 3:
  conda env: yolov7
  GPU command prefix: CUDA_VISIBLE_DEVICES=1 conda run -n yolov7 ...
```
