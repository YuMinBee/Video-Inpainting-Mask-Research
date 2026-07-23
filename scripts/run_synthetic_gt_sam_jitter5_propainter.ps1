$ErrorActionPreference = "Stop"

$probeRoot = "D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_sam_jitter5_raw"
$maskRoot = "D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_sam_jitter5_masks"
$resultRoot = "D:\DCG-TR_experiment_results\results\synthetic_gt_probe_sam_jitter5"
$stageRoot = "D:\DCG-TR_experiment_results\tmp\propainter_synthetic_gt_sam_jitter5_stage"
$logRoot = "D:\DCG-TR_experiment_results\experiments\synthetic_gt_probe_sam_jitter5_logs"
$statusPath = Join-Path $logRoot "run_status.csv"

$clips = Get-ChildItem -LiteralPath $probeRoot -Directory | ForEach-Object { $_.Name }
$variants = @(
  "boundary_only",
  "temporal_union",
  "ours_balanced",
  "area_matched_dilation",
  "area_matched_distance_only"
)

New-Item -ItemType Directory -Force -Path $resultRoot, $stageRoot, $logRoot | Out-Null
"timestamp,variant,status" | Set-Content -LiteralPath $statusPath -Encoding UTF8

foreach ($variant in $variants) {
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$stamp,$variant,start" | Add-Content -LiteralPath $statusPath -Encoding UTF8
  Write-Host "==== Running $variant ===="

  python scripts\run_propainter_batch.py `
    --probe-root (Join-Path $maskRoot $variant) `
    --frames-root $probeRoot `
    --raw-output-root (Join-Path $resultRoot "$variant\propainter_raw") `
    --output-root (Join-Path $resultRoot "$variant\propainter_outputs") `
    --clips $clips `
    --max-clips 10 `
    --gpu 0 `
    --subvideo-length 40 `
    --neighbor-length 10 `
    --ref-stride 10 `
    --raft-iter 20 `
    --mask-dilation 4 `
    --ascii-stage-root (Join-Path $stageRoot $variant) `
    --conda-env ai
  if ($LASTEXITCODE -ne 0) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp,$variant,failed" | Add-Content -LiteralPath $statusPath -Encoding UTF8
    throw "ProPainter failed for $variant with exit code $LASTEXITCODE"
  }

  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$stamp,$variant,done" | Add-Content -LiteralPath $statusPath -Encoding UTF8
}

