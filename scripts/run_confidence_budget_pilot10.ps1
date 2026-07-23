$ErrorActionPreference = "Stop"

$clips = @(
  "I-210618_I01001_W01",
  "I-210618_I01006_W04",
  "I-210627_O01004_W04",
  "I-210627_O04018_W05",
  "I-210714_O01002_T04",
  "I-210714_O01002_W02",
  "I-210714_O01003_T03",
  "I-210715_I03012_W06",
  "I-210715_I06019_T02",
  "I-210715_I09026_W02"
)

$variants = @(
  "conf_budget_occ_b1",
  "conf_budget_a0p5_s10_b1",
  "conf_budget_a0p5_s20_b1",
  "conf_budget_a0p5_s30_b1",
  "conf_budget_a1_s10_b1",
  "conf_budget_a1_s20_b1",
  "conf_budget_a1_s30_b1",
  "conf_budget_a2_s10_b1",
  "conf_budget_a2_s20_b1",
  "conf_budget_a2_s30_b1"
)

$probeFrames = "D:\DCG-TR_experiment_backup_20260627\workspace\experiments\aihub_subset_100_probe"
$maskRoot = "D:\DCG-TR_experiment_results\experiments\4090_rerun\aihub_subset_100\confidence_budget_pilot10\masks"
$resultRoot = "D:\DCG-TR_experiment_results\results\4090_rerun\aihub_subset_100\confidence_budget_pilot10"
$stageRoot = "D:\DCG-TR_experiment_results\tmp\propainter_confidence_budget_pilot10_stage"
$logRoot = "D:\DCG-TR_experiment_results\experiments\4090_rerun\aihub_subset_100\confidence_budget_pilot10\logs"
$statusPath = Join-Path $logRoot "run_status.csv"

New-Item -ItemType Directory -Force -Path $resultRoot, $stageRoot, $logRoot | Out-Null
"timestamp,variant,status" | Set-Content -LiteralPath $statusPath -Encoding UTF8

foreach ($variant in $variants) {
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$stamp,$variant,start" | Add-Content -LiteralPath $statusPath -Encoding UTF8
  Write-Host "==== Running $variant ===="

  python scripts\run_propainter_batch.py `
    --probe-root (Join-Path $maskRoot $variant) `
    --frames-root $probeFrames `
    --raw-output-root (Join-Path $resultRoot "$variant\propainter_raw") `
    --output-root (Join-Path $resultRoot "$variant\propainter_outputs") `
    --clips $clips `
    --conda-env ai `
    --max-clips 10 `
    --gpu 0 `
    --subvideo-length 40 `
    --neighbor-length 10 `
    --ref-stride 10 `
    --raft-iter 20 `
    --mask-dilation 4 `
    --ascii-stage-root (Join-Path $stageRoot $variant)
  if ($LASTEXITCODE -ne 0) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp,$variant,failed" | Add-Content -LiteralPath $statusPath -Encoding UTF8
    throw "ProPainter failed for $variant with exit code $LASTEXITCODE"
  }

  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$stamp,$variant,done" | Add-Content -LiteralPath $statusPath -Encoding UTF8
}



