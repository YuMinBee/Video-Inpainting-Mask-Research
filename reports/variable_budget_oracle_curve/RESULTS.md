# Variable-Budget Oracle Curve

## 한 줄 결론

현재 1× 고정예산은 병목이었다. 예산을 1.5×·2×로 늘리거나 프레임별
필요량을 쓰면 두 backend에서 큰 추가 이득이 나왔고, 결과 전에 고정한
variable-budget 학습 기준을 **세 후보 모두 통과했다**.

이 결과의 mask는 GT를 사용하는 oracle이다. 신규 method 성능이 아니라
학습 실험을 시작할 충분한 상한이 있다는 진단 결과다.

## 사전등록 범위

- Mask curve: 30클립 / 1,200프레임
- Downstream: 고정된 첫 10클립
- Backends: ProPainter, E2FGVI-HQ
- Methods: support-nearest 1.5×, support-nearest 2×,
  adaptive-reachable oracle
- Bootstrap: 20,000회, seed 20260818
- 상세 기준: PREREGISTRATION.md

한 후보가 두 backend 모두에서 dilation 대비 +1.00 dB, 1× support
oracle 대비 +0.50 dB, CI 하한 양수, 7/10 승리, MAE 개선, outside harm
한도를 모두 만족해야 GO로 정했다.

## Mask curve

| 방법 | Mean budget | Missing recovery | Added precision | Remaining missing / GT | False added / GT |
|---|---:|---:|---:|---:|---:|
| 1× support oracle | 1.00× | 0.5718 | 0.9534 | 0.2111 | 0.0067 |
| 1.5× support oracle | 1.50× | 0.7219 | 0.8740 | 0.1496 | 0.0278 |
| 2× support oracle | 2.00× | 0.8125 | 0.7884 | 0.1073 | 0.0688 |
| Adaptive reachable | 1.88× mean | 0.8898 | 1.0000 | 0.0663 | 0.0000 |

고정 배율 2,400개 mask의 예산 불일치는 0이었다. Adaptive는 30px
candidate band 안의 실제 누락 support만 포함하기 때문에 false filler가
없다.

## Downstream absolute results

| Method | ProPainter PSNR | E2FGVI-HQ PSNR | ProPainter MAE | E2FGVI MAE |
|---|---:|---:|---:|---:|
| Dilation 1× | 10.516 | 11.246 | 0.23808 | 0.22078 |
| Support oracle 1× | 11.068 | 11.923 | 0.22741 | 0.20584 |
| Support oracle 1.5× | 12.490 | 12.947 | 0.19502 | 0.18123 |
| Support oracle 2× | 13.320 | 13.433 | 0.17113 | 0.16614 |
| Adaptive reachable | 12.943 | 14.213 | 0.17256 | 0.14837 |

## Preregistered paired decision

| Backend | Candidate | PSNR vs dilation | PSNR vs 1× | 95% CI vs 1× | Wins | MAE delta | Outside vs dilation | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ProPainter | 1.5× | +1.974 | +1.423 | [+0.670, +2.226] | 9/10 | -0.03239 | -0.00019 | PASS |
| E2FGVI-HQ | 1.5× | +1.701 | +1.024 | [+0.569, +1.502] | 10/10 | -0.02461 | -0.00012 | PASS |
| ProPainter | 2× | +2.803 | +2.252 | [+1.467, +3.031] | 10/10 | -0.05629 | -0.00017 | PASS |
| E2FGVI-HQ | 2× | +2.187 | +1.510 | [+0.839, +2.430] | 10/10 | -0.03970 | +0.00006 | PASS |
| ProPainter | Adaptive | +2.427 | +1.876 | [+0.936, +2.828] | 9/10 | -0.05485 | -0.00023 | PASS |
| E2FGVI-HQ | Adaptive | +2.967 | +2.290 | [+1.121, +3.598] | 9/10 | -0.05748 | -0.00024 | PASS |

Overall decision: **GO**. Fixed-scale and adaptive-budget training pilots are
both authorized on a new held-out protocol.

## 중요한 해석

1. 기존 1× NO-GO는 픽셀 ranking 아이디어 전체의 실패가 아니라 예산이
   너무 작은 operating point의 실패였다.
2. ProPainter에서는 2×가 adaptive보다 0.377 dB 높고, E2FGVI에서는
   adaptive가 2×보다 0.780 dB 높다. 최적 budget은 backend 의존적이다.
3. 따라서 목표는 GT mask 복원만이 아니라 budget-quality-background
   Pareto를 조절하는 removal-support prediction이어야 한다.
4. 같은 면적의 큰 dilation과 비교하지 않으면 면적 증가 효과를 method
   효과로 오인할 수 있다. 이 통제는 별도 Experiment 13에서 수행했다.

## 다음 학습 가설

학습기는 30px band에서 두 출력을 함께 예측한다.

1. Residual-support logit: 어떤 누락 픽셀을 추가할지
2. Budget ratio: 프레임·클립에 얼마만큼 추가할지

초기 loss 후보:

- missed-support BCE/Dice
- log budget-ratio Smooth L1
- 인접 프레임 logit warp consistency
- mask fragmentation을 막는 spatial TV/connectivity regularization
- 여러 inpainting backend의 utility를 이용한 후속 calibration

Experiment 10/11/12의 downstream 10클립을 학습이나 threshold 선택에
재사용하면 안 된다. 별도 train/validation clips와 새 test split이
필수다.

## 산출물

- 설정: configs/variable_budget_oracle_curve.json
- Mask 생성: scripts/build_variable_budget_oracle_masks.py
- Paired 판정: scripts/analyze_variable_budget_curve.py
- 대용량 결과:
  D:/DCG-TR_experiment_results/experiments/12_variable_budget_oracle_curve
