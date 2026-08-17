# Same-Budget Distance Control

## 한 줄 결론

예산 증가만으로도 성능은 좋아지지만 충분하지 않다. 동일한 1.5×·2×
면적에서 GT-assisted support oracle이 순수 distance dilation보다 두
backend 모두 크게 높아, learned spatial selector 기준도 **GO**를
통과했다.

이 대조군은 Experiment 12 결과를 본 뒤 추가한 post-hoc control이다.
통제 계획과 기준은 control mask/output 생성 전에 별도로 기록했다.

## Exact-area mask comparison

| Scale | Distance recovery | Oracle recovery | Delta | Distance precision | Oracle precision |
|---|---:|---:|---:|---:|---:|
| 1.5× | 0.5562 | 0.7219 | +0.1657 | 0.6221 | 0.8740 |
| 2× | 0.6341 | 0.8125 | +0.1784 | 0.5562 | 0.7884 |

두 control의 2,400프레임 모두 reference oracle과 추가 픽셀 수가 정확히
일치했다. Distance control은 GT·영상·temporal evidence 없이 raw
mask와의 거리만 사용했다.

## Downstream absolute results

| Scale | Backend | Distance PSNR | Oracle PSNR | Delta |
|---|---|---:|---:|---:|
| 1.5× | ProPainter | 11.641 | 12.490 | +0.850 |
| 1.5× | E2FGVI-HQ | 12.079 | 12.947 | +0.868 |
| 2× | ProPainter | 11.785 | 13.320 | +1.534 |
| 2× | E2FGVI-HQ | 12.152 | 13.433 | +1.281 |

Distance dilation 자체도 원래 1× dilation보다 개선됐다.

- 1.5×: ProPainter +1.124 dB, E2FGVI +0.833 dB
- 2×: ProPainter +1.269 dB, E2FGVI +0.907 dB

즉 budget size와 pixel placement가 모두 독립적으로 중요하다.

## Preregistered learned-selector gate

| Scale | Backend | Oracle vs distance | 95% CI | Wins | MAE delta | Outside delta | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| 1.5× | ProPainter | +0.850 | [+0.181, +1.553] | 7/10 | -0.01386 | -0.00014 | PASS |
| 1.5× | E2FGVI-HQ | +0.868 | [+0.328, +1.422] | 9/10 | -0.01831 | -0.00028 | PASS |
| 2× | ProPainter | +1.534 | [+0.681, +2.391] | 8/10 | -0.03326 | -0.00016 | PASS |
| 2× | E2FGVI-HQ | +1.281 | [+0.506, +2.309] | 8/10 | -0.03172 | -0.00032 | PASS |

Overall decision: **GO** at both scales.

## 실패와 한계

- synthetic_001은 두 scale·두 backend에서 oracle이 distance보다 낮았다.
- synthetic_002는 2×에서 사실상 동률이었다.
- 따라서 GT support ranking도 모든 장면에 최적인 것은 아니다.
- 이 실험은 oracle upper bound이며 observable-input 모델 성능이 아니다.
- 10클립 exploratory control이므로 새 test set 재현이 필요하다.

## 지원되는 다음 단계

단순 2× dilation을 method로 포장하는 방향은 부적절하다. 같은 면적에서도
선택 위치가 +0.85~1.53 dB를 추가로 만들었으므로, 학습 목표는
budget prediction과 residual-support ranking을 결합해야 한다.

## 산출물

- 설정: configs/same_budget_distance_control.json
- Control 생성: scripts/build_exact_area_distance_controls.py
- Paired 판정: scripts/analyze_same_budget_distance_control.py
- 대용량 결과:
  D:/DCG-TR_experiment_results/experiments/13_same_budget_distance_control
