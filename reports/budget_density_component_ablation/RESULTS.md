# Experiment 16: Budget-Density Component Ablation

## 결론

35개 source의 고정 5-fold OOF에서 사전등록한 component gate를 적용한 결과,
**density formulation만 완전히 `SUPPORTED`** 판정을 받았다.

| Component | 판정 | 핵심 이유 |
|---|---|---|
| Density budget | **SUPPORTED** | scalar/ordinal보다 MAE `0.165~0.190` 감소, Spearman `0.450~0.459` 증가 |
| Absolute temporal probability | NOT_SUPPORTED | budget은 개선됐지만 recovery CI 하한 `-0.0074` |
| Any temporal evidence | NOT_SUPPORTED | budget은 개선됐지만 recovery CI 하한 `-0.0019` |
| 0.5 distance prior | NOT_SUPPORTED | recovery `+0.0049`, precision `-0.0111` |
| Connected decoder | NOT_SUPPORTED | selected hybrid top-k도 거의 모두 raw mask에 연결됨 |
| Temporal budget smoothing | NOT_SUPPORTED | 변동은 58.4% 감소했지만 MAE 증가 `+0.0247`로 기준 `+0.02` 초과 |

이 판정은 Experiment 15의 잠긴 모델이나 sealed-test 결과를 바꾸지 않는다.
이번 실험의 primary split에는 이미 결과를 본 Experiment 15 test를 사용하지
않았다.

## 실험 범위

- Primary data: Experiment 15와 동일한 35 development source
- Split: 고정 source-group 5-fold
- 새 학습: `density_geometry` 5-fold + `density_rank` 5-fold
- Epoch: 각 fold 고정 8회
- 새 checkpoint: 10개
- 새 OOF prediction: 3,840행
- Full-resolution 평가: 1,920 frame × 8 variant = 15,360행
- Exact-budget mismatch: 0
- Bootstrap: source 35개, 20,000회, seed 20260851

## 1. Budget ablation

| Candidate | Log-MAE | Spearman | Temporal variation |
|---|---:|---:|---:|
| Scalar absolute | 0.4952 | 0.2328 | 0.1487 |
| Ordinal absolute | 0.5198 | 0.2420 | 0.1717 |
| **Density absolute** | **0.3300** | **0.6919** | 0.2249 |
| Density + temporal smoothing | 0.3547 | 0.6728 | **0.0935** |
| Density rank-only | 0.4449 | 0.3468 | 0.2123 |
| Density geometry-only | 0.4066 | 0.4675 | 0.1949 |

### 해석

가장 강하게 지지된 것은 입력 채널보다 **budget을 만드는 방식**이다. 같은
13채널에서도 density는 scalar보다 MAE가 `0.1653`, ordinal보다 `0.1898`
낮고 Spearman은 각각 `0.4591`, `0.4499` 높았다.

Absolute probability와 temporal evidence도 budget에는 기여했다.

- Full vs rank-only MAE 이득: `+0.1149`
- Full vs geometry-only MAE 이득: `+0.0766`

따라서 이 정보가 쓸모없다는 결과는 아니다. 정확한 표현은 **budget 기여는
지지되지만, end-to-end mask recovery의 독립 필수성은 이번 공동 gate에서
확정되지 않았다**이다.

## 2. Full-resolution mask ablation

| Variant | Recovery | Precision | Unreachable fraction |
|---|---:|---:|---:|
| Full-budget distance | 0.6453 | 0.6525 | 0.0000 |
| **Full hybrid connected** | **0.7206** | 0.7383 | 0.0000 |
| Full hybrid, smoothed budget | 0.7187 | 0.7347 | 0.0000 |
| Full hybrid top-k | 0.7206 | 0.7383 | 0.00005 |
| Full learned-only connected | 0.7157 | **0.7494** | 0.0000 |
| Full learned-only top-k | 0.7142 | 0.7481 | 0.0070 |
| Geometry-only hybrid | 0.6942 | 0.6992 | 0.0000 |
| Rank-only hybrid | 0.6964 | 0.7296 | 0.0000 |

## 3. Frozen paired comparisons

| Candidate | Reference | Recovery delta | 95% CI | Wins | Precision delta |
|---|---|---:|---:|---:|---:|
| Full hybrid connected | Full-budget distance | `+0.0754` | `[+0.0634,+0.0868]` | 35/35 | `+0.0858` |
| Full hybrid connected | Rank-only hybrid | `+0.0242` | `[-0.0074,+0.0578]` | 20/35 | `+0.0087` |
| Full hybrid connected | Geometry-only hybrid | `+0.0265` | `[-0.0019,+0.0568]` | 20/35 | `+0.0391` |
| Full hybrid connected | Learned-only connected | `+0.0049` | `[-0.0121,+0.0240]` | 17/35 | `-0.0111` |
| Full hybrid connected | Full hybrid top-k | `+0.0000` | `[-0.0000,+0.0000]` | 14/35 | `+0.0000` |
| Learned-only connected | Learned-only top-k | `+0.0015` | `[+0.0010,+0.0022]` | 32/35 | `+0.0013` |
| Smoothed-budget hybrid | Unsmoothed hybrid | `-0.0020` | `[-0.0038,-0.0002]` | 12/35 | `-0.0036` |

## 4. Component별 해석

### Density formulation: 지지됨

Scalar와 ordinal 모두 같은 absolute 입력을 받았지만 density보다 budget
오차가 훨씬 컸다. Experiment 14의 실패는 correction amount가 관측
불가능해서가 아니라 global pooling 후 숫자 하나를 회귀한 구조가 병목이었다는
해석이 강화됐다.

### Absolute/temporal channels: budget에는 유효, spatial 필수성은 미확정

Full 입력의 recovery 평균은 rank-only보다 `+0.0242`, geometry-only보다
`+0.0265` 높았다. 그러나 source 차이가 커서 CI 하한이 0 아래였다.

Source별 budget-MAE 개선과 recovery 개선은 양의 상관을 보였다.

- Full vs rank-only: Pearson `0.510`, Spearman `0.415`
- Full vs geometry-only: Pearson `0.582`, Spearman `0.534`

이는 temporal/absolute 정보의 주된 관찰 효과가 spatial rank 자체보다 budget
정확도에서 나왔을 가능성을 보여준다. 더 정확한 작은 budget을 예측해 recovery는
낮지만 precision은 높아진 source도 있어, 서로 다른 예측 면적의 recovery만으로
채널 가치를 단정하면 안 된다.

### Distance prior: 필수라고 보기 어려움

Hybrid는 learned-only보다 recovery가 `+0.0049` 높았지만 CI가 0을 포함하고,
precision은 `-0.0111` 낮았다. 현재 1:1 혼합은 안전한 historical prior이지만,
ablation상 독립 핵심 기여로 주장할 수 없다.

### Connectivity: hybrid에서는 중복, learned-only에서는 작은 이득

Distance prior가 포함된 hybrid top-k의 unreachable fraction은 이미
`0.00005`뿐이어서 connected decoder와 결과가 사실상 같았다. 반면
learned-only에서는 connectivity가 top-k보다 recovery `+0.0015`, precision
`+0.0013` 높고 unreachable fraction을 `0.0070` 줄였다.

즉 distance prior와 connectivity가 비슷한 안정화 역할을 일부 중복 수행하는
것으로 보인다. 다만 frozen primary check는 selected hybrid에서 정의됐으므로
connectivity component 판정은 `NOT_SUPPORTED`이다.

### Temporal smoothing: trade-off는 있으나 gate 실패

Budget temporal variation은 58.4% 감소했다. 그러나 budget MAE가 `+0.0247`
증가해 허용치 `+0.02`를 넘었고 recovery도 `-0.0020` 감소했다. 따라서 현재
고정 kernel은 사용하지 않는 Experiment 15 선택이 타당했다.

## 5. 논문 서사에 미치는 영향

Experiment 15의 결과는 유지되지만 method 설명은 좁혀야 한다.

### 강하게 주장 가능

- Per-pixel residual density를 적분해 variable budget을 만드는 formulation
- 같은 예측 면적의 distance보다 learned placement가 높은 recovery를 보인 결과
- Source-disjoint OOF, model lock, sealed test, 두 downstream backend의 검증

### 약하게 또는 한계로 표현

- Absolute temporal probability는 budget에 도움을 주지만 spatial 필수성은 미확정
- 0.5 distance mixture와 connected decoder는 selected hybrid에서 독립 이득이 없음
- Fixed temporal smoothing은 부드러움과 정확도 사이 trade-off로 gate 실패

## 6. 다음 우선순위

현재 내부 구성 요소를 더 튜닝하기보다 다음 paper gate가 우선이다.

1. 최신 imperfect-mask refinement/robust-removal baseline
2. 자연 imperfect mask 또는 외부 real-video defect
3. 필요할 경우에만 learned-only connected와 locked hybrid의 downstream post-hoc ablation

이번 OOF 결과를 보고 Experiment 15 test에서 mixture나 decoder를 다시 선택하지
않는다.

## Reproducibility

- Config SHA-256:
  `f6b6dbb0a6fca1c260d32362ec00be9c9b10ce8150bdf713f89f887f25f0b603`
- Training wrapper SHA-256:
  `deda9af9025651389f6bf0cef8cb22e98d80e092433f5aec2e226577c25ec4ea`
- Evaluator SHA-256:
  `93681bfbdc40022c2288279dc03b94652cff6991c040eb5cb3f385d6ee693280`
- Large output:
  `D:/DCG-TR_experiment_results/experiments/16_budget_density_component_ablation`
