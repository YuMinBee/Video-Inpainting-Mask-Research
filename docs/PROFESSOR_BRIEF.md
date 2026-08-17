# 연구 요약

## 한 줄 제목

**Budget-Density Temporal Mask Correction for Video Object Removal**

## 30초 설명

> 영상에서 객체를 지울 때 segmentation mask가 객체 경계를 덜 덮으면
> 인페인팅 뒤에도 잔상이 남습니다. 저는 먼저 방법마다 마스크를 키우는
> 양이 달라 생기는 착시를 없애기 위해, 같은 수의 픽셀을 추가하는
> exact-area 평가를 만들었습니다. 그 결과 장면마다 필요한 추가 면적도
> 다르고, 같은 면적에서도 어떤 픽셀을 고르는지가 중요하다는 것을
> 확인했습니다. 그래서 RGB·마스크 기하·SAM2의 정방향/역방향 시간 정보를
> 입력으로 받아, 픽셀별 잔여 객체 밀도를 예측하는 작은 모델을 학습했습니다.
> 이 밀도의 합으로 **얼마나 확장할지**를 정하고, 밀도 순위로 **어디를
> 확장할지**를 정합니다. 잠근 30영상 test에서 같은 면적의 거리 확장보다
> mask recovery가 `+0.102`, ProPainter가 `+1.205 dB`, E2FGVI-HQ가
> `+1.101 dB` 좋아져 사전등록 gate를 모두 통과했습니다.

## 무엇을 해결하는 연구인가

Segmentation mask는 “보이는 객체의 분할”에는 맞아도, 객체를 깨끗하게
지우기 위한 removal-support mask로는 부족할 수 있습니다. 특히 털, 손가락,
가느다란 구조, 움직이며 드러나는 경계가 mask 밖에 남습니다.

핵심 질문은 두 가지입니다.

1. 이 장면은 마스크를 **얼마나 더 넓혀야 하는가?**
2. 정해진 면적 안에서 **어느 픽셀을 추가해야 하는가?**

단순 dilation은 가까운 픽셀부터 균일하게 넓히기 때문에 강한 기준선이지만,
객체 내부의 구멍이나 비대칭 누락을 효율적으로 채우지 못합니다.

## 제안 방법을 쉽게 설명하면

기존 실패 모델은 한 프레임을 요약한 뒤 확장량 숫자 하나를 바로 예측했습니다.
이 방식은 validation에서 budget 오차가 커서 중단했습니다.

새 모델은 후보 픽셀마다 “이 픽셀이 아직 남은 객체일 확률”을 예측합니다.

```text
RGB + raw mask + 경계 거리 + SAM2 정/역방향 시간 정보
                         ↓
              픽셀별 residual density
                    ↙             ↘
       density 합 = 확장량       density 순위 = 위치
                    ↘             ↙
           연결성을 지킨 최종 correction mask
```

<p align="center">
  <img
    src="https://raw.githubusercontent.com/YuMinBee/Video-Inpainting-Mask-Research/main/assets/figures/mask_correction_comparison.png"
    alt="기존 동일 면적 거리 확장과 제안 방식 비교"
    width="100%"
  />
</p>

위 예시에서 두 방법은 똑같이 `7,191`픽셀을 추가하지만, 기존 거리 확장은
누락 영역의 `45.4%`, 제안 방식은 `60.3%`를 복구합니다. 빨강은 여전히
놓친 제거 대상, 초록은 새로 정확히 찾은 영역, 주황은 불필요하게 덮은
배경입니다. 이 그림은 설명용 한 프레임이며 전체 결과는 아래 30-source
통계로 판단합니다.

- 모델 크기: 120,050 parameters
- 입력: 관측 가능한 13채널
- 출력: 픽셀별 residual-support score와 그 합으로 얻은 budget
- 최종 선택: learned rank와 distance rank를 1:1로 섞은 연결형 frontier
- 추론에서 사용하지 않는 것: GT mask, clean background, inpainting 결과

## 실험을 어떻게 공정하게 했나

- 모든 공간 비교는 매 프레임 **추가 픽셀 수가 정확히 동일**합니다.
- train/validation/test를 원본 영상 source 단위로 분리했습니다.
- 35개 development source에서 5-fold OOF로 구조를 선택했습니다.
- 학습 epoch와 gate를 미리 고정했고 fold test로 early stopping하지 않았습니다.
- 최종 모델을 hash로 잠근 뒤에만 30개 sealed-test source를 생성했습니다.
- 5%와 10% SAM box 오류를 모두 평가했습니다.
- 실제 효과는 ProPainter와 E2FGVI-HQ 두 복원기에서 확인했습니다.

## 최종 결과

### 1. Budget 예측 병목을 해결함

| 모델 | OOF budget log-MAE | Source Spearman | 판정 |
|---|---:|---:|---|
| 기존형 scalar rank | 0.5867 | 0.1706 | FAIL |
| scalar + absolute 정보 | 0.4952 | 0.2328 | FAIL |
| ordinal budget | 0.5198 | 0.2420 | FAIL |
| **density budget** | **0.3300** | **0.6919** | **PASS** |

사전 기준은 MAE `<=0.45`, Spearman `>=0.30`이었습니다. 즉 확장량 자체가
예측 불가능했던 것이 아니라, 이전의 global scalar head가 병목이었습니다.

### 2. 같은 예측 면적에서도 픽셀 위치가 더 정확함

| 평가 | Recovery delta vs exact-area distance | 95% CI | 승리 |
|---|---:|---:|---:|
| 5-fold OOF, 35 source | `+0.0754` | `[+0.0634,+0.0870]` | 35/35 |
| Sealed test, 30 source | `+0.1020` | `[+0.0830,+0.1215]` | 29/30 |

Sealed test의 added precision도 `+0.1020` 높았습니다. 따라서 결과는 단순히
더 큰 마스크를 사용해서 얻은 효과가 아닙니다.

### 3. 실제 인페인팅도 두 backend에서 개선됨

| Backend | Masked PSNR | 95% CI | 승리 | Masked MAE | 판정 |
|---|---:|---:|---:|---:|---|
| **ProPainter** | **+1.205 dB** | **[+0.400,+2.335]** | **26/30** | **-0.0230** | PASS |
| **E2FGVI-HQ** | **+1.101 dB** | **[+0.339,+2.215]** | **23/30** | **-0.0202** | PASS |

마스크 밖 MAE도 두 backend 모두 감소했습니다. 사전등록 기준인 `+0.25 dB`,
양의 CI 하한, 18/30 이상 승리, masked MAE 감소, 외부 손상 제한을 모두
통과했습니다.

### 4. 큰 성공 한 사례를 빼도 결론이 유지됨

한 영상에서 거리 기준선은 mask 내부의 사람 모양 구멍을 남겼고, 제안 방식은
이를 채워 두 backend에서 약 `+14 dB`의 큰 이득이 났습니다. 이 사례를 빼고
다시 계산해도 다음과 같습니다.

| Post-hoc 점검 | ProPainter | E2FGVI-HQ |
|---|---:|---:|
| 평균 PSNR 차이 | +0.753 dB | +0.632 dB |
| 95% CI | [+0.297,+1.292] | [+0.253,+1.068] |
| 중앙값 | +0.333 dB | +0.261 dB |

이 점검은 사전 gate가 아니라 outlier 민감도 분석입니다.

### 5. 어떤 구성 요소가 실제 핵심인지 ablation함

Sealed test를 다시 선택에 사용하지 않고, 35개 development source의 고정
5-fold OOF에서 8개 구성을 비교했습니다.

| Component | 판정 | 해석 |
|---|---|---|
| **Per-pixel density budget** | **SUPPORTED** | scalar/ordinal보다 budget MAE와 순위가 크게 개선 |
| Absolute/temporal 입력 | 공동 gate 미통과 | budget에는 도움, spatial recovery의 독립 CI는 0 포함 |
| 0.5 distance 혼합 | NOT_SUPPORTED | learned-only 대비 recovery 차이는 작고 precision은 낮음 |
| Connected decoder | NOT_SUPPORTED | hybrid top-k도 이미 거의 연결됨 |
| Temporal smoothing | NOT_SUPPORTED | 변동은 줄지만 budget MAE와 recovery 악화 |

따라서 제안 방법의 가장 강한 기술적 핵심은 **픽셀별 residual density를
예측하고 그 합으로 장면별 budget을 얻는 formulation**입니다. 시간 정보는
budget 보조 신호로는 유효하지만, 별도의 spatial 핵심 기여라고 과장하지
않습니다.

## 실패 사례에서 무엇을 배웠나

- ProPainter는 4/30, E2FGVI-HQ는 7/30 영상에서 손해를 봤습니다.
- 두 backend가 함께 실패한 영상은 1개뿐입니다.
- 그 영상은 mask recovery가 `+0.0508` 좋아졌는데도 PSNR이 ProPainter
  `-1.304 dB`, E2FGVI-HQ `-0.604 dB`였습니다.
- 학습 mask가 객체를 더 가렸지만, 복원기가 큰 갈색 얼룩을 만들었습니다.

따라서 “객체 픽셀을 더 많이 찾으면 무조건 복원이 좋아진다”는 주장은 하지
않습니다. 다음 병목은 경계 탐지 자체보다 **복원기가 잘 채울 수 있는 support와
시간적으로 안정적인 support를 고르는 것**입니다.

## 이전 실험이 실패였던 이유와 이번 차이

```text
Experiment 14
pooled scalar budget 예측
        -> validation budget MAE 0.5599
        -> 사전 기준 실패
        -> test 미개봉

Experiment 15
pixel density를 예측하고 합으로 budget 계산
        -> 5-fold OOF budget MAE 0.3300
        -> model lock
        -> sealed mask test GO
        -> 두 inpainting backend GO
```

이 흐름은 실패 결과를 버리고 threshold를 낮춘 것이 아니라, 병목 가설을 새
experiment ID로 바꾸고 같은 test를 보지 않은 채 다시 검증한 것입니다.

## 현재 어디까지 주장할 수 있나

### 주장 가능

- boundary under-coverage를 exact-area protocol로 정량화했다.
- 관측 가능한 시간 정보와 mask geometry로 확장량과 위치를 함께 학습했다.
- source-disjoint OOF와 잠긴 30-source test에서 거리 기준선을 넘었다.
- 두 inpainting backend에서 사전등록한 유용 효과 기준을 통과했다.
- 실패 사례와 큰 outlier의 원인을 따로 분석했다.

### 아직 주장하면 안 됨

- 모든 실제 사용자 마스크에서 일반화한다.
- 최신 imperfect-mask removal/refinement보다 우수하다.
- 새로운 SAM2 구조 자체를 제안했다.
- 현재 결과만으로 강한 SOTA 논문이 완성됐다.

## 논문 가능성에 대한 현재 판단

이제는 단순 진단 프로젝트가 아니라 **held-out downstream까지 통과한 학습형
method prototype**이며 신규 방법 후보로 설명할 수 있습니다.
다만 논문화의 마지막 관문은 다음 세 가지입니다.

1. 최신 경쟁 refinement/robust-removal baseline과 공정 비교
2. 자연스럽게 발생한 imperfect mask 또는 외부 실영상 defect test
3. density budget, absolute temporal 정보, distance 혼합, 연결성의 ablation

세 번째 항목의 35-source OOF ablation은 완료됐습니다. Density formulation만
완전히 지지됐고 나머지 구성은 주장 범위를 좁혔습니다. 경쟁 baseline과
실영상 하나까지 재현되면 applied paper로 밀기
좋고, 둘 다 무너지면 강한 연구형 프로젝트와 실패 분석으로 남기는 것이 맞습니다.

## 다음 관문 준비 상태 — 아직 결과는 아님

현재 GPU 학습과 충돌하지 않도록 **실험은 실행하지 않고 프로토콜과 코드만
준비**했습니다.

- Exp17: 기존 DAVIS/Cutie 자연 오류 30편에 잠근 모델을 그대로 적용하는 전이
  파일럿. 과거 통계를 본 데이터라 최종 외부 test로 부르지 않음.
- Exp18: VOST validation 70편을 영상이나 GT를 보기 전에 이름 hash로
  `pilot 30 / confirmatory 40`으로 고정. Cutie `base-nomose`도 hash로 잠금.
- Exp19: 최신 SVOR는 mask 방법인 것처럼 exact-area 표에 넣지 않고,
  clean-background가 있는 기존 30-source test에서 end-to-end removal system으로
  비교하도록 고정.
- 세 실험 모두 기존 결과와 다른 경로에 저장하며, 첫 prompt frame 제외,
  clip 단위 bootstrap, gate threshold를 결과 전에 기록함.

Exp17의 CPU 입력 감사에서는 30편·1,999프레임이 모두 정합했고, Cutie raw
mask가 평균 GT의 `3.98%`를 놓쳤으며 30-pixel 후보 영역이 그 누락의
`96.68%`를 포함했습니다. 따라서 교정할 자연 오류와 방법의 headroom은
존재합니다. 이 수치는 **입력 난이도 점검일 뿐 제안법의 성능 결과는 아닙니다.**

실행 순서는 **Exp17 → 통과 시 Exp18 → 통과 시 Exp19**입니다. 자세한 실행 및
주장 규칙은 [외부 검증 준비안](EXTERNAL_VALIDATION_PLAN.md)에 있습니다.

## 본인 기여 설명

1. segmentation mask와 removal-support mask의 차이를 boundary under-coverage로 정의
2. 마스크 크기 착시를 없앤 exact-area 평가 설계
3. fixed budget과 variable budget, 면적 효과와 위치 효과를 단계적으로 분리
4. 이전 budget-head 실패 원인을 진단하고 density formulation 제안
5. source-disjoint 5-fold, model lock, sealed test, 두 backend gate 설계와 수행
6. 공통 실패, backend별 실패, outlier가 평균에 미치는 영향 분석

## 근거 문서

- [Experiment 15 전체 결과](../reports/learned_budget_density_cv/RESULTS.md)
- [연구 흐름과 근거 맵](RESEARCH_FLOW.md)
- [Experiment 15 사전등록](../reports/learned_budget_density_cv/PREREGISTRATION.md)
- [Downstream 사전 계획](../reports/learned_budget_density_cv/DOWNSTREAM_PLAN.md)
- [이전 Experiment 14 결과](../reports/learned_budget_support_pilot/RESULTS.md)
- [Experiment 16 component ablation](../reports/budget_density_component_ablation/RESULTS.md)

대용량 모델·프레임·복원 결과는
`D:/DCG-TR_experiment_results/experiments/15_learned_budget_density_cv`에
기존 실험과 분리해 보존했습니다.
