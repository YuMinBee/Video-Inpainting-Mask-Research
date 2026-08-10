# Video Inpainting Mask Research

이 저장소는 비디오 객체 제거에서 depth와 temporal 단서를 탐색하고, 최종적으로 **분할 마스크와 실제 제거에 필요한 support mask가 같은가**를 검증한 독립 연구의 코드와 결과 요약을 담고 있습니다. 연구는 Depth Anything 3 기반 경계 안정화에서 출발해 temporal mask refinement로 확장되었으며, 각 단계에서 단순 기준선과 통제 실험을 통해 단서의 실제 기여를 분리했습니다.

현재 결과가 지지하는 결론은 다음과 같습니다.

> 마스크 품질은 제거 성능에 직접 영향을 주지만, 현재 synthetic-SAM 설정의 주된 오류는 국소적인 경계 누락이다. 따라서 단순한 area-matched dilation이 매우 강한 기준선이며, temporal occupancy만으로는 우월한 제거 마스크를 만든다는 주장을 지지하기 어렵다.

## 연구 흐름

1. **Depth-aware temporal refinement** — Depth Anything 3 pseudo-depth를 optical-flow 기반 경계 blending의 reliability cue로 사용했습니다. AI-Hub 20클립에서 baseline보다 BTE를 낮추고 경계 형태를 보존했지만, temporal error 감소에서는 단순 boundary-only smoothing을 이기지 못했습니다.
2. **연구 질문 전환** — depth가 모든 flicker를 설명하지 못하고 post-processing만으로는 불완전한 제거 영역을 해결할 수 없음을 확인해, segmentation mask가 removal-support mask로 충분한지 질문을 확장했습니다.
3. **Temporal mask refinement** — temporal union과 occupancy를 이용해 누락 가능 영역을 선택적으로 확장하고, 보수적인 boundary-only와 공격적인 temporal-union 사이의 trade-off를 측정했습니다.
4. **실영상 프록시 평가** — AI-Hub 100클립과 DAVIS 30클립에서 BTE, Outside change, Extra mask, residue proxy를 평가했습니다.
5. **강한 통제 기준선 추가** — 동일 extra-mask 예산의 dilation과 distance-only를 비교해 temporal occupancy의 고유 기여를 분리했습니다.
6. **Synthetic-GT 검증** — 깨끗한 배경 정답이 있는 10클립에서 실제 복원 MAE/PSNR/SSIM을 측정했습니다.
7. **오류 원인 및 대안 분석** — SAM 누락 픽셀의 거리 분포와 mask correction 효율을 분석하고, SAM prompt-ensemble uncertainty를 동일 예산에서 검증했습니다.
8. **결론** — depth는 구조 보존을 위한 보조 단서로 유효했지만, temporal occupancy와 함께 단독 해결책으로는 부족했습니다. 현재 설정에서는 국소 경계 누락에 맞는 단순 dilation이 강한 기준선입니다.

초기 depth 실험과 전환 근거는 [Depth-aware temporal refinement 요약](reports/depth_aware_temporal_refinement.md), 전체 주장-근거 연결은 [연구 흐름과 해석](docs/RESEARCH_FLOW.md), 후속 실험 기록은 [연구 로그](docs/mask_removal_research_log.md), 재현 명령은 [재현 가이드](reproducibility/README.md)를 참고하십시오.

## 핵심 결과

| 평가 | 관찰 | 연구적 해석 |
|---|---|---|
| Depth-aware 20 | BTE를 baseline 대비 20.16% 낮췄지만, boundary-only smoothing의 52.88% 감소보다는 작음 | depth는 경계 보존용 reliability cue로는 유효하지만 temporal 안정성 향상의 주된 원인이라고 보기 어려움 |
| AI-Hub 100 | Ours-Balanced는 Boundary-only보다 BTE가 낮지만 Outside/Extra가 큼 | 보수성과 제거력 사이 trade-off는 개선했으나, 선택성 자체의 독립적 이득은 작음 |
| 동일 예산 기준선 | Ours-Balanced와 distance-only의 평균 BTE가 거의 같음 | occupancy gate의 고유 기여를 강하게 주장하기 어려움 |
| DAVIS 30 | Ours-Balanced는 BTE를 낮추지만 Outside change가 증가 | 다른 데이터에서도 같은 trade-off가 재현됨 |
| Synthetic-GT 10 | SAM-jitter 조건에서 area-matched dilation의 MAE가 가장 낮음 | 프록시가 아닌 복원 정답 평가에서 단순 기하 기준선이 우세 |
| 결손 거리 | 누락 픽셀 55.4%가 raw mask 10 px 이내 | 현재 오류가 시간적 누락보다 경계 축소에 치우침 |
| SAM uncertainty | 같은 예산에서 recovery 약 0.283, dilation은 0.799 | box-jitter 불확실성은 유효한 누락 단서가 아님 |

Git에 포함된 경량 결과는 [결과 스냅샷](reports/results_snapshot_2026-06-28/README.md)에 있습니다. 프레임 이미지, 모델 출력, 체크포인트 등 대용량 산출물은 D: 드라이브에 유지하고 Git에서는 제외합니다.

## 저장소 구성

```text
configs/          안정화된 실험 설정
docs/             연구 질문, 실험 흐름, 결론과 한계
reports/          Git에 보존할 핵심 표와 clip-level 지표
reproducibility/  환경과 전체 재현 절차
scripts/          데이터 준비, 마스크 생성, 평가, 분석 코드
external/         외부 저장소 복원 안내
```

원본 데이터와 전체 산출물의 역할별 디렉터리 규칙은 [프로젝트 구조](PROJECT_STRUCTURE.md)에 정리되어 있습니다.
