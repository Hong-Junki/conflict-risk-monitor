# 모델: XGBoost + SE

## 접근법

본 프로젝트의 목표는 가이드라인에서 권장한 주 타깃인 `y_escalation`을 예측하는 것이다. `y_escalation`은 단순히 향후 3일 내 사건이 발생하는지를 보는 `y`보다 더 의미 있는 타깃으로, 평시에서 분쟁 상태로 전환되거나 기존 분쟁이 급격히 악화되는 경우를 포착한다.

데이터 분할은 가이드라인을 그대로 따랐다.

- 학습 데이터: 2022-01-01 ~ 2023-12-31
- 검증 데이터: 2024-01-01 ~ 2024-06-30
- 테스트 데이터: 2024-07-01 ~ 2025-03-28

모델 선택과 하이퍼파라미터 튜닝은 검증 데이터 기준으로만 수행했다. 테스트 데이터는 최종 성능 확인에만 사용했다.

최종 모델은 `XGBoost + SE`이다. 기본 ACLED, GDELT, 경제 지표 피처에 `macis_se_score`를 추가한 `full_se.parquet` 데이터를 사용했다. 기본 피처만 사용한 모델보다 SE 피처를 추가했을 때 검증 성능이 크게 개선되었기 때문에, 최종 제출 모델은 SE 피처셋을 기반으로 구축했다.

## 모델 구성

최종 모델은 XGBoost 이진 분류기이다. 검증 데이터의 PR-AUC를 기준으로 모델을 선택했고, 최종 하이퍼파라미터는 다음과 같다.

```python
n_estimators=500
learning_rate=0.04
max_depth=4
min_child_weight=8
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=3.0
eval_metric="aucpr"
tree_method="hist"
```

튜닝 방향은 희소한 escalation 이벤트에 과적합하지 않도록 모델 복잡도를 낮추고 정규화를 강화하는 것이었다. 구체적으로 `max_depth`를 낮춰 트리의 깊이를 제한했고, `min_child_weight`와 `reg_lambda`를 키워 너무 작은 패턴에 과하게 반응하지 않도록 했다. 또한 이 문제는 양성 비율이 약 4% 수준인 희소 이벤트 탐지 문제이므로, 일반적인 `logloss`보다 PR-AUC에 가까운 `aucpr`를 학습 지표로 사용했다.

날짜 피처는 `date_ordinal` 형태로 변환했고, 국가는 범주형 변수로 one-hot encoding하여 사용했다. 라벨 및 미래 정보를 포함할 수 있는 메타 컬럼은 모두 학습 피처에서 제외했다.

## 결과

검증 데이터 기준 성능은 다음과 같다.

| 지표 | 값 |
|---|---:|
| PR-AUC | 0.162968 |
| P@top5% | 0.212121 |
| R@P>=0.10 | 0.627907 |
| ECE, 10 bins | 0.016864 |

테스트 데이터 기준 성능은 가이드라인의 `LightGBM + SE` baseline과 비교하면 다음과 같다.

| 지표 | 본 모델 | LightGBM + SE baseline | 차이 |
|---|---:|---:|---:|
| PR-AUC | 0.149685 | 0.130700 | +0.018985 |
| P@top5% | 0.198473 | 0.190000 | +0.008473 |
| R@P>=0.10 | 0.652038 | 0.558000 | +0.094038 |

테스트 기준으로 본 모델은 가이드라인의 `LightGBM + SE` baseline을 PR-AUC, P@top5%, R@P>=0.10 세 지표 모두에서 상회했다.

최종 option A 제출 파일은 다음과 같다.

```text
artifacts/submissions/predictions__xgboost__tuned_final.csv
```

제출 파일은 `date`, `country`, `y_prob` 세 컬럼으로 구성되어 있으며, 테스트 셋 전체인 15,718행을 포함한다.

## 위험 점수 산출

참고용으로 가이드라인의 0-100점 위험도 산식을 구현했다.

```text
risk_score = 0.2 * B + 0.4 * C_state + 0.4 * F + hotspot
```

각 구성요소는 다음과 같이 정의했다.

- `B`: 국가별 장기 사상자 prior. `processed/features/baseline_scores.parquet`의 `B_score`를 사용했다.
- `C_state`: 현재 상태 점수. 공유 데이터에 완성된 U/C/S/I 점수가 포함되어 있지 않기 때문에, ACLED와 GDELT 피처를 이용해 proxy 점수를 만들었다.
- `F`: ML 모델의 예측 확률을 0-100 범위로 변환한 점수. 검증 데이터 예측값의 99분위수를 100점 기준으로 사용했다.
- `hotspot`: 인접국 위험도 보정값. 현재 공유 데이터에 국가 인접 관계 테이블이 없으므로 기본값은 0으로 두었다.

`C_state`는 네 개의 하위 점수 평균으로 계산했다.

- `U_score`: 민간인 피해 및 사상자 관련 신호
- `C_score`: 전투, 폭발, 최근 ACLED 사건량 관련 신호
- `S_score`: GDELT의 갈등성 이벤트 비중과 부정적 뉴스 톤 관련 신호
- `I_score`: GDELT 언급량, 이벤트 수, Goldstein 변동성 관련 신호

각 하위 점수는 국가별 90일 rolling z-score를 계산한 뒤 0-100 범위로 변환했다. 최종 점수 파일은 다음과 같다.

```text
artifacts/submissions/risk_scores__xgboost__tuned_final.csv
```

위험 점수 파일도 테스트 셋과 동일하게 15,718행이며, 생성된 `risk_score`의 범위는 12.071 ~ 81.410이다.

## 한계 및 해석상 주의점

첫째, 최종 제출의 핵심은 `predictions__xgboost__tuned_final.csv`의 `y_prob`이다. 위험 점수 파일은 가이드라인의 0-100점 변환식을 참고용으로 구현한 부가 산출물이다.

둘째, `C_state`는 가이드라인의 원래 U/C/S/I 점수를 그대로 사용한 것이 아니라, 현재 제공된 ACLED/GDELT 피처로 만든 proxy이다. 따라서 운영용 위험 점수로 사용하려면 원래 정의의 U/C/S/I 점수와 비교 검증이 필요하다.

셋째, `hotspot` 항목은 인접국 정보를 필요로 하지만, 현재 공유 폴더에는 국가 간 인접 관계 데이터가 포함되어 있지 않다. 따라서 본 구현에서는 기본값을 0으로 두었고, 추후 adjacency table이 제공되면 해당 보정값을 추가할 수 있다.

요약하면, 본 모델은 SE 피처를 포함한 XGBoost 모델이며, 검증셋 기준으로 모델을 선택한 뒤 테스트셋에서 가이드라인의 `LightGBM + SE` baseline을 주요 지표에서 상회했다. 제출용 산출물은 option A 형식의 `predictions__xgboost__tuned_final.csv`이고, `risk_scores__xgboost__tuned_final.csv`는 결과 해석을 돕기 위한 참고용 점수 파일이다.
