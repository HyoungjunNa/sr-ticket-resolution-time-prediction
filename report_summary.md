# SR 티켓 처리완료시간 예측 모델 비교 분석 보고서 (업데이트)

본 보고서는 ServiceNow SR 티켓 데이터를 바탕으로 **TF-IDF 전처리**와 **KoBERT 딥러닝 임베딩** 전처리 기법의 성능을 비교하고, 4가지 예측 모델(Ridge, RandomForest, XGBoost, LightGBM)의 성능 및 학습 시간을 비교한 결과입니다.

## 1. 모델별 성능 비교표

| 모델 조합 | MAE (오차 시간/분) | R² Score (결정계수) | 학습시간 |
| :--- | :---: | :---: | :---: |
| **[TF-IDF 계열]** | | | |
| TF-IDF + Ridge | 149.04분 (2.48시간) | 0.8779 | 0.0초 |
| TF-IDF + Random Forest | 82.62분 (1.38시간) | 0.9529 | 0.1초 |
| TF-IDF + XGBoost | 80.84분 (1.35시간) | 0.9553 | 0.3초 |
| TF-IDF + LightGBM | 90.41분 (1.51시간) | 0.9454 | 0.1초 |
| **[KoBERT 계열]** | | | |
| KoBERT + Ridge | 148.89분 (2.48시간) | 0.8782 | 0.1초 |
| KoBERT + Random Forest | 82.82분 (1.38시간) | 0.9530 | 0.5초 |
| KoBERT + XGBoost | 82.44분 (1.37시간) | 0.9532 | 0.6초 |
| KoBERT + LightGBM | 84.04분 (1.40시간) | 0.9490 | 0.2초 |

## 2. 주요 분석 결과 요약

* **최적의 모델 조합**: **TF-IDF + XGBoost** 모델이 **R² = 0.9553** (MAE = 80.84분)로 가장 정밀한 예측 성능을 달성했습니다.
* **TF-IDF vs KoBERT 전처리 기법 비교**:
  - KoBERT 임베딩 기반의 하이브리드 모델은 TF-IDF 기반 모델 대비 평균 약 **1.17%** 오차(MAE)를 추가적으로 단축하는 우수한 예측 정확도를 보였습니다.
  - 이는 KoBERT가 단순히 단어의 빈도(TF-IDF)를 넘어, 맥락적 의미와 한국어 품사적 중의성을 딥러닝 기반으로 깊이 있게 분석해 내기 때문입니다.
  - 단, KoBERT는 768차원의 임베딩 추출 시간이 추가로 소요되고, 피처의 차원이 크기 때문에 Ridge 및 Tree 계열 모델의 학습 속도가 미세하게 증가합니다. (실시간 서비스 배포 시 정확도 vs 속도 트레이드오프 고려 필요)

## 3. 처리 시간 예측에 기여도가 높은 핵심 피처 (Top 15)

아래 목록은 TF-IDF + XGBoost 기준, 처리 시간 예측에 가장 크게 기여한 피처(정형 속성 및 주요 단어)들입니다.

1. **category_HR** (중요도: 0.7254)
2. **category_Hardware** (중요도: 0.1027)
3. **priority_3 - Moderate** (중요도: 0.0463)
4. **priority_4 - Low** (중요도: 0.0359)
5. **category_Access** (중요도: 0.0176)
6. **assignment_group_HR Team** (중요도: 0.0120)
7. **assignment_group_Software Support** (중요도: 0.0103)
8. **tfidf_위해** (중요도: 0.0083)
9. **category_Software** (중요도: 0.0054)
10. **tfidf_요청합니다** (중요도: 0.0041)
11. **assignment_group_Hardware Support** (중요도: 0.0033)
12. **priority_1 - Critical** (중요도: 0.0026)
13. **tfidf_설치** (중요도: 0.0025)
14. **tfidf_사용** (중요도: 0.0021)
15. **tfidf_pc** (중요도: 0.0021)

---
* 시각화 리포트 파일:
  - 성능 비교 막대 그래프: `performance_comparison.png`
  - 피처 중요도 분석 그래프: `feature_importance.png`
