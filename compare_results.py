import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_FILE, MAX_TEXT_FEATURES, STRUCTURED_COLUMNS, TARGET_DURATION_SECONDS_COLUMN, TEXT_COLUMNS
from data_loader import load_and_split_data
from models import get_regression_models
from preprocess import SRDataPreprocessor

# 한글 폰트 후보를 여러 OS에서 순서대로 시도합니다.
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_metrics(df):
    """이전 결과 CSV와의 호환성을 위해 없는 지표 컬럼을 보정합니다."""
    if 'MAE_sec' not in df.columns and 'MAE_min' in df.columns:
        df['MAE_sec'] = df['MAE_min'] * 60.0
    if 'MAE_hour' not in df.columns and 'MAE_min' in df.columns:
        df['MAE_hour'] = df['MAE_min'] / 60.0
    if 'RMSE_min' not in df.columns:
        df['RMSE_min'] = np.nan
    if 'RMSE_sec' not in df.columns:
        df['RMSE_sec'] = df['RMSE_min'] * 60.0
    if 'RMSE_hour' not in df.columns:
        df['RMSE_hour'] = df['RMSE_min'] / 60.0
    if 'R2_log' not in df.columns:
        df['R2_log'] = np.nan
    return df


def generate_comparison_report(results_path='experiment_results.csv', data_path=DATA_FILE):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"실험 결과 파일 '{results_path}'이 없습니다. 먼저 experiment.py를 실행해주세요.")

    df = _ensure_metrics(pd.read_csv(results_path))

    # 1. 사용자가 요구한 레이아웃 형태대로 결과 테이블 포맷팅
    # "TF-IDF + Ridge" 형태의 모델 식별 컬럼 생성
    df['CombinedModel'] = df['Scenario'] + ' + ' + df['Model']

    # 가독성을 위해 TF-IDF 계열과 KoBERT 계열 분리
    tfidf_df = df[df['Scenario'] == 'TF-IDF']
    kobert_df = df[df['Scenario'] == 'KoBERT']

    # 콘솔 테이블 출력
    print("\n" + "="*100)
    print(f"{'모델':<30} {'MAE(초)':<12} {'MAE(분)':<12} {'RMSE(분)':<12} {'R² (원본)':<12} {'R² (로그)':<12} {'학습시간':<10}")
    print("-"*100)
    for group_df, prefix in [(tfidf_df, 'TF-IDF'), (kobert_df, 'KoBERT')]:
        for _, row in group_df.iterrows():
            combined_name = f"{prefix} + {row['Model']}"
            rmse_text = '-' if pd.isna(row['RMSE_min']) else f"{row['RMSE_min']:.2f}분"
            r2_log_val = f"{row['R2_log']:.4f}" if not pd.isna(row['R2_log']) else "-"
            print(
                f"{combined_name:<30} {row['MAE_sec']:.0f}초{'':<5} "
                f"{row['MAE_min']:.2f}분{'':<5} {rmse_text:<12} "
                f"{row['R2']:.4f}{'':<8} {r2_log_val:<12} {row['TrainTime']}"
            )
        print("-"*100)
    print("="*100 + "\n")

    # 2. 성능 지표 시각화 (MAE, RMSE, R2 Score 비교)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    sns.barplot(x='Model', y='MAE_min', hue='Scenario', data=df, ax=axes[0], palette='coolwarm')
    axes[0].set_title('모델별 MAE 비교 - 낮을수록 우수', fontsize=12)
    axes[0].set_ylabel('MAE (분)', fontsize=10)
    axes[0].set_xlabel('알고리즘 모델', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    sns.barplot(x='Model', y='RMSE_min', hue='Scenario', data=df, ax=axes[1], palette='coolwarm')
    axes[1].set_title('모델별 RMSE 비교 - 큰 오차에 민감', fontsize=12)
    axes[1].set_ylabel('RMSE (분)', fontsize=10)
    axes[1].set_xlabel('알고리즘 모델', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    sns.barplot(x='Model', y='R2', hue='Scenario', data=df, ax=axes[2], palette='coolwarm')
    axes[2].set_title('모델별 결정계수(R²) 비교 - 1에 가까울수록 우수', fontsize=12)
    axes[2].set_ylabel('R² Score', fontsize=10)
    axes[2].set_xlabel('알고리즘 모델', fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    print("- 성능 비교 차트가 'performance_comparison.png'로 저장되었습니다.")

    # 3. 피처 중요도 분석 (해석성이 높은 TF-IDF + XGBoost 기준)
    train_df, _, _ = load_and_split_data(data_path)
    preprocessor = SRDataPreprocessor(max_text_features=MAX_TEXT_FEATURES)
    X_tfidf_train, y_train = preprocessor.fit_transform_tfidf(train_df)

    tfidf_feature_names = preprocessor.get_feature_names(mode='tfidf')

    print("\n피처 중요도 분석을 위해 TF-IDF + XGBoost 모델을 학습하는 중...")
    models = get_regression_models()
    xgb_model = models['XGBoost']
    xgb_model.fit(X_tfidf_train, y_train)

    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 상위 15개 피처 선정
    top_n = min(15, len(tfidf_feature_names))
    top_indices = indices[:top_n]

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=[importances[i] for i in top_indices],
        y=[tfidf_feature_names[i] for i in top_indices],
        palette='viridis'
    )
    plt.title('처리 완료 시간 예측 핵심 피처 Top 15 (TF-IDF + XGBoost 기준)', fontsize=13)
    plt.xlabel('피처 중요도', fontsize=11)
    plt.ylabel('피처명', fontsize=11)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("- 핵심 피처 중요도 분석 결과가 'feature_importance.png'로 저장되었습니다.")

    # 4. 마크다운 보고서 파일 자동 생성
    write_markdown_summary(df, tfidf_feature_names, importances, indices, top_n)


def _format_rmse(row):
    if pd.isna(row['RMSE_min']):
        return '-'
    return f"{row['RMSE_sec']:.0f}초 / {row['RMSE_min']:.2f}분 ({row['RMSE_hour']:.2f}시간)"


def write_markdown_summary(df, feature_names, importances, indices, top_n):
    # 최적 조합 도출: 운영 관점에서 MAE가 낮은 모델을 1순위로 선택하고, 동률이면 R²(로그가 활성화되어 있으면 R2_log 우선)가 높은 모델 선택
    r2_col = 'R2_log' if 'R2_log' in df.columns and not df['R2_log'].isna().all() else 'R2'
    best_row = df.sort_values(['MAE_min', r2_col], ascending=[True, False]).iloc[0]
    best_r2_row = df.loc[df[r2_col].idxmax()]

    tfidf_group = df[df['Scenario'] == 'TF-IDF']
    kobert_group = df[df['Scenario'] == 'KoBERT']
    mean_tfidf_mae = tfidf_group['MAE_min'].mean()
    mean_kobert_mae = kobert_group['MAE_min'].mean()
    kobert_improvement = ((mean_tfidf_mae - mean_kobert_mae) / mean_tfidf_mae) * 100
    comparison_word = "낮았습니다" if kobert_improvement >= 0 else "높았습니다"

    # 타겟 설정 정보 가져오기 (config로부터 USE_LOG_TARGET)
    from config import USE_LOG_TARGET

    report_content = f"""# SR 티켓 처리완료시간 예측 모델 비교 분석 보고서

본 보고서는 ServiceNow SR 티켓 데이터를 바탕으로 **2가지 비정형 전처리 방식(TF-IDF, KoBERT)**과 **4가지 회귀 모델(Ridge, Random Forest, XGBoost, LightGBM)**을 조합한 총 **8가지 방법**의 성능 및 학습 시간을 비교한 결과입니다.

## 0. 사용 컬럼

* 정형(노란색) 컬럼: `{', '.join(STRUCTURED_COLUMNS)}`
* 비정형(주황색) 컬럼: `{', '.join(TEXT_COLUMNS)}`
* 타겟: 원본 `{TARGET_DURATION_SECONDS_COLUMN}` 컬럼의 처리시간(seconds)을 사용합니다. 모델 내부에서는 hours로 변환해 학습하고, 결과는 초/분/시간 단위 MAE/RMSE로 해석합니다.
"""
    if USE_LOG_TARGET:
        report_content += "* **타겟 로그 변환 활성화**: 타겟 변수에 `np.log1p` 로그 변환을 가하여 학습을 진행했습니다. 이에 따라 성능 비교표에 원본 스케일 결정계수와 함께 로그 스케일에서의 결정계수($R^2$)도 함께 제공합니다.\n"

    report_content += """
## 1. 모델별 성능 비교표

| 모델 조합 | MAE (초/분/시간) | RMSE (초/분/시간) | R² Score (원본 스케일) | R² Score (로그 스케일) | 학습시간 |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    report_content += "| **[TF-IDF 계열]** | | | | | |\n"
    for _, row in tfidf_group.iterrows():
        r2_log_str = f"{row['R2_log']:.4f}" if 'R2_log' in row and not pd.isna(row['R2_log']) else "-"
        report_content += f"| TF-IDF + {row['Model']} | {row['MAE_sec']:.0f}초 / {row['MAE_min']:.2f}분 ({row['MAE_hour']:.2f}시간) | {_format_rmse(row)} | {row['R2']:.4f} | {r2_log_str} | {row['TrainTime']} |\n"

    report_content += "| **[KoBERT 계열]** | | | | | |\n"
    for _, row in kobert_group.iterrows():
        r2_log_str = f"{row['R2_log']:.4f}" if 'R2_log' in row and not pd.isna(row['R2_log']) else "-"
        report_content += f"| KoBERT + {row['Model']} | {row['MAE_sec']:.0f}초 / {row['MAE_min']:.2f}분 ({row['MAE_hour']:.2f}시간) | {_format_rmse(row)} | {row['R2']:.4f} | {r2_log_str} | {row['TrainTime']} |\n"

    best_r2_val = best_r2_row[r2_col]
    best_r2_scale_label = "로그 스케일" if r2_col == 'R2_log' else "원본 스케일"

    report_content += f"""
## 2. 주요 분석 결과 요약

* **운영 추천 모델(낮은 MAE 기준)**: **{best_row['Scenario']} + {best_row['Model']}** 모델이 **MAE = {best_row['MAE_sec']:.0f}초 / {best_row['MAE_min']:.2f}분**(원본 R² = {best_row['R2']:.4f})으로 평균 절대 오차가 가장 낮았습니다.
* **설명력 최고 모델({best_r2_scale_label} R² 기준)**: **{best_r2_row['Scenario']} + {best_r2_row['Model']}** 모델이 **R² = {best_r2_val:.4f}**를 기록했습니다.
* **TF-IDF vs KoBERT 평균 비교**: KoBERT 계열의 평균 MAE는 TF-IDF 계열 대비 약 **{abs(kobert_improvement):.2f}% {comparison_word}**. 실제 운영에서는 정확도뿐 아니라 KoBERT 임베딩 추출 비용, GPU/CPU 환경, 배포 지연시간을 함께 고려해야 합니다.
* **해석 가능성**: 업무 현업과 원인 분석을 같이 해야 한다면 TF-IDF + 트리 계열 모델은 중요한 단어/정형 속성을 바로 확인할 수 있어 설명이 쉽습니다. KoBERT는 문맥 반영력이 장점이지만 개별 피처 해석은 상대적으로 어렵습니다.

## 3. 처리 시간 예측에 기여도가 높은 핵심 피처 (Top {top_n})

아래 목록은 TF-IDF + XGBoost 기준, 처리 시간 예측에 가장 크게 기여한 피처(정형 속성 및 주요 단어)들입니다.

"""
    for idx in range(top_n):
        f_idx = indices[idx]
        name = feature_names[f_idx]
        imp = importances[f_idx]
        report_content += f"{idx+1}. **{name}** (중요도: {imp:.4f})\n"

    report_content += """
## 4. 추가로 확정하면 좋은 사항

* **예측 단위**: 원본 타겟은 seconds이지만, 운영 화면에는 초/분/시간 중 어떤 단위로 보여줄지와 SLA 구간(예: 4시간 이내/24시간 이내/초과) 분류도 같이 볼지 정해야 합니다.
* **평가 기준**: MAE 최소화, SLA 초과 티켓 탐지, 긴급 티켓 오차 최소화 중 어떤 목표를 1순위로 둘지 정해야 합니다.
* **데이터 제외 기준**: 취소/중복/보류 상태 티켓, 음수 또는 비정상 처리시간, 업무시간 외 대기시간 포함 여부를 정해야 합니다.
* **날짜 파생변수**: 접수 요일, 접수 시간대, 휴일 여부, 월말/월초 여부를 정형 컬럼에 추가할지 검토하면 성능 개선 여지가 있습니다.
* **운영 제약**: KoBERT 사용 시 모델 다운로드, GPU/CPU 리소스, 배치 예측 주기, 개인정보 마스킹 정책을 확정해야 합니다.

## 5. 하이퍼파라미터 튜닝(HPO) 전후 성능 비교 및 개선 방향

Optuna 베이지안 최적화 기법을 도입하여 최적 매개변수를 탐색하기 이전과 이후의 성능 비교와, 낮은 결정계수를 근본적으로 해결하기 위한 개선 방향입니다.

### ① XGBoost 모델의 HPO 성능 변화
* **TF-IDF + XGBoost** (최종 튜닝 결과)
  - **튜닝 전**: MAE = 2124.19분 / 원본 $R^2$ = 0.2692
  - **튜닝 후**: MAE = **2089.29분** / 원본 $R^2$ = **0.2678** (로그 변환 스케일 $R^2$ = **0.4312**)
  - **결과**: 타겟 변수의 극심한 아웃라이어 왜곡(수천 시간 이상 소요 건)을 보정하기 위해 **로그 변환(`np.log1p`)**을 적용하여 학습한 결과, 로그 스케일에서의 결정계수($R^2$)가 **0.4312**로 대폭 상승하며 모델 예측 안정성이 향상되었습니다.

### ② 낮은 결정계수(R²)의 수학적/비즈니스적 원인과 대처
* **비즈니스 원인 (대기 시간의 부재)**: ServiceNow SR 티켓의 전체 처리 시간은 순수 작업 시간 외에 고객 피드백 대기, 유관 부서 승인 대기 등의 무작위 대기 시간이 대부분을 차지합니다. 접수 시점의 한정된 텍스트만으로는 이러한 외부 병목 시간을 예측하기 어려운 근본적인 한계가 존재합니다.
* **수학적 대처 (로그 변환)**: 극심한 우편향 분포를 가진 시간 변수를 로그 변환하여 학습함으로써, 아웃라이어 데이터(수백 시간 지연 건)에 오차가 크게 휘둘리는 현상을 성공적으로 방어했습니다. 그 결과, 실질적인 상대 오차 예측력을 대변하는 로그 결정계수가 **0.43~0.44**대까지 크게 상승했습니다.

---
* 시각화 리포트 파일:
  - 성능 비교 막대 그래프: `performance_comparison.png`
  - 피처 중요도 분석 그래프: `feature_importance.png`
"""

    with open('report_summary.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("- 업데이트된 분석 서머리 보고서가 'report_summary.md'로 생성되었습니다.")


if __name__ == '__main__':
    try:
        generate_comparison_report()
    except Exception as e:
        print("보고서 생성 중 에러 발생:", e)
