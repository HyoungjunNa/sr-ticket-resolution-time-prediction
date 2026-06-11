import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import MAX_TEXT_FEATURES, STRUCTURED_COLUMNS, TEXT_COLUMNS
from data_loader import load_and_split_data
from models import get_regression_models
from preprocess import SRDataPreprocessor

# 한글 폰트 후보를 여러 OS에서 순서대로 시도합니다.
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_metrics(df):
    """이전 결과 CSV와의 호환성을 위해 없는 지표 컬럼을 보정합니다."""
    if 'RMSE_min' not in df.columns:
        df['RMSE_min'] = np.nan
    return df


def generate_comparison_report(results_path='experiment_results.csv', data_path='sr_data.csv'):
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
    print("\n" + "="*86)
    print(f"{'모델':<30} {'MAE(분)':<12} {'RMSE(분)':<12} {'R²':<10} {'학습시간':<10}")
    print("-"*86)
    for group_df, prefix in [(tfidf_df, 'TF-IDF'), (kobert_df, 'KoBERT')]:
        for _, row in group_df.iterrows():
            combined_name = f"{prefix} + {row['Model']}"
            rmse_text = '-' if pd.isna(row['RMSE_min']) else f"{row['RMSE_min']:.2f}분"
            print(f"{combined_name:<30} {row['MAE_min']:.2f}분{'':<5} {rmse_text:<12} {row['R2']:.4f}{'':<6} {row['TrainTime']}")
        print("-"*86)
    print("="*86 + "\n")

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
    train_df, _ = load_and_split_data(data_path)
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
    return f"{row['RMSE_min']:.2f}분 ({row['RMSE_min']/60.0:.2f}시간)"


def write_markdown_summary(df, feature_names, importances, indices, top_n):
    # 최적 조합 도출: 운영 관점에서 MAE가 낮은 모델을 1순위로 선택하고, 동률이면 R²가 높은 모델 선택
    best_row = df.sort_values(['MAE_min', 'R2'], ascending=[True, False]).iloc[0]
    best_r2_row = df.loc[df['R2'].idxmax()]

    tfidf_group = df[df['Scenario'] == 'TF-IDF']
    kobert_group = df[df['Scenario'] == 'KoBERT']
    mean_tfidf_mae = tfidf_group['MAE_min'].mean()
    mean_kobert_mae = kobert_group['MAE_min'].mean()
    kobert_improvement = ((mean_tfidf_mae - mean_kobert_mae) / mean_tfidf_mae) * 100
    comparison_word = "낮았습니다" if kobert_improvement >= 0 else "높았습니다"

    report_content = f"""# SR 티켓 처리완료시간 예측 모델 비교 분석 보고서

본 보고서는 ServiceNow SR 티켓 데이터를 바탕으로 **2가지 비정형 전처리 방식(TF-IDF, KoBERT)**과 **4가지 회귀 모델(Ridge, Random Forest, XGBoost, LightGBM)**을 조합한 총 **8가지 방법**의 성능 및 학습 시간을 비교한 결과입니다.

## 0. 사용 컬럼

* 정형(노란색) 컬럼: `{', '.join(STRUCTURED_COLUMNS)}`
* 비정형(주황색) 컬럼: `{', '.join(TEXT_COLUMNS)}`
* 타겟: `closed_at - opened_at`으로 계산한 처리시간(시간 단위)을 예측하고, 결과는 분 단위 MAE/RMSE로 해석합니다.

## 1. 모델별 성능 비교표

| 모델 조합 | MAE (오차 시간/분) | RMSE (오차 시간/분) | R² Score (결정계수) | 학습시간 |
| :--- | :---: | :---: | :---: | :---: |
"""
    report_content += "| **[TF-IDF 계열]** | | | | |\n"
    for _, row in tfidf_group.iterrows():
        report_content += f"| TF-IDF + {row['Model']} | {row['MAE_min']:.2f}분 ({row['MAE_min']/60.0:.2f}시간) | {_format_rmse(row)} | {row['R2']:.4f} | {row['TrainTime']} |\n"

    report_content += "| **[KoBERT 계열]** | | | | |\n"
    for _, row in kobert_group.iterrows():
        report_content += f"| KoBERT + {row['Model']} | {row['MAE_min']:.2f}분 ({row['MAE_min']/60.0:.2f}시간) | {_format_rmse(row)} | {row['R2']:.4f} | {row['TrainTime']} |\n"

    report_content += f"""
## 2. 주요 분석 결과 요약

* **운영 추천 모델(낮은 MAE 기준)**: **{best_row['Scenario']} + {best_row['Model']}** 모델이 **MAE = {best_row['MAE_min']:.2f}분**(R² = {best_row['R2']:.4f})으로 평균 절대 오차가 가장 낮았습니다.
* **설명력 최고 모델(R² 기준)**: **{best_r2_row['Scenario']} + {best_r2_row['Model']}** 모델이 **R² = {best_r2_row['R2']:.4f}**를 기록했습니다.
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

* **예측 단위**: 처리시간을 시간/분으로 예측할지, SLA 구간(예: 4시간 이내/24시간 이내/초과) 분류도 같이 볼지 정해야 합니다.
* **평가 기준**: MAE 최소화, SLA 초과 티켓 탐지, 긴급 티켓 오차 최소화 중 어떤 목표를 1순위로 둘지 정해야 합니다.
* **데이터 제외 기준**: 취소/중복/보류 상태 티켓, 음수 또는 비정상 처리시간, 업무시간 외 대기시간 포함 여부를 정해야 합니다.
* **날짜 파생변수**: 접수 요일, 접수 시간대, 휴일 여부, 월말/월초 여부를 정형 컬럼에 추가할지 검토하면 성능 개선 여지가 있습니다.
* **운영 제약**: KoBERT 사용 시 모델 다운로드, GPU/CPU 리소스, 배치 예측 주기, 개인정보 마스킹 정책을 확정해야 합니다.

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
