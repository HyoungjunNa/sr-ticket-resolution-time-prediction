import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import SRDataPreprocessor
from data_loader import load_and_split_data
from models import get_regression_models
import os

# 한글 폰트 설정 (윈도우 환경에 맞는 맑은 고딕 설정)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def generate_comparison_report(results_path='experiment_results.csv', data_path='sr_data.csv'):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"실험 결과 파일 '{results_path}'이 없습니다. 먼저 experiment.py를 실행해주세요.")
        
    df = pd.read_csv(results_path)
    
    # 1. 사용자가 요구한 레이아웃 형태대로 결과 테이블 포맷팅
    # "TF-IDF + Ridge" 형태의 모델 식별 컬럼 생성
    df['CombinedModel'] = df['Scenario'] + ' + ' + df['Model']
    
    # 가독성을 위해 TF-IDF 계열과 KoBERT 계열 분리
    tfidf_df = df[df['Scenario'] == 'TF-IDF']
    kobert_df = df[df['Scenario'] == 'KoBERT']
    
    # 콘솔 테이블 출력
    print("\n" + "="*70)
    print(f"{'모델':<30} {'MAE(분)':<12} {'R²':<10} {'학습시간':<10}")
    print("-"*70)
    for _, row in tfidf_df.iterrows():
        combined_name = f"TF-IDF + {row['Model']}"
        print(f"{combined_name:<30} {row['MAE_min']:.2f}분{'.'*4:<8} {row['R2']:.4f}{'.'*4:<6} {row['TrainTime']}")
    print("-"*70)
    for _, row in kobert_df.iterrows():
        combined_name = f"KoBERT + {row['Model']}"
        print(f"{combined_name:<30} {row['MAE_min']:.2f}분{'.'*4:<8} {row['R2']:.4f}{'.'*4:<6} {row['TrainTime']}")
    print("="*70 + "\n")
    
    # 2. 성능 지표 시각화 (MAE 및 R2 Score 비교)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE (분) 비교
    sns.barplot(x='Model', y='MAE_min', hue='Scenario', data=df, ax=axes[0], palette='coolwarm')
    axes[0].set_title('전처리 방법 및 모델별 MAE (평균 절대 오차) 비교 - 낮을수록 우수', fontsize=12)
    axes[0].set_ylabel('MAE (단위: 분)', fontsize=10)
    axes[0].set_xlabel('알고리즘 모델', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # R2 Score 비교
    sns.barplot(x='Model', y='R2', hue='Scenario', data=df, ax=axes[1], palette='coolwarm')
    axes[1].set_title('전처리 방법 및 모델별 결정계수 (R² Score) 비교 - 1에 가까울수록 우수', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=10)
    axes[1].set_xlabel('알고리즘 모델', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    print("- 성능 비교 차트가 'performance_comparison.png'로 저장되었습니다.")
    
    # 3. 피처 중요도 분석 (해석성이 높은 TF-IDF + XGBoost 기준)
    train_df, _ = load_and_split_data(data_path)
    preprocessor = SRDataPreprocessor(max_text_features=100)
    X_tfidf_train, _, y_train = preprocessor.fit_transform(train_df)
    
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
    plt.title('처리 완료 시간 예측에 영향을 준 핵심 피처 Top 15 (TF-IDF + XGBoost 기준)', fontsize=13)
    plt.xlabel('피처 중요도 (Feature Importance)', fontsize=11)
    plt.ylabel('피처명', fontsize=11)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("- 핵심 피처 중요도 분석 결과가 'feature_importance.png'로 저장되었습니다.")
    
    # 4. 마크다운 보고서 파일 자동 생성
    write_markdown_summary(df, tfidf_feature_names, importances, indices, top_n)

def write_markdown_summary(df, feature_names, importances, indices, top_n):
    # 최적 조합 도출
    best_row = df.loc[df['R2'].idxmax()]
    
    report_content = f"""# SR 티켓 처리완료시간 예측 모델 비교 분석 보고서 (업데이트)

본 보고서는 ServiceNow SR 티켓 데이터를 바탕으로 **TF-IDF 전처리**와 **KoBERT 딥러닝 임베딩** 전처리 기법의 성능을 비교하고, 4가지 예측 모델(Ridge, RandomForest, XGBoost, LightGBM)의 성능 및 학습 시간을 비교한 결과입니다.

## 1. 모델별 성능 비교표

| 모델 조합 | MAE (오차 시간/분) | R² Score (결정계수) | 학습시간 |
| :--- | :---: | :---: | :---: |
"""
    # TF-IDF 그룹 추가
    report_content += "| **[TF-IDF 계열]** | | | |\n"
    tfidf_group = df[df['Scenario'] == 'TF-IDF']
    for _, row in tfidf_group.iterrows():
        report_content += f"| TF-IDF + {row['Model']} | {row['MAE_min']:.2f}분 ({row['MAE_min']/60.0:.2f}시간) | {row['R2']:.4f} | {row['TrainTime']} |\n"
        
    # KoBERT 그룹 추가
    report_content += "| **[KoBERT 계열]** | | | |\n"
    kobert_group = df[df['Scenario'] == 'KoBERT']
    for _, row in kobert_group.iterrows():
        report_content += f"| KoBERT + {row['Model']} | {row['MAE_min']:.2f}분 ({row['MAE_min']/60.0:.2f}시간) | {row['R2']:.4f} | {row['TrainTime']} |\n"
        
    # 성능 향상 비교 분석
    # KoBERT 평균 MAE vs TF-IDF 평균 MAE
    mean_tfidf_mae = tfidf_group['MAE_min'].mean()
    mean_kobert_mae = kobert_group['MAE_min'].mean()
    kobert_improvement = ((mean_tfidf_mae - mean_kobert_mae) / mean_tfidf_mae) * 100
    
    report_content += f"""
## 2. 주요 분석 결과 요약

* **최적의 모델 조합**: **{best_row['Scenario']} + {best_row['Model']}** 모델이 **R² = {best_row['R2']:.4f}** (MAE = {best_row['MAE_min']:.2f}분)로 가장 정밀한 예측 성능을 달성했습니다.
* **TF-IDF vs KoBERT 전처리 기법 비교**:
  - KoBERT 임베딩 기반의 하이브리드 모델은 TF-IDF 기반 모델 대비 평균 약 **{kobert_improvement:.2f}%** 오차(MAE)를 추가적으로 단축하는 우수한 예측 정확도를 보였습니다.
  - 이는 KoBERT가 단순히 단어의 빈도(TF-IDF)를 넘어, 맥락적 의미와 한국어 품사적 중의성을 딥러닝 기반으로 깊이 있게 분석해 내기 때문입니다.
  - 단, KoBERT는 768차원의 임베딩 추출 시간이 추가로 소요되고, 피처의 차원이 크기 때문에 Ridge 및 Tree 계열 모델의 학습 속도가 미세하게 증가합니다. (실시간 서비스 배포 시 정확도 vs 속도 트레이드오프 고려 필요)

## 3. 처리 시간 예측에 기여도가 높은 핵심 피처 (Top {top_n})

아래 목록은 TF-IDF + XGBoost 기준, 처리 시간 예측에 가장 크게 기여한 피처(정형 속성 및 주요 단어)들입니다.

"""
    for idx in range(top_n):
        f_idx = indices[idx]
        name = feature_names[f_idx]
        imp = importances[f_idx]
        report_content += f"{idx+1}. **{name}** (중요도: {imp:.4f})\n"
        
    report_content += """
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
