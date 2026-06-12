import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import DATA_FILE, RANDOM_STATE, BERT_MODEL_NAME, MAX_TEXT_FEATURES
from data_loader import load_and_split_data
from preprocess import SRDataPreprocessor
from models import get_regression_models

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_shap_analysis():
    print("=========================================")
    # 1. 데이터 로드
    train_df, val_df, test_df = load_and_split_data(DATA_FILE)
    print(f"데이터 로드 완료. 학습: {len(train_df)}건, 테스트: {len(test_df)}건")
    
    # 2. KoBERT 피처 추출
    print("KoBERT 피처 추출 시작 (KyKim BERT 임베딩 추출)...")
    preprocessor = SRDataPreprocessor(max_text_features=MAX_TEXT_FEATURES, bert_model_name=BERT_MODEL_NAME)
    
    # fit_transform과 transform을 통해 KoBERT 피처를 뽑습니다.
    # 전처리 속도 단축을 위해 TF-IDF를 제외한 KoBERT 피처만 사용할 수 있게 되어있으므로
    # fit_transform을 그대로 사용합니다.
    _, X_train_kobert, y_train = preprocessor.fit_transform(train_df)
    _, X_test_kobert, y_test = preprocessor.transform(test_df)
    
    # KoBERT 피처 이름 리스트 가져오기
    kobert_feature_names = preprocessor.get_feature_names(mode='kobert')
    print(f"KoBERT 피처 결합 완료 (차원 수: {X_train_kobert.shape})")
    
    # 3. XGBoost 최적 하이퍼파라미터 모델 생성 및 학습
    # models.py에 등록해 둔 KoBERT 시나리오 전용 XGBoost 인스턴스를 가져옵니다.
    print("최적화된 XGBoost 모델 학습 중...")
    models = get_regression_models(scenario='KoBERT', random_state=RANDOM_STATE)
    xgb_model = models['XGBoost']
    
    start_t = time.time()
    xgb_model.fit(X_train_kobert, y_train)
    print(f"모델 학습 완료 (소요 시간: {time.time() - start_t:.1f}초)")
    
    # 4. SHAP 분석 수행
    print("SHAP TreeExplainer 초기화 및 SHAP Value 계산 중...")
    # 계산 가속화를 위해 테스트 데이터에서 200개 샘플만 무작위 추출하여 SHAP 계산 진행
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(X_test_kobert.shape[0], size=min(200, X_test_kobert.shape[0]), replace=False)
    X_sample = X_test_kobert[sample_indices]
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    print("SHAP Value 계산 완료!")
    
    # 5. 피처 중요도 가공 (768개 KoBERT 텍스트 임베딩 차원을 하나로 병합)
    # shap_values의 shape은 (200, 962)
    # kobert_feature_names 리스트에서 'kobert_' 로 시작하는 피처들의 인덱스를 모읍니다.
    structured_indices = []
    kobert_indices = []
    
    for idx, name in enumerate(kobert_feature_names):
        if name.startswith('kobert_'):
            kobert_indices.append(idx)
        else:
            structured_indices.append(idx)
            
    # 정형 피처들의 평균 절대 SHAP 값 계산
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 정형 피처 이름 및 중요도 모음
    shap_summary = []
    for idx in structured_indices:
        shap_summary.append({
            'Feature': kobert_feature_names[idx],
            'Mean_Abs_SHAP': mean_abs_shap[idx],
            'Type': '정형 속성'
        })
        
    # 768개 KoBERT 차원의 중요도 합산
    kobert_shap_total = np.sum(mean_abs_shap[kobert_indices])
    shap_summary.append({
        'Feature': 'KoBERT 텍스트 임베딩 (768차원 병합)',
        'Mean_Abs_SHAP': kobert_shap_total,
        'Type': '텍스트 (KoBERT)'
    })
    
    shap_df = pd.DataFrame(shap_summary)
    shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    print("\n[KoBERT + XGBoost의 SHAP 분석 결과 (상위 15개 피처)]")
    print(shap_df.head(15))
    
    # 6. SHAP 결과 시각화 및 저장
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(shap_df))
    colors = ['#ff7f0e' if t == '텍스트 (KoBERT)' else '#1f77b4' for t in shap_df['Type'][:top_n]]
    
    sns.barplot(
        x='Mean_Abs_SHAP',
        y='Feature',
        data=shap_df.head(top_n),
        palette=colors if colors else 'viridis'
    )
    
    plt.title('KoBERT + XGBoost 모델의 SHAP 피처 중요도 분석 (텍스트 병합)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('평균 절대 SHAP 값 (시간 단위 기여도)', fontsize=12)
    plt.ylabel('피처명', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_img = 'shap_kobert_xgboost.png'
    plt.savefig(output_img, dpi=150)
    print(f"\nSHAP 분석 시각화 그래프가 '{output_img}'로 저장되었습니다.")
    
    # 7. 리포트에 내용 추가 제언 준비
    print("\n=========================================")
    print("SHAP 분석 완료!")

if __name__ == '__main__':
    run_shap_analysis()
