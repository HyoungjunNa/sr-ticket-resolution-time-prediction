import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import load_and_split_data
from preprocess import SRDataPreprocessor
from models import get_regression_models
from config import BERT_MODEL_NAME, DATA_FILE, MAX_TEXT_FEATURES

def run_experiments(data_path=DATA_FILE, output_path='experiment_results.csv'):
    # 1. 데이터 로드 및 70:30 분할
    train_df, test_df = load_and_split_data(data_path)
    
    # 2. 전처리 모듈 초기화 (TF-IDF 및 KoBERT 추출 준비)
    preprocessor = SRDataPreprocessor(max_text_features=MAX_TEXT_FEATURES, bert_model_name=BERT_MODEL_NAME)
    
    # X_tfidf_*: 정형 + TF-IDF 피처
    # X_kobert_*: 정형 + KoBERT 피처
    X_train_tfidf, X_train_kobert, y_train = preprocessor.fit_transform(train_df)
    X_test_tfidf, X_test_kobert, y_test = preprocessor.transform(test_df)
    
    scenarios = {
        'TF-IDF': {
            'X_train': X_train_tfidf,
            'X_test': X_test_tfidf
        },
        'KoBERT': {
            'X_train': X_train_kobert,
            'X_test': X_test_kobert
        }
    }
    
    results = []
    
    # 3. 실험 매트릭스 순회 (2가지 전처리 x 4가지 알고리즘)
    for scenario_name, data_dict in scenarios.items():
        X_tr = data_dict['X_train']
        X_te = data_dict['X_test']
        
        print(f"\n===== 시나리오 시작: {scenario_name} (입력 피처 차원: {X_tr.shape[1]}) =====")
        
        models = get_regression_models()
        
        for model_name, model in models.items():
            print(f"[{scenario_name} + {model_name}] 모델 학습 및 평가 진행 중...")
            
            # 학습 시간 정밀 계측
            start_time = time.time()
            model.fit(X_tr, y_train)
            end_time = time.time()
            
            # 예측
            y_pred = model.predict(X_te)
            
            # 소요 시간 포맷팅 (초/분 단위)
            elapsed_seconds = end_time - start_time
            if elapsed_seconds < 60.0:
                training_time_str = f"{elapsed_seconds:.1f}초"
            else:
                training_time_str = f"{elapsed_seconds / 60.0:.1f}분"
                
            # 평가지표 계산
            # y는 원본 complete duration(seconds)을 hours로 변환한 값입니다.
            # 보고서 가독성을 위해 seconds/minutes/hours 지표를 함께 저장합니다.
            mae_hours = mean_absolute_error(y_test, y_pred)
            mae_seconds = mae_hours * 3600.0
            mae_minutes = mae_hours * 60.0
            
            # R2 Score 및 RMSE
            rmse_hours = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_seconds = rmse_hours * 3600.0
            rmse_minutes = rmse_hours * 60.0
            r2 = r2_score(y_test, y_pred)
            
            print(
                f"결과 - MAE: {mae_minutes:.2f}분 ({mae_seconds:.0f}초) "
                f"| R2: {r2:.4f} | 학습시간: {training_time_str}"
            )
            
            # 결과 기록
            results.append({
                'Scenario': scenario_name,
                'Model': model_name,
                'MAE_sec': mae_seconds,
                'MAE_min': mae_minutes,
                'MAE_hour': mae_hours,
                'RMSE_sec': rmse_seconds,
                'RMSE_min': rmse_minutes,
                'RMSE_hour': rmse_hours,
                'R2': r2,
                'TrainTime': training_time_str,
                'ElapsedSeconds': elapsed_seconds
            })
            
    # 4. 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n모든 비교 실험 완료! 상세 결과가 '{output_path}'에 저장되었습니다.")
    
    return results_df

if __name__ == '__main__':
    try:
        run_experiments()
    except Exception as e:
        print("실험 중 에러 발생:", e)
