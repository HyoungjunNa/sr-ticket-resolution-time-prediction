import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from data_loader import load_and_split_data
from preprocess import SRDataPreprocessor
from models import get_regression_models
from config import BERT_MODEL_NAME, DATA_FILE, MAX_TEXT_FEATURES, TARGET_UNIT, RANDOM_STATE, USE_LOG_TARGET

def run_experiments(data_path=DATA_FILE, output_path='experiment_results.csv'):
    # 1. 데이터 로드 및 전체 결합
    train_df, val_df, test_df = load_and_split_data(data_path)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"\n[5-Fold 교차 검증] 전체 데이터 건수: {len(df)} 건")
    
    # 2. 전처리 모듈 초기화 및 전체 데이터 변환
    preprocessor = SRDataPreprocessor(max_text_features=MAX_TEXT_FEATURES, bert_model_name=BERT_MODEL_NAME)
    
    # X_tfidf_all: 전체 정형 + TF-IDF 피처
    # X_kobert_all: 전체 정형 + KoBERT 피처
    X_tfidf_all, X_kobert_all, y_all = preprocessor.fit_transform(df)
    
    scenarios = {
        'TF-IDF': X_tfidf_all,
        'KoBERT': X_kobert_all
    }
    
    # K-Fold 교차 검증 설정 (5-Fold)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    # 3. 실험 매트릭스 순회 (2가지 전처리 x 4가지 알고리즘)
    for scenario_name, X_all in scenarios.items():
        print(f"\n===== 시나리오 시작: {scenario_name} (입력 피처 차원: {X_all.shape[1]}) =====")
        
        # 각 시나리오별 모델 리스트 가져오기
        models = get_regression_models(scenario=scenario_name, random_state=RANDOM_STATE)
        
        for model_name, model in models.items():
            print(f"[{scenario_name} + {model_name}] 5-Fold 교차 검증 평가 진행 중...")
            
            fold_maes = []
            fold_rmses = []
            fold_r2s = []
            fold_r2s_log = []
            fold_times = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
                X_tr, X_te = X_all[train_idx], X_all[test_idx]
                y_train_fold, y_test_fold = y_all[train_idx], y_all[test_idx]
                
                # 학습 시간 정밀 계측
                start_time = time.time()
                if USE_LOG_TARGET:
                    model.fit(X_tr, np.log1p(y_train_fold))
                else:
                    model.fit(X_tr, y_train_fold)
                end_time = time.time()
                
                # 예측
                y_pred = model.predict(X_te)
                
                # 예측치를 로그 역변환하여 원본 스케일 평가
                if USE_LOG_TARGET:
                    y_pred_log = y_pred.copy()
                    y_pred_orig = np.expm1(y_pred_log)
                    y_pred_orig = np.maximum(y_pred_orig, 0)
                    
                    # 로그 스케일에서의 R2 계산
                    y_test_log = np.log1p(y_test_fold)
                    r2_log = r2_score(y_test_log, y_pred_log)
                else:
                    y_pred_orig = y_pred
                    r2_log = np.nan
                
                # 예측치와 실측치를 평가를 위해 시간(hours) 단위로 임시 변환
                if TARGET_UNIT == "minutes":
                    y_test_hours = y_test_fold / 60.0
                    y_pred_hours = y_pred_orig / 60.0
                elif TARGET_UNIT == "seconds":
                    y_test_hours = y_test_fold / 3600.0
                    y_pred_hours = y_pred_orig / 3600.0
                else:
                    y_test_hours = y_test_fold
                    y_pred_hours = y_pred_orig
                    
                # 음수 예측 보정
                y_pred_hours = np.maximum(y_pred_hours, 0)
                
                # 평가지표 계산
                mae_hours = mean_absolute_error(y_test_hours, y_pred_hours)
                rmse_hours = np.sqrt(mean_squared_error(y_test_hours, y_pred_hours))
                r2 = r2_score(y_test_hours, y_pred_hours)
                
                fold_maes.append(mae_hours)
                fold_rmses.append(rmse_hours)
                fold_r2s.append(r2)
                fold_r2s_log.append(r2_log)
                fold_times.append(end_time - start_time)
                
            # 5개 Fold의 평균값 계산
            mean_mae_hours = np.mean(fold_maes)
            mean_mae_seconds = mean_mae_hours * 3600.0
            mean_mae_minutes = mean_mae_hours * 60.0
            
            mean_rmse_hours = np.mean(fold_rmses)
            mean_rmse_seconds = mean_rmse_hours * 3600.0
            mean_rmse_minutes = mean_rmse_hours * 60.0
            
            mean_r2 = np.mean(fold_r2s)
            mean_r2_log = np.mean(fold_r2s_log) if USE_LOG_TARGET else np.nan
            mean_time = np.mean(fold_times)
            
            if mean_time < 60.0:
                training_time_str = f"{mean_time:.1f}초"
            else:
                training_time_str = f"{mean_time / 60.0:.1f}분"
                
            log_r2_str = f" (Log R2: {mean_r2_log:.4f})" if USE_LOG_TARGET else ""
            print(
                f"결과(평균) - MAE: {mean_mae_minutes:.2f}분 ({mean_mae_seconds:.0f}초) "
                f"| R2: {mean_r2:.4f}{log_r2_str} | 평균 학습시간: {training_time_str}"
            )
            
            # 결과 기록
            results.append({
                'Scenario': scenario_name,
                'Model': model_name,
                'MAE_sec': mean_mae_seconds,
                'MAE_min': mean_mae_minutes,
                'MAE_hour': mean_mae_hours,
                'RMSE_sec': mean_rmse_seconds,
                'RMSE_min': mean_rmse_minutes,
                'RMSE_hour': mean_rmse_hours,
                'R2': mean_r2,
                'R2_log': mean_r2_log,
                'TrainTime': training_time_str,
                'ElapsedSeconds': mean_time
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
