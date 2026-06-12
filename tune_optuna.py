import os
import time
import argparse
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import DATA_FILE, RANDOM_STATE, TARGET_UNIT
from data_loader import load_and_split_data
from preprocess import SRDataPreprocessor

# Optuna 로그 레벨을 WARNING으로 설정하여 과도한 출력을 줄입니다.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_model(scenario='TF-IDF', n_trials=100, model_type='XGBoost'):
    """
    Optuna를 사용하여 전체 데이터셋에 대해 글로벌 모델의 하이퍼파라미터를 튜닝합니다.
    """
    print(f"\n=========================================")
    print(f" 글로벌 튜닝 시작 - 모델: {model_type} | 피처: {scenario}")
    print(f"=========================================")
    
    # 1. 글로벌 데이터 로드 및 분할 (학습 및 검증용)
    train_df, val_df, _ = load_and_split_data(DATA_FILE)
    print(f" - 전체 학습 데이터 크기 (N): {len(train_df)} 건")
    print(f" - 전체 검증 데이터 크기 (N): {len(val_df)} 건")
    
    # 2. 피처 추출 및 전처리
    print(" - 데이터 전처리 및 피처 추출 진행 중 (KoBERT 선택 시 시간이 조금 더 걸릴 수 있습니다)...")
    preprocessor = SRDataPreprocessor()
    X_tfidf_train, X_kobert_train, y_train = preprocessor.fit_transform(train_df)
    X_tfidf_val, X_kobert_val, y_val = preprocessor.transform(val_df)
    
    if scenario == 'TF-IDF':
        X_tr = X_tfidf_train
        X_va = X_tfidf_val
    elif scenario == 'KoBERT':
        X_tr = X_kobert_train
        X_va = X_kobert_val
    else:
        raise ValueError(f"알 수 없는 시나리오: {scenario}")
        
    print(f" - 학습 피처 차원: {X_tr.shape} | 검증 피처 차원: {X_va.shape}")
    
    # 3. 3-Fold 교차 검증 및 목적 함수 정의
    def objective(trial):
        if model_type == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'device': 'cuda'  # XGBoost GPU 가속 사용
            }
            model_class = XGBRegressor
        elif model_type == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            model_class = LGBMRegressor
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        model = model_class(**params)
        model.fit(X_tr, y_train)
        
        # 예측값은 음수가 될 수 없으므로 0으로 보정
        preds = np.maximum(model.predict(X_va), 0)
        mae = mean_absolute_error(y_val, preds)
        return mae

    # 4. Optuna 최적화 수행
    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    
    # 진행 상황 출력을 위해 callback 추가
    def callback(study, trial):
        UNIT_LABELS = {
            "hours": "시간",
            "minutes": "분",
            "seconds": "초"
        }
        unit_label = UNIT_LABELS.get(TARGET_UNIT, "시간")
        print(f" [Trial {trial.number:02d}] 현재 시도 MAE: {trial.value:.4f}{unit_label} | 최적 MAE: {study.best_value:.4f}{unit_label}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    elapsed = time.time() - start_time
    
    # 단위 변환
    if TARGET_UNIT == "minutes":
        mae_hours = study.best_value / 60.0
    elif TARGET_UNIT == "seconds":
        mae_hours = study.best_value / 3600.0
    else:
        mae_hours = study.best_value

    mae_seconds = mae_hours * 3600.0
    mae_minutes = mae_hours * 60.0
    
    UNIT_LABELS = {
        "hours": "시간",
        "minutes": "분",
        "seconds": "초"
    }
    unit_label = UNIT_LABELS.get(TARGET_UNIT, "시간")

    print("\n" + "="*40)
    print(f" 튜닝 완료! (소요 시간: {elapsed:.1f}초)")
    print(f" 최적 모델: {model_type}")
    print(f" 최적 검증 MAE: {study.best_value:.4f}{unit_label} (시간 환산: {mae_hours:.4f}시간 / {mae_minutes:.2f}분 / {mae_seconds:.0f}초)")
    print(f" 최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"   - {k}: {v}")
    print("="*40)
    
    return study.best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Global SR Predictor")
    parser.add_argument("--scenario", type=str, default="TF-IDF", choices=["TF-IDF", "KoBERT"], help="텍스트 시나리오 선택")
    parser.add_argument("--trials", type=int, default=30, help="텍스트 반복 횟수 (n_trials)")
    parser.add_argument("--model", type=str, default="XGBoost", choices=["XGBoost", "LightGBM"], help="튜닝할 모델 타입")
    
    args = parser.parse_args()
    
    tune_model(
        scenario=args.scenario,
        n_trials=args.trials,
        model_type=args.model
    )
