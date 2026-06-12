from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_regression_models(scenario='TF-IDF', random_state=42):
    """
    비교 실험에 사용될 4가지 알고리즘 모델의 인스턴스를 반환합니다.
    시나리오별 최적 하이퍼파라미터를 다르게 세팅합니다.
    """
    # 1. XGBoost 파라미터 세팅
    if scenario == 'KoBERT':
        xgb_params = {
            'n_estimators': 188,
            'learning_rate': 0.06406,
            'max_depth': 6,
            'subsample': 0.961,
            'colsample_bytree': 0.904,
            'min_child_weight': 2,
            'gamma': 0.439
        }
    else: # TF-IDF (Optuna 최적화 결과 적용 - HTML 전처리 데이터 기반)
        xgb_params = {
            'n_estimators': 199,
            'learning_rate': 0.02795,
            'max_depth': 9,
            'subsample': 0.997,
            'colsample_bytree': 0.7398,
            'min_child_weight': 1,
            'gamma': 0.673
        }

    # 2. LightGBM 파라미터 세팅 (TF-IDF에서 튜닝된 최적값 우선 적용)
    # 필요시 시나리오별 구분 가능
    if scenario == 'KoBERT':
        lgb_params = {
            'n_estimators': 131,
            'learning_rate': 0.12616,
            'max_depth': 6,
            'num_leaves': 64,
            'min_child_samples': 5,
            'subsample': 0.824,
            'colsample_bytree': 0.894
        }
    else:
        lgb_params = {
            'n_estimators': 131,
            'learning_rate': 0.12616,
            'max_depth': 6,
            'num_leaves': 64,
            'min_child_samples': 5,
            'subsample': 0.824,
            'colsample_bytree': 0.894
        }

    models = {
        'Ridge': Ridge(
            alpha=1.0, 
            random_state=random_state
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            **xgb_params,
            random_state=random_state,
            n_jobs=-1,
            device='cuda'
        ),
        'LightGBM': LGBMRegressor(
            **lgb_params,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1 # 불필요한 로그 출력 방지
        )
    }
    return models

