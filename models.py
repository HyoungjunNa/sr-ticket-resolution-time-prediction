from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_regression_models(random_state=42):
    """
    비교 실험에 사용될 4가지 알고리즘 모델의 인스턴스를 반환합니다.
    """
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
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1 # 불필요한 로그 출력 방지
        )
    }
    return models
