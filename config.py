"""Experiment configuration for ServiceNow SR resolution-time prediction.

Edit this file when the highlighted columns in the source spreadsheet change.
"""

# 기본 입력 데이터 파일입니다.
DATA_FILE = "sc_req_item.csv"

# 노란색(정형)으로 선정한 컬럼입니다. 범주형/숫자형을 모두 넣을 수 있습니다.
STRUCTURED_COLUMNS = [
    "cat_item",
    "business_service",
    "sys_domain",
    "closed_by",
    "Created_day_of_week",
    "Created_hour",
    "Created_is_weekend",
]

# 주황색(비정형)으로 선정한 텍스트 컬럼입니다. 여러 컬럼은 공백으로 합쳐서 사용합니다.
TEXT_COLUMNS = [
    "short_description",
    "merged_description",
]

# 처리시간 타겟 생성에 사용할 원본 컬럼입니다.
# 원본 파일의 컬럼명이 실제로 "u_rpt_complte_duration"이며, 단위는 seconds입니다.
TARGET_DURATION_SECONDS_COLUMN = "u_rpt_complte_duration"

# 모델 학습에는 초 단위를 시간 단위로 변환한 내부 타겟을 사용합니다.
TARGET_COLUMN = "resolution_time_hours"

# 타겟 변수 예측 단위 설정 ("hours", "minutes", "seconds" 중 선택)
TARGET_UNIT = "hours"

DURATION_OUTLIER_MAX_DAYS = 100

# "complte duration" 컬럼이 없는 예제 데이터용 fallback 컬럼입니다.
OPENED_AT_COLUMN = "opened_at"
CLOSED_AT_COLUMN = "closed_at"

# 실험 설정
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
RANDOM_STATE = 42
MAX_TEXT_FEATURES = 100
BERT_MODEL_NAME = "kykim/bert-kor-base"

# 실제 업무 시간(Business Hours: 평일 09:00~18:00) 기준 모델링 여부
USE_BUSINESS_HOURS = True

# 타겟 변수 로그 변환 (np.log1p) 적용 여부 (결정계수 및 모델 안정성 향상)
USE_LOG_TARGET = True

# 특정 Item들만 지정하여 실험하고 싶을 때 리스트로 입력합니다 (None이면 전체 실행).
ONLY_EXPERIMENT_ITEMS = None
