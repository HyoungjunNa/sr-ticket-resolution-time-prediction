"""Experiment configuration for ServiceNow SR resolution-time prediction.

Edit this file when the highlighted columns in the source spreadsheet change.
"""

# 기본 입력 데이터 파일입니다.
DATA_FILE = "sc_req_item.csv"

# 노란색(정형)으로 선정한 컬럼입니다. 범주형/숫자형을 모두 넣을 수 있습니다.
STRUCTURED_COLUMNS = [
    "Item",
    "Service",
    "State",
    "Stage",
    "Approval",
    "Assignment group",
    "Work time(Hour)",
    "Updates",
]

# 주황색(비정형)으로 선정한 텍스트 컬럼입니다. 여러 컬럼은 공백으로 합쳐서 사용합니다.
TEXT_COLUMNS = [
    "Short description",
    "Description",
    "HTML Description",
]

# 처리시간 타겟 생성에 사용할 원본 컬럼입니다.
# 원본 파일의 컬럼명이 실제로 "complte duration"이며, 단위는 seconds입니다.
TARGET_DURATION_SECONDS_COLUMN = "complte duration"

# 모델 학습에는 초 단위를 시간 단위로 변환한 내부 타겟을 사용합니다.
TARGET_COLUMN = "resolution_time_hours"

# Item별 분리 모델링 설정입니다.
ITEM_GROUP_COLUMN = "Item"
DURATION_OUTLIER_MAX_DAYS = 10
MIN_ITEM_SAMPLES = 50

# "complte duration" 컬럼이 없는 예제 데이터용 fallback 컬럼입니다.
OPENED_AT_COLUMN = "opened_at"
CLOSED_AT_COLUMN = "closed_at"

# 실험 설정
TEST_SIZE = 0.3
RANDOM_STATE = 42
MAX_TEXT_FEATURES = 100
BERT_MODEL_NAME = "kykim/bert-kor-base"
