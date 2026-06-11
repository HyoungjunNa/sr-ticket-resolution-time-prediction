"""Experiment configuration for ServiceNow SR resolution-time prediction.

Edit this file when the highlighted columns in the source spreadsheet change.
"""

# 노란색(정형)으로 선정한 컬럼입니다. 범주형/숫자형을 모두 넣을 수 있습니다.
STRUCTURED_COLUMNS = [
    "category",
    "priority",
    "assignment_group",
]

# 주황색(비정형)으로 선정한 텍스트 컬럼입니다. 여러 컬럼은 공백으로 합쳐서 사용합니다.
TEXT_COLUMNS = [
    "short_description",
    "description",
]

# 처리시간 타겟 생성에 사용할 시작/종료 시각 컬럼입니다.
OPENED_AT_COLUMN = "opened_at"
CLOSED_AT_COLUMN = "closed_at"
TARGET_COLUMN = "resolution_time"

# 실험 설정
TEST_SIZE = 0.3
RANDOM_STATE = 42
MAX_TEXT_FEATURES = 100
BERT_MODEL_NAME = "kykim/bert-kor-base"
