import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLOSED_AT_COLUMN, OPENED_AT_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def _validate_required_columns(df, required_columns, file_path):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"데이터 파일 '{file_path}'에 필수 컬럼이 없습니다: {missing}. "
            f"현재 사용 가능한 컬럼: {available}. config.py에서 컬럼명을 실제 파일에 맞게 수정하세요."
        )


def load_and_split_data(
    file_path='sr_data.csv',
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    opened_at_col=OPENED_AT_COLUMN,
    closed_at_col=CLOSED_AT_COLUMN,
    target_col=TARGET_COLUMN,
):
    """
    ServiceNow SR 티켓 데이터를 로드하고, 처리 시간(타겟 변수)을 계산한 뒤,
    학습 데이터와 테스트 데이터로 분할합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다. 먼저 데이터를 준비해주세요.")

    df = pd.read_csv(file_path)
    _validate_required_columns(df, [opened_at_col, closed_at_col], file_path)

    # 날짜 데이터 파싱
    df[opened_at_col] = pd.to_datetime(df[opened_at_col], errors='coerce')
    df[closed_at_col] = pd.to_datetime(df[closed_at_col], errors='coerce')

    invalid_date_rows = df[df[[opened_at_col, closed_at_col]].isna().any(axis=1)]
    if not invalid_date_rows.empty:
        raise ValueError(
            f"{opened_at_col}/{closed_at_col} 날짜 파싱에 실패한 행이 {len(invalid_date_rows)}건 있습니다. "
            "원본 날짜 형식을 확인해주세요."
        )

    # 처리 완료 시간(Resolution Time) 계산 (단위: 시간)
    # closed_at - opened_at 차이를 시간(hours) 수치로 변환
    df[target_col] = (df[closed_at_col] - df[opened_at_col]).dt.total_seconds() / 3600.0
    df = df[df[target_col] >= 0].copy()

    # 학습 및 테스트 데이터 분할 (기본 70% Train, 30% Test)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    print(f"데이터 로드 및 분할 완료:")
    print(f" - 전체 데이터 크기: {len(df)}")
    print(f" - 학습 데이터 크기: {len(train_df)}")
    print(f" - 테스트 데이터 크기: {len(test_df)}")
    print(f" - 타겟 컬럼: {target_col} (시간 단위)")

    return train_df, test_df


if __name__ == '__main__':
    # 모듈 독립 실행 테스트
    try:
        train, test = load_and_split_data()
    except Exception as e:
        print("에러 발생:", e)
