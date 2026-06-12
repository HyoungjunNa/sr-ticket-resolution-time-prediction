import os
import re
from html import unescape

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLOSED_AT_COLUMN,
    DATA_FILE,
    DURATION_OUTLIER_MAX_DAYS,
    OPENED_AT_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_DURATION_SECONDS_COLUMN,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    USE_BUSINESS_HOURS,
    TARGET_UNIT,
)


def clean_html(text):
    if not isinstance(text, str):
        return ""
    # HTML 엔티티 제거 (예: &nbsp;, &lt;, &gt; 등)
    text = unescape(text)
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)
    # 연속된 공백 및 줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_business_hours(start_series, end_series):
    """
    start_series와 end_series (datetime64) 사이의 평일 업무 시간(09:00 ~ 18:00)을 시간 단위(Hour)로 계산합니다.
    """
    results = []
    for start, end in zip(start_series, end_series):
        if pd.isna(start) or pd.isna(end) or start > end:
            results.append(0.0)
            continue
        
        start_date = start.date()
        end_date = end.date()
        
        total_hours = 0.0
        
        # start와 end가 같은 날인 경우
        if start_date == end_date:
            if start.weekday() < 5:  # 평일(월~금)
                s_hour = max(9.0, min(18.0, start.hour + start.minute / 60.0 + start.second / 3600.0))
                e_hour = max(9.0, min(18.0, end.hour + end.minute / 60.0 + end.second / 3600.0))
                total_hours = max(0.0, e_hour - s_hour)
        else:
            # 첫날 계산
            if start.weekday() < 5:
                s_hour = max(9.0, min(18.0, start.hour + start.minute / 60.0 + start.second / 3600.0))
                total_hours += max(0.0, 18.0 - s_hour)
            
            # 마지막날 계산
            if end.weekday() < 5:
                e_hour = max(9.0, min(18.0, end.hour + end.minute / 60.0 + end.second / 3600.0))
                total_hours += max(0.0, e_hour - 9.0)
            
            # 중간 날짜들 계산 (start 다음날부터 end 전날까지)
            if (end_date - start_date).days > 1:
                mid_days = pd.bdate_range(start=start_date + pd.Timedelta(days=1), 
                                          end=end_date - pd.Timedelta(days=1))
                total_hours += len(mid_days) * 9.0  # 하루 9시간 근무
                
        results.append(total_hours)
    return pd.Series(results, index=start_series.index)


def _validate_required_columns(df, required_columns, file_path):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"데이터 파일 '{file_path}'에 필수 컬럼이 없습니다: {missing}. "
            f"현재 사용 가능한 컬럼: {available}. config.py에서 컬럼명을 실제 파일에 맞게 수정하세요."
        )


def load_and_split_data(
    file_path=DATA_FILE,
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    duration_seconds_col=TARGET_DURATION_SECONDS_COLUMN,
    opened_at_col=OPENED_AT_COLUMN,
    closed_at_col=CLOSED_AT_COLUMN,
    target_col=TARGET_COLUMN,
):
    """
    ServiceNow SR 티켓 데이터를 로드하고, 처리 시간(타겟 변수)을 준비한 뒤,
    학습 데이터와 테스트 데이터로 분할합니다.

    USE_BUSINESS_HOURS가 True인 경우 평일 업무 시간(09:00 ~ 18:00) 기준으로 변환하며,
    그렇지 않으면 원본의 `complte duration` 컬럼을 사용합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다. 먼저 데이터를 준비해주세요.")

    df = pd.read_csv(file_path)
    
    # description과 u_common_html_description 컬럼을 하나의 컬럼으로 병합 후 HTML 제거
    merged = (
        df['description'].fillna('').astype(str) + ' ' + 
        df['u_common_html_description'].fillna('').astype(str)
    )
    df['merged_description'] = merged.apply(clean_html)

    # sys_created_on 및 closed_at 컬럼 파싱
    if 'sys_created_on' in df.columns:
        df['Created_datetime'] = pd.to_datetime(df['sys_created_on'], errors='coerce')
        df['Created_day_of_week'] = df['Created_datetime'].dt.day_name()
        df['Created_hour'] = df['Created_datetime'].dt.hour
        df['Created_is_weekend'] = df['Created_datetime'].dt.dayofweek.isin([5, 6]).astype(int)

    if 'closed_at' in df.columns:
        df['Closed_datetime'] = pd.to_datetime(df['closed_at'], errors='coerce')

    # 실제 업무 시간(Business Hours) 여부에 따라 타겟 컬럼 계산
    if USE_BUSINESS_HOURS and 'Created_datetime' in df.columns and 'Closed_datetime' in df.columns:
        print(" - 실제 처리 시간(Business Hours: 평일 09~18시) 기준으로 타겟 변수를 계산합니다.")
        df[target_col] = calculate_business_hours(df['Created_datetime'], df['Closed_datetime'])
        target_source_text = "Closed - Created (Business Hours)"
        df['duration_days'] = df[target_col] / 9.0  # 업무 시간 기준 하루는 9시간
    else:
        if duration_seconds_col in df.columns:
            duration_seconds = (
                df[duration_seconds_col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
                .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
            )
            df[target_col] = pd.to_numeric(duration_seconds, errors='coerce') / 3600.0

            invalid_target_rows = df[df[target_col].isna()]
            if not invalid_target_rows.empty:
                raise ValueError(
                    f"{duration_seconds_col} 컬럼을 숫자(seconds)로 변환할 수 없는 행이 "
                    f"{len(invalid_target_rows)}건 있습니다. 원본 값을 확인해주세요."
                )
            target_source_text = f"{duration_seconds_col} (seconds -> hours)"
        else:
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

            df[target_col] = (df[closed_at_col] - df[opened_at_col]).dt.total_seconds() / 3600.0
            target_source_text = f"{closed_at_col} - {opened_at_col} (seconds -> hours)"

        df['duration_days'] = df[target_col] / 24.0
    before_filter = len(df)
    df = df[df['duration_days'] <= DURATION_OUTLIER_MAX_DAYS].copy()
    after_filter = len(df)
    print(f" - 이상치 제거 전 데이터 수: {before_filter}")
    print(f" - {DURATION_OUTLIER_MAX_DAYS}일 초과 이상치 제거 후 데이터 수: {after_filter} (제거됨: {before_filter - after_filter}건)")

    # 선택한 단위(TARGET_UNIT)에 따른 스케일링 변환
    if TARGET_UNIT == "minutes":
        df[target_col] = df[target_col] * 60.0
        print(f" - [스케일링] 타겟 예측 단위를 '분(minutes)'으로 설정하여 변환 완료")
    elif TARGET_UNIT == "seconds":
        df[target_col] = df[target_col] * 3600.0
        print(f" - [스케일링] 타겟 예측 단위를 '초(seconds)'으로 설정하여 변환 완료")
    else:
        print(f" - [스케일링] 타겟 예측 단위를 '시간(hours)'으로 설정하여 변환 완료")

    # 학습, 검증, 테스트 데이터 분할 (70% Train, 20% Val, 10% Test)
    temp_size = 1.0 - train_size
    train_df, temp_df = train_test_split(df, test_size=temp_size, random_state=random_state)
    
    test_ratio_in_temp = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio_in_temp, random_state=random_state)

    print(f"데이터 로드 및 분할 완료:")
    print(f" - 전체 데이터 크기: {len(df)}")
    print(f" - 학습 데이터 크기: {len(train_df)}")
    print(f" - 검증 데이터 크기: {len(val_df)}")
    print(f" - 테스트 데이터 크기: {len(test_df)}")
    print(f" - 타겟 원본: {target_source_text}")
    print(f" - 모델 타겟 컬럼: {target_col} ({TARGET_UNIT})")

    return train_df, val_df, test_df


if __name__ == '__main__':
    # 모듈 독립 실행 테스트
    try:
        train, val, test = load_and_split_data()
    except Exception as e:
        print("에러 발생:", e)
