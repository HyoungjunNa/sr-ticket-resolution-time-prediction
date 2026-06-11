import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(file_path='sr_data.csv', test_size=0.3, random_state=42):
    """
    ServiceNow SR 티켓 데이터를 로드하고, 처리 시간(타겟 변수)을 계산한 뒤,
    학습 데이터와 테스트 데이터로 분할합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다. 먼저 데이터를 준비해주세요.")
        
    df = pd.read_csv(file_path)
    
    # 날짜 데이터 파싱
    df['opened_at'] = pd.to_datetime(df['opened_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])
    
    # 처리 완료 시간(Resolution Time) 계산 (단위: 시간)
    # closed_at - opened_at 차이를 시간(hours) 수치로 변환
    df['resolution_time'] = (df['closed_at'] - df['opened_at']).dt.total_seconds() / 3600.0
    
    # 학습 및 테스트 데이터 분할 (70% Train, 30% Test)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"데이터 로드 및 분할 완료:")
    print(f" - 전체 데이터 크기: {len(df)}")
    print(f" - 학습 데이터 크기: {len(train_df)}")
    print(f" - 테스트 데이터 크기: {len(test_df)}")
    
    return train_df, test_df

if __name__ == '__main__':
    # 모듈 독립 실행 테스트
    try:
        train, test = load_and_split_data()
    except Exception as e:
        print("에러 발생:", e)
