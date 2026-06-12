import os
import sys
import numpy as np
import pandas as pd
from data_loader import load_and_split_data
from preprocess import SRDataPreprocessor
from models import get_regression_models
from config import DATA_FILE, RANDOM_STATE, BERT_MODEL_NAME, MAX_TEXT_FEATURES

# Windows에서 출력 인코딩 충돌 방지
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def analyze_word_importance():
    print("=========================================")
    print("KoBERT + XGBoost 단어 수준 영향도 분석 시작")
    print("=========================================")
    
    # 1. 데이터 로드 및 모델 학습
    train_df, _, test_df = load_and_split_data(DATA_FILE)
    preprocessor = SRDataPreprocessor(max_text_features=MAX_TEXT_FEATURES, bert_model_name=BERT_MODEL_NAME)
    
    print("KoBERT 피처 추출 중...")
    _, X_train_kobert, y_train = preprocessor.fit_transform(train_df)
    
    print("XGBoost 모델 학습 중...")
    models = get_regression_models(scenario='KoBERT', random_state=RANDOM_STATE)
    xgb_model = models['XGBoost']
    xgb_model.fit(X_train_kobert, y_train)
    
    # 2. 테스트 데이터에서 분석할 흥미로운 티켓 선택
    # 예측 처리 시간이 아주 길게 나온 상위 티켓들 중 텍스트가 풍부한 것을 고릅니다.
    print("테스트 데이터 예측 수행 및 분석 티켓 선정 중...")
    _, X_test_kobert, y_test = preprocessor.transform(test_df)
    test_preds = np.maximum(xgb_model.predict(X_test_kobert), 0)
    
    # 실제 처리 시간과 예측 시간이 모두 긴 티켓 인덱스 정렬
    test_df = test_df.copy()
    test_df['pred_hours'] = test_preds
    test_df['actual_hours'] = y_test
    
    # 텍스트 단어 수가 5개 이상이고 예측 시간이 24시간 이상인 티켓들 필터링
    rich_text_df = test_df[
        (test_df['short_description'].str.split().str.len() >= 4) & 
        (test_df['pred_hours'] > 15)
    ].sort_values(by='pred_hours', ascending=False)
    
    if len(rich_text_df) == 0:
        print("조건에 맞는 티켓이 없어 전체 테스트 데이터 중 상위 티켓을 선택합니다.")
        rich_text_df = test_df.sort_values(by='pred_hours', ascending=False)
        
    # 상위 3개 티켓에 대해 단어별 Perturbation 분석 진행
    target_tickets = rich_text_df.head(3)
    
    for idx, (t_idx, ticket) in enumerate(target_tickets.iterrows()):
        print(f"\n-----------------------------------------")
        print(f"[*] 분석 대상 티켓 {idx+1} (실제 인덱스: {t_idx})")
        print(f" - 신청 구분 (cat_item): {ticket['cat_item']}")
        print(f" - 담당 부서 (closed_by): {ticket['closed_by']}")
        print(f" - 제목 (short_description): {ticket['short_description']}")
        print(f" - 본문 요약 (merged_description): {str(ticket['merged_description'])[:60]}...")
        print(f" - [모델 예측 완료시간]: {ticket['pred_hours']:.2f}시간 | [실제 소요시간]: {ticket['actual_hours']:.2f}시간")
        
        # 텍스트 병합 및 단어 분할
        # KoBERT 모델이 입력받는 최종 병합 텍스트를 기준으로 띄어쓰기 단위로 쪼갭니다.
        full_text = f"{ticket['short_description']} {ticket['merged_description']}".strip()
        words = [w for w in full_text.split() if len(w) > 1] # 1글자짜리 조사/기호 등 제외
        words = list(dict.fromkeys(words))[:15] # 중복 단어 제거 및 최대 15개 단어만 분석
        
        if not words:
            print("분석할 단어가 없습니다.")
            continue
            
        print(f" - 분석 대상 단어 목록 ({len(words)}개): {words}")
        
        # 가상의 데이터프레임 빌드 (단어를 하나씩 지운 버전을 만듭니다.)
        perturbed_rows = []
        
        # 1) 원본 행 추가
        perturbed_rows.append(ticket.copy())
        
        # 2) 각 단어를 하나씩 제외한 행들 추가
        for word in words:
            temp_ticket = ticket.copy()
            # 단어 제거
            temp_short = " ".join([w for w in str(ticket['short_description']).split() if w != word])
            temp_merged = " ".join([w for w in str(ticket['merged_description']).split() if w != word])
            temp_ticket['short_description'] = temp_short
            temp_ticket['merged_description'] = temp_merged
            perturbed_rows.append(temp_ticket)
            
        perturbed_df = pd.DataFrame(perturbed_rows)
        
        # 피처 변환 및 예측
        _, X_pert_kobert, _ = preprocessor.transform(perturbed_df)
        pert_preds = np.maximum(xgb_model.predict(X_pert_kobert), 0)
        
        orig_pred = pert_preds[0]
        word_preds = pert_preds[1:]
        
        # 단어 기여도 계산 (단어가 빠졌을 때 예측값이 얼마나 줄어드는지)
        # 즉, (원본 예측치) - (단어 제외 예측치) = 해당 단어의 기여도
        # 양수 값이 클수록 "이 단어가 들어감으로써 처리 시간이 길어지게 예측했다"는 뜻입니다.
        contributions = []
        for word, p_pred in zip(words, word_preds):
            contrib = orig_pred - p_pred
            contributions.append({
                'Word': word,
                'Contribution_Hours': contrib,
                'Prediction_Without_Word': p_pred
            })
            
        contrib_df = pd.DataFrame(contributions)
        contrib_df = contrib_df.sort_values(by='Contribution_Hours', ascending=False).reset_index(drop=True)
        
        print(f"\n >> [단어별 영향도 결과]")
        for c_idx, row in contrib_df.iterrows():
            mark = "[지연 요인]" if row['Contribution_Hours'] > 0 else "[단축 요인]"
            print(f"    {c_idx+1}. '{row['Word']}' ({mark}): {row['Contribution_Hours']:+.2f}시간 (제외 시 예측: {row['Prediction_Without_Word']:.2f}시간)")

if __name__ == '__main__':
    try:
        analyze_word_importance()
    except Exception as e:
        print("에러 발생:", e)

