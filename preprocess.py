import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class SRDataPreprocessor:
    def __init__(self, max_text_features=100, bert_model_name='kykim/bert-kor-base', batch_size=32):
        self.categorical_cols = ['category', 'priority', 'assignment_group']
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # 1. TF-IDF Vectorizer 설정
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )
        
        # 2. BERT 임베딩을 위한 설정
        self.bert_model_name = bert_model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.bert_model = None
        
    def _init_bert(self):
        """BERT 모델과 토크나이저를 지연 로딩(Lazy Loading)합니다."""
        if self.tokenizer is None or self.bert_model is None:
            print(f"KoBERT 모델({self.bert_model_name}) 로드 중... (디바이스: {self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
            self.bert_model.eval()

    def _get_bert_embeddings(self, texts):
        """텍스트 리스트로부터 BERT CLS 토큰 임베딩을 추출합니다."""
        self._init_bert()
        embeddings = []
        
        # 배치 단위로 처리하여 메모리 효율성 도모
        for i in tqdm(range(0, len(texts), self.batch_size), desc="KoBERT 임베딩 추출"):
            batch_texts = texts[i:i + self.batch_size]
            
            # 토큰화 및 패딩
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # CLS 토큰의 hidden state 추출 (batch_size, hidden_size)
                # kykim/bert-kor-base 모델의 hidden_size는 768차원입니다.
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
                
        return np.vstack(embeddings)

    def fit_transform(self, train_df):
        """
        학습 데이터를 기반으로 피처 변환기들을 fit하고 변환한 후, 두 가지 시나리오를 반환합니다.
        1. X_tfidf: 정형 피처 + TF-IDF 텍스트 피처
        2. X_kobert: 정형 피처 + KoBERT 임베딩 피처
        """
        # 1. 정형 피처 인코딩 (두 시나리오 공통)
        X_cat = self.encoder.fit_transform(train_df[self.categorical_cols])
        self.categorical_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        
        # 텍스트 병합 (제목 + 본문)
        train_texts = (train_df['short_description'].fillna('') + ' ' + train_df['description'].fillna('')).tolist()
        
        # 2. TF-IDF 피처 추출
        X_tfidf_text = self.tfidf_vectorizer.fit_transform(train_texts).toarray()
        self.tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        
        # 3. BERT 임베딩 피처 추출
        X_kobert_text = self._get_bert_embeddings(train_texts)
        self.kobert_feature_names = [f"kobert_{i}" for i in range(X_kobert_text.shape[1])]
        
        # 4. 결합 피처 생성
        X_tfidf = np.hstack([X_cat, X_tfidf_text])
        X_kobert = np.hstack([X_cat, X_kobert_text])
        
        # 타겟 값 (처리 완료 시간)
        y = train_df['resolution_time'].values
        
        return X_tfidf, X_kobert, y
        
    def transform(self, test_df):
        """
        테스트 데이터를 기존에 fit된 변환기들을 기반으로 변환합니다.
        """
        # 1. 정형 피처 인코딩
        X_cat = self.encoder.transform(test_df[self.categorical_cols])
        
        # 텍스트 병합
        test_texts = (test_df['short_description'].fillna('') + ' ' + test_df['description'].fillna('')).tolist()
        
        # 2. TF-IDF 피처 추출
        X_tfidf_text = self.tfidf_vectorizer.transform(test_texts).toarray()
        
        # 3. BERT 임베딩 피처 추출
        X_kobert_text = self._get_bert_embeddings(test_texts)
        
        # 4. 결합 피처 생성
        X_tfidf = np.hstack([X_cat, X_tfidf_text])
        X_kobert = np.hstack([X_cat, X_kobert_text])
        
        y = test_df['resolution_time'].values
        
        return X_tfidf, X_kobert, y

    def get_feature_names(self, mode='tfidf'):
        if mode == 'tfidf':
            return self.categorical_feature_names + self.tfidf_feature_names
        elif mode == 'kobert':
            return self.categorical_feature_names + self.kobert_feature_names
        else:
            raise ValueError("mode는 'tfidf' 또는 'kobert'여야 합니다.")
