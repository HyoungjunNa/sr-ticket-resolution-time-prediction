import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from config import MAX_TEXT_FEATURES, STRUCTURED_COLUMNS, TARGET_COLUMN, TEXT_COLUMNS


def _build_one_hot_encoder():
    """scikit-learn 버전별 sparse 파라미터 차이를 흡수합니다."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')


class SRDataPreprocessor:
    def __init__(
        self,
        structured_cols=None,
        text_cols=None,
        target_col=TARGET_COLUMN,
        max_text_features=MAX_TEXT_FEATURES,
        bert_model_name='kykim/bert-kor-base',
        batch_size=32,
    ):
        self.structured_cols = structured_cols or STRUCTURED_COLUMNS
        self.text_cols = text_cols or TEXT_COLUMNS
        self.target_col = target_col
        self.max_text_features = max_text_features
        self.batch_size = batch_size

        self.structured_transformer = None
        self.structured_feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []

        # 1. TF-IDF Vectorizer 설정
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )

        # 2. BERT 임베딩을 위한 설정
        self.bert_model_name = bert_model_name
        self.device = None

        self.tokenizer = None
        self.bert_model = None

    def _validate_columns(self, df):
        required_cols = self.structured_cols + self.text_cols + [self.target_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            available = ", ".join(df.columns)
            raise ValueError(
                f"전처리에 필요한 컬럼이 없습니다: {missing}. "
                f"현재 사용 가능한 컬럼: {available}. config.py의 STRUCTURED_COLUMNS/TEXT_COLUMNS를 확인하세요."
            )

    def _split_structured_columns(self, df):
        structured_df = df[self.structured_cols]
        self.numeric_cols = structured_df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        self.categorical_cols = [col for col in self.structured_cols if col not in self.numeric_cols]

    def _fit_transform_structured(self, train_df):
        self._split_structured_columns(train_df)

        transformers = []
        if self.numeric_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])
            transformers.append(('num', numeric_pipeline, self.numeric_cols))

        if self.categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', _build_one_hot_encoder()),
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_cols))

        if not transformers:
            raise ValueError("STRUCTURED_COLUMNS가 비어 있습니다. 최소 1개 이상의 정형 컬럼을 지정하세요.")

        self.structured_transformer = ColumnTransformer(transformers=transformers, remainder='drop')
        X_structured = self.structured_transformer.fit_transform(train_df[self.structured_cols])
        X_structured = np.asarray(X_structured)
        self.structured_feature_names = self._get_structured_feature_names()
        return X_structured

    def _transform_structured(self, df):
        X_structured = self.structured_transformer.transform(df[self.structured_cols])
        return np.asarray(X_structured)

    def _get_structured_feature_names(self):
        feature_names = []
        if self.numeric_cols:
            feature_names.extend(self.numeric_cols)
        if self.categorical_cols:
            cat_pipeline = self.structured_transformer.named_transformers_['cat']
            onehot = cat_pipeline.named_steps['onehot']
            feature_names.extend(onehot.get_feature_names_out(self.categorical_cols).tolist())
        return feature_names

    def _merge_texts(self, df):
        return df[self.text_cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()

    def _init_bert(self):
        """BERT 모델과 토크나이저를 지연 로딩(Lazy Loading)합니다."""
        if self.tokenizer is None or self.bert_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

            import torch

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # CLS 토큰의 hidden state 추출 (batch_size, hidden_size)
                # kykim/bert-kor-base 모델의 hidden_size는 768차원입니다.
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

        return np.vstack(embeddings)


    def fit_transform_tfidf(self, train_df):
        """피처 중요도/빠른 리포트용: 정형 + TF-IDF 피처만 생성합니다."""
        self._validate_columns(train_df)
        X_structured = self._fit_transform_structured(train_df)
        train_texts = self._merge_texts(train_df)
        X_tfidf_text = self.tfidf_vectorizer.fit_transform(train_texts).toarray()
        self.tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        y = train_df[self.target_col].values
        return np.hstack([X_structured, X_tfidf_text]), y

    def fit_transform(self, train_df):
        """
        학습 데이터를 기반으로 피처 변환기들을 fit하고 변환한 후, 두 가지 시나리오를 반환합니다.
        1. X_tfidf: 정형 피처 + TF-IDF 텍스트 피처
        2. X_kobert: 정형 피처 + KoBERT 임베딩 피처
        """
        self._validate_columns(train_df)

        # 1. 정형 피처 인코딩 (두 시나리오 공통)
        X_structured = self._fit_transform_structured(train_df)

        # 텍스트 병합 (설정된 모든 비정형 컬럼)
        train_texts = self._merge_texts(train_df)

        # 2. TF-IDF 피처 추출
        X_tfidf_text = self.tfidf_vectorizer.fit_transform(train_texts).toarray()
        self.tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]

        # 3. BERT 임베딩 피처 추출
        X_kobert_text = self._get_bert_embeddings(train_texts)
        self.kobert_feature_names = [f"kobert_{i}" for i in range(X_kobert_text.shape[1])]

        # 4. 결합 피처 생성
        X_tfidf = np.hstack([X_structured, X_tfidf_text])
        X_kobert = np.hstack([X_structured, X_kobert_text])

        # 타겟 값 (처리 완료 시간)
        y = train_df[self.target_col].values

        return X_tfidf, X_kobert, y

    def transform(self, test_df):
        """
        테스트 데이터를 기존에 fit된 변환기들을 기반으로 변환합니다.
        """
        self._validate_columns(test_df)

        # 1. 정형 피처 인코딩
        X_structured = self._transform_structured(test_df)

        # 텍스트 병합
        test_texts = self._merge_texts(test_df)

        # 2. TF-IDF 피처 추출
        X_tfidf_text = self.tfidf_vectorizer.transform(test_texts).toarray()

        # 3. BERT 임베딩 피처 추출
        X_kobert_text = self._get_bert_embeddings(test_texts)

        # 4. 결합 피처 생성
        X_tfidf = np.hstack([X_structured, X_tfidf_text])
        X_kobert = np.hstack([X_structured, X_kobert_text])

        y = test_df[self.target_col].values

        return X_tfidf, X_kobert, y

    def get_feature_names(self, mode='tfidf'):
        if mode == 'tfidf':
            return self.structured_feature_names + self.tfidf_feature_names
        if mode == 'kobert':
            return self.structured_feature_names + self.kobert_feature_names
        raise ValueError("mode는 'tfidf' 또는 'kobert'여야 합니다.")
