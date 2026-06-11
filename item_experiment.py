import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    BERT_MODEL_NAME,
    DATA_FILE,
    DURATION_OUTLIER_MAX_DAYS,
    ITEM_GROUP_COLUMN,
    MAX_TEXT_FEATURES,
    MIN_ITEM_SAMPLES,
    RANDOM_STATE,
    STRUCTURED_COLUMNS,
    TARGET_COLUMN,
    TARGET_DURATION_SECONDS_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
)
from models import get_regression_models
from preprocess import SRDataPreprocessor

plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _parse_duration_seconds(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '', regex=False).str.strip(),
        errors='coerce',
    )


def load_filtered_item_data(data_path=DATA_FILE):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일 '{data_path}'을 찾을 수 없습니다.")

    df = pd.read_csv(data_path)
    missing = [
        col for col in [ITEM_GROUP_COLUMN, TARGET_DURATION_SECONDS_COLUMN] + TEXT_COLUMNS
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    duration_seconds = _parse_duration_seconds(df[TARGET_DURATION_SECONDS_COLUMN])
    df[TARGET_COLUMN] = duration_seconds / 3600.0
    df['duration_days'] = duration_seconds / 86400.0

    before_count = len(df)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] >= 0)].copy()
    valid_count = len(df)
    df = df[df['duration_days'] <= DURATION_OUTLIER_MAX_DAYS].copy()
    filtered_count = len(df)

    print("Item별 모델링 데이터 준비 완료:")
    print(f" - 전체 데이터: {before_count}")
    print(f" - 유효 duration 데이터: {valid_count}")
    print(f" - {DURATION_OUTLIER_MAX_DAYS}일 초과 이상치 제외: {valid_count - filtered_count}")
    print(f" - 모델링 대상 데이터: {filtered_count}")

    return df


def _metric_row(item_name, scenario_name, model_name, y_test, y_pred, elapsed_seconds, n_total, n_train, n_test):
    mae_hours = mean_absolute_error(y_test, y_pred)
    rmse_hours = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        'Item': item_name,
        'Scenario': scenario_name,
        'Model': model_name,
        'N_Total': n_total,
        'N_Train': n_train,
        'N_Test': n_test,
        'MAE_sec': mae_hours * 3600.0,
        'MAE_min': mae_hours * 60.0,
        'MAE_hour': mae_hours,
        'MAE_day': mae_hours / 24.0,
        'RMSE_sec': rmse_hours * 3600.0,
        'RMSE_min': rmse_hours * 60.0,
        'RMSE_hour': rmse_hours,
        'RMSE_day': rmse_hours / 24.0,
        'R2': r2_score(y_test, y_pred),
        'TrainTime': f"{elapsed_seconds:.1f}초" if elapsed_seconds < 60 else f"{elapsed_seconds / 60.0:.1f}분",
        'ElapsedSeconds': elapsed_seconds,
    }


def run_item_experiments(
    data_path=DATA_FILE,
    output_path='item_experiment_results.csv',
    summary_path='item_model_summary.csv',
    skipped_path='item_skipped.csv',
):
    df = load_filtered_item_data(data_path)
    item_counts = df[ITEM_GROUP_COLUMN].value_counts(dropna=False)
    target_items = item_counts[item_counts >= MIN_ITEM_SAMPLES].index.tolist()
    skipped_items = item_counts[item_counts < MIN_ITEM_SAMPLES].reset_index()
    skipped_items.columns = ['Item', 'N_After_Filter']
    skipped_items.to_csv(skipped_path, index=False, encoding='utf-8-sig')

    print(f" - Item별 모델링 대상: {len(target_items)}개 Item")
    print(f" - 표본 부족으로 제외: {len(skipped_items)}개 Item (< {MIN_ITEM_SAMPLES}건)")

    structured_cols = [col for col in STRUCTURED_COLUMNS if col != ITEM_GROUP_COLUMN and col in df.columns]
    text_cols = [col for col in TEXT_COLUMNS if col in df.columns]

    results = []
    for item_idx, item_name in enumerate(target_items, start=1):
        item_df = df[df[ITEM_GROUP_COLUMN] == item_name].copy()
        train_df, test_df = train_test_split(
            item_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        print(f"\n[{item_idx}/{len(target_items)}] Item='{item_name}' | N={len(item_df)}")
        preprocessor = SRDataPreprocessor(
            structured_cols=structured_cols,
            text_cols=text_cols,
            max_text_features=MAX_TEXT_FEATURES,
            bert_model_name=BERT_MODEL_NAME,
        )
        X_train_tfidf, X_train_kobert, y_train = preprocessor.fit_transform(train_df)
        X_test_tfidf, X_test_kobert, y_test = preprocessor.transform(test_df)

        scenarios = {
            'TF-IDF': (X_train_tfidf, X_test_tfidf),
            'KoBERT': (X_train_kobert, X_test_kobert),
        }

        for scenario_name, (X_train, X_test) in scenarios.items():
            for model_name, model in get_regression_models().items():
                print(f"  - {scenario_name} + {model_name} 학습/평가")
                start_time = time.time()
                model.fit(X_train, y_train)
                elapsed_seconds = time.time() - start_time
                y_pred = np.maximum(model.predict(X_test), 0)
                row = _metric_row(
                    item_name,
                    scenario_name,
                    model_name,
                    y_test,
                    y_pred,
                    elapsed_seconds,
                    len(item_df),
                    len(train_df),
                    len(test_df),
                )
                results.append(row)
                print(f"    MAE={row['MAE_hour']:.2f}시간({row['MAE_day']:.2f}일), R2={row['R2']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    summary_df = (
        results_df
        .sort_values(['Item', 'MAE_hour', 'R2'], ascending=[True, True, False])
        .groupby('Item', as_index=False)
        .first()
        .sort_values('MAE_hour')
    )
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    write_item_report(results_df, summary_df, skipped_items)
    plot_item_summary(summary_df)

    print(f"\nItem별 전체 결과 저장: {output_path}")
    print(f"Item별 최적 모델 요약 저장: {summary_path}")
    print(f"표본 부족 제외 Item 저장: {skipped_path}")
    return results_df, summary_df


def plot_item_summary(summary_df, output_path='item_model_summary.png'):
    top_df = summary_df.sort_values('N_Total', ascending=False).head(20).copy()
    top_df['ItemLabel'] = top_df['Item'].astype(str).str.slice(0, 24)

    plt.figure(figsize=(12, 9))
    sns.barplot(data=top_df, x='MAE_day', y='ItemLabel', hue='Scenario')
    plt.title(f'Item별 최적 모델 MAE 비교 ({DURATION_OUTLIER_MAX_DAYS}일 초과 제외)')
    plt.xlabel('MAE (일)')
    plt.ylabel('Item')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Item별 요약 그래프 저장: {output_path}")


def write_item_report(results_df, summary_df, skipped_items, output_path='item_model_report.md'):
    best_overall = summary_df.sort_values(['MAE_hour', 'R2'], ascending=[True, False]).iloc[0]
    weighted_mae_hour = np.average(summary_df['MAE_hour'], weights=summary_df['N_Test'])

    report = f"""# Item별 SR 처리시간 예측 모델링 결과

## 설정

* 기준 데이터: `{DATA_FILE}`
* 타겟: `{TARGET_DURATION_SECONDS_COLUMN}` seconds
* 이상치 기준: 처리시간이 `{DURATION_OUTLIER_MAX_DAYS}`일을 초과하면 제외
* Item별 최소 표본 수: `{MIN_ITEM_SAMPLES}`건
* 비교 방식: 각 Item별로 TF-IDF/KoBERT x Ridge/Random Forest/XGBoost/LightGBM 총 8개 모델 평가

## 전체 요약

* 개별 모델링 대상 Item: {summary_df['Item'].nunique()}개
* 표본 부족 제외 Item: {len(skipped_items)}개
* Item별 최적 모델 기준 가중 평균 MAE: {weighted_mae_hour:.2f}시간 ({weighted_mae_hour / 24.0:.2f}일)
* 가장 낮은 MAE Item/모델: `{best_overall['Item']}` / {best_overall['Scenario']} + {best_overall['Model']} / MAE {best_overall['MAE_hour']:.2f}시간

## Item별 최적 모델

| Item | N | 최적 모델 | MAE(시간) | MAE(일) | RMSE(시간) | R² |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
"""
    for _, row in summary_df.sort_values('N_Total', ascending=False).iterrows():
        report += (
            f"| {row['Item']} | {int(row['N_Total'])} | {row['Scenario']} + {row['Model']} | "
            f"{row['MAE_hour']:.2f} | {row['MAE_day']:.2f} | {row['RMSE_hour']:.2f} | {row['R2']:.4f} |\n"
        )

    report += """
## 해석 메모

* 10일 초과 장기 티켓을 제거했기 때문에, 이 결과는 단기/일반 처리 티켓 예측에 더 적합합니다.
* R²가 낮거나 음수인 Item은 같은 Item 안에서도 담당팀, 승인/대기, 업무 난이도 차이가 크다는 뜻입니다.
* 운영 적용 시에는 Item별 최적 모델을 쓰거나, Item별 SLA 구간 분류를 추가하는 방식이 좋습니다.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Item별 보고서 저장: {output_path}")


if __name__ == '__main__':
    run_item_experiments()
