import os
from generate_mock_data import generate_sr_data
from experiment import run_experiments
from compare_results import generate_comparison_report

def main():
    data_file = 'sr_data.csv'
    
    # 1. 데이터 확인 및 가상 데이터 생성
    if not os.path.exists(data_file):
        print(f"[1단계] '{data_file}' 데이터 파일이 존재하지 않아 가상 데이터를 생성합니다...")
        generate_sr_data(num_samples=1000)
    else:
        print(f"[1단계] 기존 데이터 파일 '{data_file}'을 사용하여 진행합니다.")
        
    # 2. 모델 학습 및 실험 진행 (TF-IDF vs KoBERT / Ridge, RF, XGBoost, LightGBM 총 8가지)
    print("\n[2단계] 시나리오별 모델 학습 및 비교 실험을 시작합니다...")
    run_experiments(data_path=data_file, output_path='experiment_results.csv')
    
    # 3. 결과 분석 및 보고서/시각화 그래프 생성
    print("\n[3단계] 실험 결과 비교 차트 및 보고서를 생성합니다...")
    generate_comparison_report(results_path='experiment_results.csv', data_path=data_file)
    
    print("\n" + "="*50)
    print("  전체 프로세스가 완료되었습니다! 다음 파일을 확인하세요:")
    print("  - 성능 비교 차트: performance_comparison.png")
    print("  - 피처 중요도 분석: feature_importance.png")
    print("  - 종합 마크다운 보고서: report_summary.md")
    print("="*50)

if __name__ == '__main__':
    main()
