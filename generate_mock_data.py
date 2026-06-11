import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 일관된 난수 생성을 위한 시드 설정
np.random.seed(42)
random.seed(42)

def generate_sr_data(num_samples=1000):
    categories = ['Network', 'Hardware', 'Software', 'Access', 'HR']
    priorities = ['1 - Critical', '2 - High', '3 - Moderate', '4 - Low']
    
    assignment_groups = {
        'Network': ['Network Support', 'Service Desk'],
        'Hardware': ['Hardware Support', 'Service Desk'],
        'Software': ['Software Support', 'Service Desk'],
        'Access': ['Security Team', 'Service Desk'],
        'HR': ['HR Team', 'Service Desk']
    }
    
    templates = {
        'Network': [
            ("인터넷 연결 끊김 장애", "오늘 아침부터 특정 층 전체 인터넷 연결이 끊겼습니다. 급한 업무가 많은데 빠른 확인 부탁드립니다."),
            ("사내 VPN 접속 오류 발생", "외부 출장 중에 사내 VPN에 접속하려고 하면 403 에러가 발생하며 로그인이 되지 않습니다."),
            ("와이파이 신호 불량 및 속도 저하", "회의실 A에서 사내 와이파이 연결이 자주 끊기고 속도가 너무 느려서 회의 진행이 어렵습니다."),
            ("네트워크 스위치 재부팅 요청", "개발 서버 대역의 네트워크 연결 상태가 불안정하여 스위치 장비 확인 및 재부팅을 요청합니다."),
            ("방화벽 포트 오픈 신청", "신규 개발 시스템 구축을 위해 개발망과 운영망 간의 특정 포트 오픈이 필요합니다.")
        ],
        'Hardware': [
            ("듀얼 모니터 미인식 문제", "모니터를 새로 연결했는데 PC에서 듀얼 모니터 인식이 안 되고 화면이 나오지 않습니다."),
            ("노트북 전원 불량 및 켜지지 않음", "어제 저녁까지 잘 쓰던 노트북이 오늘 아침에 전원 버튼을 눌러도 반응이 없고 충전 표시등도 켜지지 않습니다."),
            ("마우스 및 키보드 교체 요청", "키보드의 특정 키가 눌린 채로 안 올라오고, 마우스 휠이 정상적으로 작동하지 않아 업무에 지장이 있습니다. 새 장비로 교체 요청합니다."),
            ("회의실 빔프로젝터 연결 케이블 파손", "대회의실 빔프로젝터에 연결하는 HDMI 케이블 단자가 파손되어 화면 출력이 되지 않습니다. 교체 바랍니다."),
            ("PC 메모리 추가 장착 요청", "최근 코딩 작업 시 메모리 부족으로 인해 PC 속도가 급격히 느려집니다. 기존 8GB에서 16GB로 업그레이드 요청합니다.")
        ],
        'Software': [
            ("개발 IDE 라이선스 만료 및 갱신", "사용 중인 IntelliJ 개발 라이선스가 만료되어 실행이 되지 않습니다. 라이선스 키 갱신 또는 재발급 요청합니다."),
            ("MS Office 365 로그인 인증 오류", "아웃룩과 액셀 실행 시 라이선스 인증 창이 계속 뜨며 정품 인증이 풀리는 현상이 있습니다."),
            ("신규 백신 프로그램 설치 실패", "사내 보안 정책에 따라 백신 프로그램을 설치하려는데 설치 도중 에러가 나면서 롤백됩니다. 설치 지원 요청합니다."),
            ("Git 클라이언트 소스 동기화 에러", "로컬 저장소에서 원격 저장소로 push를 할 때 권한 오류 또는 알 수 없는 오류로 실패합니다."),
            ("데이터베이스 클라이언트 툴 설치 요청", "업무를 위해 DBeaver 설치가 필요합니다. 사내 소프트웨어 포털에서 설치가 안 되어 지원 요청합니다.")
        ],
        'Access': [
            ("신규 프로젝트 공유 폴더 접근 권한 신청", "이번에 신규 생성된 공유 폴더인 'Project_2026'에 대한 읽기/쓰기 권한 부여를 신청합니다."),
            ("사내 인트라넷 계정 잠금 해제 요청", "비밀번호를 5회 잘못 입력하여 사내 포털 계정이 잠겼습니다. 잠금 해제 및 임시 비밀번호 발급 부탁드립니다."),
            ("신규 입사자 시스템 권한 일괄 신청", "다음 주 입사 예정인 신입 사원의 메일, 메신저, 사내 시스템 접근 권한 일괄 생성을 요청합니다."),
            ("Jira 및 Confluence 프로젝트 관리자 권한 부여", "부서 이동으로 인해 특정 Jira 프로젝트의 관리 권한 인계가 필요합니다."),
            ("보안 서버 접근용 SSH 키 등록", "운영 서버 점검을 위해 개발자 공개 SSH 키 등록을 요청합니다.")
        ],
        'HR': [
            ("재직증명서 국문 및 영문 발급 요청", "은행 제출용 재직증명서 국문 1부, 비자 신청용 영문 1부를 급히 발급 요청합니다."),
            ("올해 연차 잔여 일수 확인 요청", "포털에 표시되는 잔여 연차 일수와 실제 사용 일수가 상이한 것 같아 정확한 확인을 요청합니다."),
            ("경조사 휴가 및 경조금 신청", "가족 상가로 인한 특별 휴가 신청서와 경조금 지급 신청서류를 제출합니다."),
            ("사원증 분실로 인한 재발급 신청", "출근 중 사원증을 분실하여 재발급을 요청합니다. 출입 권한 이전도 같이 요청합니다."),
            ("사내 교육 수강 이력 반영 요청", "지난주 완료한 직무 교육 수강 확인서를 첨부하오니 교육 이력에 반영해 주시기 바랍니다.")
        ]
    }
    
    # 기본 처리 완료 시간 규칙 정의 (시간 단위)
    # 카테고리별 기본 처리 시간 (시간)
    base_hours_by_category = {
        'Network': 4.0,
        'Hardware': 12.0,
        'Software': 6.0,
        'Access': 2.0,
        'HR': 24.0
    }
    
    # 우선순위별 곱해질 가중치 (우선순위가 높을수록 긴급하여 빠르게 처리됨)
    priority_multipliers = {
        '1 - Critical': 0.25,  # 긴급하게 처리
        '2 - High': 0.5,
        '3 - Moderate': 1.0,
        '4 - Low': 1.5
    }
    
    data = []
    start_date = datetime(2026, 1, 1)
    
    for i in range(num_samples):
        sys_id = f"SR{i+10000:05d}"
        
        # 카테고리 임의 선택
        cat = random.choice(categories)
        
        # 우선순위 선택 (보통 Moderate, Low가 많음)
        priority = np.random.choice(priorities, p=[0.05, 0.15, 0.60, 0.20])
        
        # 배정 그룹
        group = random.choice(assignment_groups[cat])
        
        # 템플릿에서 제목과 상세 설명 선택
        template = random.choice(templates[cat])
        short_desc = template[0]
        desc = template[1]
        
        # 신청 시간 무작위 설정 (2026년 1월 ~ 5월 중)
        random_days = random.randint(0, 150)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        opened_at = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        
        # 기본 처리 시간 계산
        base_h = base_hours_by_category[cat]
        mult = priority_multipliers[priority]
        
        # 배정 그룹에 따른 영향 (Service Desk는 1차 지원 그룹이므로 조금 더 빨리 처리하거나 전달함)
        group_mult = 0.8 if group == 'Service Desk' else 1.2
        
        # 완료 시간 계산 (노이즈 추가)
        noise = np.random.normal(0, base_h * 0.15) # 15% 정도의 표준편차를 가지는 노이즈
        resolution_hours = max(0.5, (base_h * mult * group_mult) + noise)
        
        # 텍스트 데이터 내 특정 키워드에 의한 처리 시간 변동 규칙 심기 (자연어 모델이 이 패턴을 잡을 수 있게 함)
        # 예: 제목이나 본문에 "급합", "장애", "업무 불가", "긴급" 등이 들어가면 처리 속도가 20% 빨라짐
        urgent_keywords = ["장애", "에러", "오류", "분실", "만료"]
        for keyword in urgent_keywords:
            if keyword in short_desc or keyword in desc:
                resolution_hours *= 0.8
                break
                
        # 반대로 "구매", "추가", "신규", "교육" 등은 시간이 30% 늘어남
        slow_keywords = ["추가", "신규", "교육", "발급"]
        for keyword in slow_keywords:
            if keyword in short_desc or keyword in desc:
                resolution_hours *= 1.3
                break

        # 완료 시간 계산
        closed_at = opened_at + timedelta(hours=resolution_hours)
        
        data.append({
            'sys_id': sys_id,
            'opened_at': opened_at.strftime('%Y-%m-%d %H:%M:%S'),
            'closed_at': closed_at.strftime('%Y-%m-%d %H:%M:%S'),
            'category': cat,
            'priority': priority,
            'assignment_group': group,
            'short_description': short_desc,
            'description': desc
        })
        
    df = pd.DataFrame(data)
    df.to_csv('sr_data.csv', index=False, encoding='utf-8-sig')
    print(f"가상 데이터 생성 완료! 'sr_data.csv' 파일에 {num_samples}개의 티켓 데이터가 저장되었습니다.")

if __name__ == '__main__':
    generate_sr_data()
