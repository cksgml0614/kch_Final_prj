# constants.py
from collections import namedtuple

# Task T(가격+거시지표 트랜스포머 베이스라인) 전용 스트레스 구간 경계.
# ⚠️ 자동 재탐지 금지 — 갱신은 사람이 analysis/detect_stress_period.py 수동 재실행 후 값을 직접 교체. 상세는 CLAUDE.md 참조.
STRESS_PERIOD_START = "2026-02-02"
STRESS_PERIOD_END = "2026-07-31"

# 뉴스 크롤러(뉴스_최초적재.py) 백필 대상 범위.
# ⚠️ 고정값 — 실행 시점 기준 자동 재계산 안 됨. 상세는 CLAUDE.md 참조.
NEWS_BACKFILL_START = "2023-08-23"
NEWS_BACKFILL_END = "2026-08-23"

# 주가 초기적재(주가_초기적재.py) 시작일
STOCK_INITIAL_LOAD_START = "2020-01-01"

# 뉴스/감성 트랙(Task E/F) 학습 대상 기간.
# ⚠️ 고정값 — 데이터가 더 쌓여도 자동으로 늘어나지 않음. NEWS_BACKFILL_START/END와의 관계·갱신
# 기준은 CLAUDE.md 참조.
TRAIN_PERIOD_START = "2023-08-23"
TRAIN_PERIOD_END = "2026-08-28"

# Task E/F 시계열 분할 비율 (train, val, test) — D-8
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# 종목 마스터. 배경은 CLAUDE.md 참조.
Stock = namedtuple("Stock", ["ticker", "name", "active"])

# 2026-09-06: Task T 다종목 pooled 모델 착수 — 기존 섹터 분산 후보 5종목 활성화 + 신규 2종목
# (034730 SK/지주, 015760 한국전력/에너지) 추가해 총 8종목 활성화, 000660(SK하이닉스)은
# 계속 비활성 유지(Task C 사람 결정 그대로). ⚠️ 034730은 애초 "현대건설"로 잘못 등록했던
# 것을 같은 날 50종목 확장 작업 중 발견해 정정(034730의 실제 종목은 SK 지주회사 — 현대건설의
# 실제 코드는 000720이며 별도 미등록. 가격 데이터 자체는 034730 그대로였으니 이미 실행한
# GARCH/하이브리드/Parkinson 실험 결과에는 영향 없음, 이름표만 틀렸던 것).
#
# 2026-09-06(같은 날 추가 확장): 변동성 예측 트랙 표본 확대를 위해 KOSPI 시가총액 상위 50종목
# (우선주 제외, fdr.StockListing('KOSPI') 기준)으로 확장. 기존 9종목(활성 8 + SK하이닉스)이
# 전부 이 50위 안에 이미 포함됨을 확인. 000660(SK하이닉스)도 이번엔 활성화 — 섹터 분산
# 원칙은 종목 수가 적을 때(8개) 의미가 컸지만 50개 규모에서는 시가총액 기준 표본 확대가
# 우선이라고 판단.
STOCKS = [
    Stock("005930", "삼성전자", True),
    Stock("000660", "SK하이닉스", True),          # 2026-09-06 활성화(50종목 확장, 시총 2위)
    Stock("005380", "현대차", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("051910", "LG화학", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("105560", "KB금융", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("207940", "삼성바이오로직스", True),     # 2026-09-06 활성화(Task T pooled 모델)
    Stock("035420", "NAVER", True),               # 2026-09-06 활성화(Task T pooled 모델)
    Stock("034730", "SK", True),                  # 2026-09-06 신규 추가, 같은 날 이름 오류 정정(구 "현대건설")
    Stock("015760", "한국전력", True),             # 2026-09-06 신규 추가(에너지/전력 섹터, Task T pooled 모델)
    # 2026-09-06 확장(시가총액 상위 50, Task T 변동성 pooled 모델 표본 확대) — 우선주 제외,
    # fdr.StockListing('KOSPI') 시총 순위 기준
    Stock("402340", "SK스퀘어", True),
    Stock("009150", "삼성전기", True),
    Stock("373220", "LG에너지솔루션", True),
    Stock("028260", "삼성물산", True),
    Stock("032830", "삼성생명", True),
    Stock("012450", "한화에어로스페이스", True),
    Stock("055550", "신한지주", True),
    Stock("034020", "두산에너빌리티", True),
    Stock("000270", "기아", True),
    Stock("329180", "HD현대중공업", True),
    Stock("006400", "삼성SDI", True),
    Stock("068270", "셀트리온", True),
    Stock("012330", "현대모비스", True),
    Stock("086790", "하나금융지주", True),
    Stock("066570", "LG전자", True),
    Stock("000810", "삼성화재", True),
    Stock("010120", "LS ELECTRIC", True),
    Stock("005490", "POSCO홀딩스", True),
    Stock("042660", "한화오션", True),
    Stock("267260", "HD현대일렉트릭", True),
    Stock("010130", "고려아연", True),
    Stock("298040", "효성중공업", True),
    Stock("009540", "HD한국조선해양", True),
    Stock("316140", "우리금융지주", True),
    Stock("096770", "SK이노베이션", True),
    Stock("138040", "메리츠금융지주", True),
    Stock("042700", "한미반도체", True),
    Stock("017670", "SK텔레콤", True),
    Stock("011200", "HMM", True),
    Stock("267250", "HD현대", True),
    Stock("010140", "삼성중공업", True),
    Stock("006800", "미래에셋증권", True),
    Stock("000150", "두산", True),
    Stock("018260", "삼성에스디에스", True),
    Stock("003550", "LG", True),
    Stock("033780", "KT&G", True),
    Stock("010950", "S-Oil", True),
    Stock("024110", "기업은행", True),
    Stock("003670", "포스코퓨처엠", True),
    Stock("035720", "카카오", True),
    Stock("086280", "현대글로비스", True),
    # 2026-09-07 확장(50→100, Task T 변동성 pooled 모델 표본 추가 확대) — 어제(2026-09-06)
    # 8→50 확장에서 분산 붕괴 완화 + 종목평균 제거 후에도 baseline 대비 우위 확인된 효과가
    # 표본을 더 늘려도 이어지는지 검증하기 위함. fdr.StockListing('KOSPI') 재조회, 우선주
    # 필터(이름이 "...우"/"...우B" 등으로 끝나는 종목, 109개 제외) 동일 방식, 시가총액 상위
    # 100위까지 추출 — 기존 50종목 전부 상위 50위 안에 포함됨을 재확인(그대로 유지, 순서
    # 변경 없음). 상위 51~100위 후보 중 3종목은 train 구간(<=2024-09-24) 표본이 0~38행으로
    # 지나치게 적어(momentum 60일 롤링 등 피처 lead-in을 감안하면 dropna 후 train 시퀀스가
    # 0개가 될 위험) 종목 임베딩이 사실상 학습되지 않는 문제가 있어 제외하고, 그 다음 순위
    # 종목으로 대체했다: 0126Z0(삼성에피스홀딩스, 상장 2025-11-24, train 0행/val 0행 — 전
    # 표본이 test 구간) / 064400(LG씨엔에스, 상장 2025-02-05, train 0행) / 062040(산일전기,
    # 상장 2024-07-29, train 38행) 제외 → 036570(NC)/088980(맥쿼리인프라)/052690(한전기술)로
    # 대체(전부 2020-01-02부터 전체 이력 보유, train 1,166행). 100종목 전부 daily_stock_prices
    # 신규 적재 완료(가격예측용 — 뉴스 수집 대상 자동 확대 아님, 위 50종목 확장 절과 동일 원칙).
    Stock("079550", "LIG디펜스앤에어로스페이스", True),
    Stock("278470", "에이피알", True),
    Stock("000720", "현대건설", True),
    Stock("064350", "현대로템", True),
    Stock("011070", "LG이노텍", True),
    Stock("030200", "KT", True),
    Stock("272210", "한화시스템", True),
    Stock("047810", "한국항공우주", True),
    Stock("005830", "DB손해보험", True),
    Stock("078930", "GS", True),
    Stock("003490", "대한항공", True),
    Stock("307950", "현대오토에버", True),
    Stock("071050", "한국금융지주", True),
    Stock("323410", "카카오뱅크", True),
    Stock("003230", "삼양식품", True),
    Stock("259960", "크래프톤", True),
    Stock("180640", "한진칼", True),
    Stock("005940", "NH투자증권", True),
    Stock("006260", "LS", True),
    Stock("047050", "포스코인터내셔널", True),
    Stock("443060", "HD현대마린솔루션", True),
    Stock("028050", "삼성E&A", True),
    Stock("090430", "아모레퍼시픽", True),
    Stock("007660", "이수페타시스", True),
    Stock("161390", "한국타이어앤테크놀로지", True),
    Stock("016360", "삼성증권", True),
    Stock("352820", "하이브", True),
    Stock("047040", "대우건설", True),
    Stock("000880", "한화", True),
    Stock("039490", "키움증권", True),
    Stock("000500", "가온전선", True),
    Stock("021240", "코웨이", True),
    Stock("267270", "HD건설기계", True),
    Stock("009830", "한화솔루션", True),
    Stock("326030", "SK바이오팜", True),
    Stock("032640", "LG유플러스", True),
    Stock("377300", "카카오페이", True),
    Stock("241560", "두산밥캣", True),
    Stock("128940", "한미약품", True),
    Stock("000100", "유한양행", True),
    Stock("175330", "JB금융지주", True),
    Stock("001440", "대한전선", True),
    Stock("029780", "삼성카드", True),
    Stock("353200", "대덕전자", True),
    Stock("088350", "한화생명", True),
    Stock("271560", "오리온", True),
    Stock("138930", "BNK금융지주", True),
    Stock("036570", "NC", True),               # 대체(원래 순위는 0126Z0 삼성에피스홀딩스)
    Stock("088980", "맥쿼리인프라", True),       # 대체(원래 순위는 064400 LG씨엔에스)
    Stock("052690", "한전기술", True),          # 대체(원래 순위는 062040 산일전기)
]

# 뉴스 크롤러: 종목코드 -> 검색어(회사명) 매핑 (비활성 포함 전체)
STOCK_NAMES = {s.ticker: s.name for s in STOCKS}

# 활성 종목코드만 (주가 로더 등 "무엇을 수집할지"만 필요한 쪽에서 사용)
ACTIVE_TICKERS = [s.ticker for s in STOCKS if s.active]
