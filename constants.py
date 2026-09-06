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
# (034730 현대건설/건설, 015760 한국전력/에너지) 추가해 총 8종목 활성화. 000660(SK하이닉스)은
# 계속 비활성 유지(Task C 사람 결정 그대로).
STOCKS = [
    Stock("005930", "삼성전자", True),
    Stock("000660", "SK하이닉스", False),        # 자리표시자 유지, Task C 재검토 전까지 비활성
    Stock("005380", "현대차", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("051910", "LG화학", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("105560", "KB금융", True),              # 2026-09-06 활성화(Task T pooled 모델)
    Stock("207940", "삼성바이오로직스", True),     # 2026-09-06 활성화(Task T pooled 모델)
    Stock("035420", "NAVER", True),               # 2026-09-06 활성화(Task T pooled 모델)
    Stock("034730", "현대건설", True),             # 2026-09-06 신규 추가(건설 섹터, Task T pooled 모델)
    Stock("015760", "한국전력", True),             # 2026-09-06 신규 추가(에너지/전력 섹터, Task T pooled 모델)
]

# 뉴스 크롤러: 종목코드 -> 검색어(회사명) 매핑 (비활성 포함 전체)
STOCK_NAMES = {s.ticker: s.name for s in STOCKS}

# 활성 종목코드만 (주가 로더 등 "무엇을 수집할지"만 필요한 쪽에서 사용)
ACTIVE_TICKERS = [s.ticker for s in STOCKS if s.active]
