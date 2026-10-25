# constants.py — 여러 단계가 함께 쓰는 상수: 종목 마스터(STOCKS·ACTIVE_TICKERS), 주가 초기적재 시작일, 최소 train 시퀀스 수,
# 수집 확정 컷오프. torch 등 무거운 패키지를 import하지 않는다(수집기·월간 갱신이 가볍게 쓰도록).
# ⚠️ STOCKS 블록은 가격예측.종목_월간갱신 --apply가 줄 단위로 고쳐 쓴다 — "STOCKS = [" 줄, 닫는 "]" 줄, Stock(...) 한 줄 형식을 바꾸지 말 것.
# 배경: 결정기록.md의 "종목 유니버스는 KOSPI 시가총액 상위 100", "최소 train 시퀀스 10", 확정 컷오프는 주가데이터/결정기록.md
from collections import namedtuple
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# 주가 초기적재(주가_초기적재.py) 시작일
STOCK_INITIAL_LOAD_START = "2020-01-01"

# pooled 변동성 모델이 종목 하나를 학습에 넣기 위한 최소 train 시퀀스 수.
# 가격예측_변동성_공통.train_daily_pooled_model(경고)과 종목_월간갱신.py(신규 진입 게이팅)가 함께 쓴다.
MIN_TRAIN_SEQUENCES = 10

# 수집 확정 컷오프: KST 오늘 날짜 행은 이 시각 전에는 장중 잠정값이라 저장하지 않는다(주가·USD_KRW, 증분·--force).
# Actions cron(19:54 KST)이 이 시각 뒤라 제시간 실행은 당일분을 받는다 — cron을 이보다 앞으로 옮기면 매일 당일분을 놓친다.
# Actions 러너는 UTC라 KST를 명시한다.
KST = ZoneInfo("Asia/Seoul")
CLOSE_FINAL_CUTOFF_KST = time(19, 40)


def last_storable_date(now=None):
    """
    설명: 저장해도 되는 마지막 날짜(KST)를 돌려준다 — 컷오프 전이면 어제, 이후면 오늘.
    주의 사항: now를 주지 않으면 지금 KST 시각. 주말·휴장일은 따지지 않는다(그날 행이 소스에 없을 뿐).
    입력 타입->반환 타입: (datetime|None) -> date
    """
    now = now or datetime.now(KST)
    today = now.date()
    return today if now.time() >= CLOSE_FINAL_CUTOFF_KST else today - timedelta(days=1)


# 종목 마스터. active=False 종목은 수집·학습에서 빠지지만 이력 때문에 줄은 남긴다.
Stock = namedtuple("Stock", ["ticker", "name", "active"])

# 활성 100종목 = KOSPI 시가총액 상위 100(우선주 제외, fdr.StockListing('KOSPI') 기준). 첫 줄 005930이 분할 경계 기준 종목이다.
# 순서: 처음 9종목 → 시총 상위 50까지 → 51~100위(train 표본이 거의 없는 3종목은 다음 순위로 대체, 맨 아래 3줄).
STOCKS = [
    Stock("005930", "삼성전자", True),
    Stock("000660", "SK하이닉스", True),
    Stock("005380", "현대차", True),
    Stock("051910", "LG화학", True),
    Stock("105560", "KB금융", True),
    Stock("207940", "삼성바이오로직스", True),
    Stock("035420", "NAVER", True),
    Stock("034730", "SK", True),                  # SK(지주). 현대건설은 000720(아래)
    Stock("015760", "한국전력", True),
    # 시가총액 상위 50까지
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
    # 시가총액 51~100위
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
    Stock("036570", "NC", True),               # 대체(원래 순위는 0126Z0 삼성에피스홀딩스, train 0행)
    Stock("088980", "맥쿼리인프라", True),       # 대체(원래 순위는 064400 LG씨엔에스, train 0행)
    Stock("052690", "한전기술", True),          # 대체(원래 순위는 062040 산일전기, train 38행)
]

# 활성 종목코드만(STOCKS 순서 유지 — 수집 순서이자 ACTIVE_TICKERS[0]이 분할 경계 기준 종목)
ACTIVE_TICKERS = [s.ticker for s in STOCKS if s.active]
