# 가격예측_변동성_일일수집.py — pooled 하이브리드 변동성 모델의 일일 실행 진입점(CLI 인자 파싱만, 로직은 가격예측_변동성_공통.py).
# 파이프라인 단계: 진입점(daily_pipeline.bat이 실행). 주요 진입 함수: main.
# 종목은 constants.ACTIVE_TICKERS(--tickers로 바꿀 수 있음).
# 실행: python -m 가격예측.가격예측_변동성_일일수집 [--tickers 005930 ...] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

import argparse
import sys
from datetime import date

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_변동성_공통 import run_pooled_volatility_pipeline

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    """
    설명: 인자를 읽어 종목·기간을 정하고 run_pooled_volatility_pipeline을 실행한다.
    주의 사항: 운영 DB·MLflow·체크포인트·GARCH 캐시에 쓴다. --end 생략 시 오늘. 종목이 2개 미만이면 실행하지 않는다.
    입력 타입->반환 타입: () -> None
    """
    parser = argparse.ArgumentParser(
        description="Task T pooled 하이브리드 변동성 모델 — 전 종목 통합 학습 + 종목별 다음 거래일 예측 저장"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="예: --tickers 005930 000660 (생략 시 constants.ACTIVE_TICKERS 전체, 현재 100종목)")
    parser.add_argument("--start", type=str, default=STOCK_INITIAL_LOAD_START,
                         help="학습 시작일 YYYY-MM-DD (기본: constants.STOCK_INITIAL_LOAD_START)")
    parser.add_argument("--end", type=str, default=None,
                         help="학습에 사용할 마지막 날짜 YYYY-MM-DD (생략 시 오늘)")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else ACTIVE_TICKERS
    end_date = args.end if args.end else date.today().isoformat()

    if not tickers:
        print("⚠️ 처리할 종목이 없습니다 — constants.ACTIVE_TICKERS가 비어있는지 확인하세요.")
        return
    if len(tickers) < 2:
        print("⚠️ pooled 모델은 2종목 이상이 전제입니다 — 단일 종목만으로는 실행할 수 없습니다.")
        return

    run_pooled_volatility_pipeline(tickers, args.start, end_date)


if __name__ == "__main__":
    main()
