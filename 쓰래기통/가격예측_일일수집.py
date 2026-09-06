# 가격예측_일일수집.py
# Task T 자동화 파이프라인의 일일 실행 진입점(2026-09-06). 다른 트랙(주가_일일수집.py 등)의
# "공통/일일수집" 분리 패턴을 따른다 — 실제 로직은 전부 가격예측_공통.py에 있고, 이 파일은
# 종목 순회 + 예외 격리 + CLI 인자만 담당한다.
#
# 종목 목록은 constants.ACTIVE_TICKERS를 순회한다(2026-09-06, 다종목 확장 대비 — 지금은 활성
# 종목이 005930 하나뿐이라 동작 결과는 단일 종목 실행과 동일하다). --tickers로 재정의 가능.
#
# 종목별 실행은 train_common.run_isolated()로 감싼다 — 한 종목이 실패해도(예: 가격 데이터
# 갱신 지연) 나머지 종목은 계속 처리된다. 실패는 콘솔 + 로그 파일(가격예측/logs/)에 남는다.
#
# 실행: python -m 가격예측.가격예측_일일수집 [--tickers 005930 ...] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

import argparse
import os
from datetime import date

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_공통 import run_daily_pipeline
from 가격예측.train_common import run_isolated

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def main():
    parser = argparse.ArgumentParser(
        description="Task T 자동화 파이프라인 — 매일 전체 재학습 + 다음 거래일 예측 저장"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="예: --tickers 005930 (생략 시 constants.ACTIVE_TICKERS 전체 순회)")
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

    results = {}
    for ticker in tickers:
        r = run_isolated(run_daily_pipeline, ticker, args.start, end_date, label=ticker, log_dir=LOG_DIR)
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    print("\n=== 종목별 처리 결과 요약 ===")
    any_failed = False
    for ticker, status in results.items():
        if isinstance(status, str) and status.startswith("실패"):
            any_failed = True
        print(f"{ticker}: {status}")
    if any_failed:
        print("⚠️ 실패한 종목이 있습니다 — 가격예측/logs/ 에서 상세 로그를 확인하세요.")


if __name__ == "__main__":
    main()
