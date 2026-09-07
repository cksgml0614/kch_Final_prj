# 가격예측_변동성_일일수집.py
# Task T 자동화 파이프라인의 일일 실행 진입점(2026-09-07, 100종목 검증 완료 후 신규 작성 —
# CLAUDE.md "다음 세션 시작 시 할 일" 2번 항목). 실제 로직은 전부 가격예측_변동성_공통.py에
# 있고, 이 파일은 CLI 인자 파싱 + 진입점만 담당한다(다른 트랙의 "공통/일일수집" 분리 패턴).
#
# 종목 목록은 constants.ACTIVE_TICKERS를 그대로 쓴다(현재 100종목, 종목_월간갱신.py가 매달
# 갱신). --tickers로 재정의 가능(디버깅/부분 재실행용).
#
# 실행: python -m 가격예측.가격예측_변동성_일일수집 [--tickers 005930 ...] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
#
# ⚠️ 2026-09-07 세션에서는 코드만 작성 — 실제 실행/스케줄러 등록은 다음 세션 과제:
#   1) Database/가격예측/테이블_생성.sql 하단 마이그레이션(parkinson_sma20_baseline/
#      gate_vs_parkinson 컬럼)을 운영 DB에 적용
#   2) 최초 1회 이 스크립트를 수동 실행해 정상 동작 확인(GARCH 파라미터 캐시가 비어있어
#      첫 실행은 100종목 전부 재적합 — 이후 실행부터는 캐시 재사용으로 빨라짐)
#   3) 크론/스케줄러 등록

import argparse
import sys
from datetime import date

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_변동성_공통 import run_pooled_volatility_pipeline

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
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
