# 주가_초기적재.py
# 주가_공통.py만 참조한다(주가_일일수집.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). DB에 데이터가 전혀 없는 종목만 골라 STOCK_INITIAL_LOAD_START부터
# 전체 적재한다.

from constants import STOCK_INITIAL_LOAD_START
from 주가데이터.주가_공통 import TICKERS, _needs_initial_load, update_stock_data

if __name__ == "__main__":
    results = {}
    for ticker in TICKERS:
        try:
            if not _needs_initial_load(ticker):
                print(f"⏭️ {ticker}: 이미 데이터가 있어 초기적재 대상이 아닙니다 (주가_일일수집.py 사용).")
                results[ticker] = "스킵 (이미 초기적재됨)"
                continue
            results[ticker] = update_stock_data(ticker, start_date_override=STOCK_INITIAL_LOAD_START)
        except Exception as e:
            print(f"❌ {ticker}: 초기적재 실패: {e}")
            results[ticker] = f"실패 ({e})"

    print("\n=== 초기적재 결과 요약 ===")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")
