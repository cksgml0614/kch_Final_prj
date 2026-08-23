# 주가_일일수집.py
# 주가_공통.py만 참조한다(주가_초기적재.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). 종목별 DB 최신 날짜 다음날부터 오늘까지 증분 수집한다.
# 아직 데이터가 전혀 없는 종목(초기적재 필요)은 처리하지 않고 경고만 출력한다.

from 주가데이터.주가_공통 import TICKERS, update_stock_data

if __name__ == "__main__":
    results = {}
    for ticker in TICKERS:
        try:
            results[ticker] = update_stock_data(ticker)
        except Exception as e:
            print(f"❌ {ticker}: 업데이트 실패: {e}")
            results[ticker] = f"실패 ({e})"

    print("\n=== 처리 결과 요약 ===")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")
