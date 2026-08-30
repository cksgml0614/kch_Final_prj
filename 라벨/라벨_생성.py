# 라벨_생성.py
# Task E — CLAUDE.md D-2 / TASK_EF_라벨링_비교실험.md "절차" 구현. 라벨_공통.
# compute_multi_horizon_z_scores()로 종목별 z_score(윈도우 20/60/120 x horizon 1/3/5/10)를
# 계산해 daily_labels에 저장한다. 라벨(방향 5/3-class, 변동성 2-class) 자체는 저장하지 않는다
# — 라벨_보고.py/Task F 스크립트가 조회 시점에 임계값을 적용해 파생한다.
#
# 2026-09-02: horizon_h 확장 반영 — 익일(h=1) 단일거래일 라벨 대신/추가로 h거래일(3/5/10)
# 누적 초과수익률 라벨을 계산한다. 기존 h=1 값은 동일 공식으로 재계산되므로(compute_multi_
# horizon_z_scores의 h=1 케이스가 옛 compute_z_scores와 수학적으로 동일) UPSERT의
# IS DISTINCT FROM 조건에 걸려 그대로 보존된다 — 재실행해도 값이 바뀌지 않는다.
#
# 뉴스 테이블(daily_news, daily_news_bigkinds)에는 아무것도 쓰지 않는다 — 이 스크립트는
# daily_labels에만 쓴다.
#
# 재실행 안전: daily_labels PK(ticker, date, window_n, horizon_h) UPSERT라 다시 실행해도
# 안전하다(값이 같으면 IS DISTINCT FROM 조건에 걸려 건드리지 않는다). 외부 API 호출이 아니라
# 이미 적재된 daily_stock_prices/market_indicators의 재계산이라 비용이 낮으므로, 다른
# 도메인처럼 "초기적재/일일수집"을 분리하지 않았다 — 가격/KOSPI 데이터가 갱신되면 이
# 스크립트를 그대로 다시 실행하면 된다.

import sys

from constants import ACTIVE_TICKERS
from db_manager import get_db_connection
from 라벨.라벨_공통 import HORIZONS, WINDOW_SIZES, compute_multi_horizon_z_scores, to_records, upsert_labels

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    summary = {}
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — 라벨_생성")
        with conn.cursor() as cur:
            for ticker in ACTIVE_TICKERS:
                merged = compute_multi_horizon_z_scores(ticker, cur)
                if merged.empty:
                    print(f"⚠️ {ticker}: 계산 대상 없음 (daily_stock_prices에 데이터 없음)")
                    summary[ticker] = 0
                    continue

                records = to_records(merged, ticker)
                inserted, updated = upsert_labels(cur, records)
                conn.commit()
                unchanged = len(records) - inserted - updated
                print(
                    f"✅ {ticker}: {merged['date'].min().date()}~{merged['date'].max().date()} "
                    f"({len(merged)}거래일) x 윈도우 {WINDOW_SIZES} x horizon {HORIZONS} = {len(records)}행 계산 "
                    f"— 신규 {inserted} / 갱신 {updated} / 변경없음 {unchanged}"
                )
                summary[ticker] = len(records)

    print("\n=== 종목별 daily_labels 계산 건수 요약 ===")
    any_zero = False
    for ticker in ACTIVE_TICKERS:
        cnt = summary.get(ticker, 0)
        if cnt == 0:
            any_zero = True
        print(f"  {ticker}: {cnt}행")
    if any_zero:
        print("⚠️ 0건 종목이 있습니다 — daily_stock_prices 적재 상태를 확인하세요.")


if __name__ == "__main__":
    main()
