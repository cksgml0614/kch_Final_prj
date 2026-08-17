# detect_stress_period.py — 일회성 분석 도구 (파이프라인에서 호출하지 않음)
#
# 005930 일별 |등락률| 10일 이동평균 + KOSPI 월별 평균 |등락률|로 변동성 레짐 변곡점을 찾는다.
# constants.py의 STRESS_PERIOD_START/END는 이 스크립트의 2026-08-17 실행 결과를 사람이 확인하고
# 수동으로 확정한 값이다. 데이터가 추가돼 재평가가 필요할 때만 다시 실행하고, 재실행 결과를
# 자동으로 constants.py에 반영하지 않는다 — 사람이 값을 보고 판단해서 직접 고친다.

import sys

import pandas as pd

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def monthly_avg_abs_change_rate(ticker, since="2023-09-01"):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, change_rate FROM daily_stock_prices
                WHERE ticker = %s AND date >= %s ORDER BY date
                """,
                (ticker, since),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "change_rate"])
    df["change_rate"] = df["change_rate"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym")["change_rate"].apply(lambda s: s.abs().mean())


def kospi_monthly_avg_abs_change_rate(since="2023-09-01"):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, value FROM market_indicators
                WHERE indicator_code = 'KOSPI' AND date >= %s ORDER BY date
                """,
                (since,),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["change_rate"] = df["value"].pct_change()
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym")["change_rate"].apply(lambda s: s.abs().mean())


def daily_rolling10_abs_change_rate(ticker, start, end):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, change_rate FROM daily_stock_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s ORDER BY date
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "change_rate"])
    df["change_rate"] = df["change_rate"].astype(float)
    df["roll10"] = df["change_rate"].abs().rolling(10, min_periods=1).mean()
    return df


if __name__ == "__main__":
    TICKER = "005930"

    print("=== 005930 월별 평균|최대 |등락률| (2023-09~) ===")
    m = monthly_avg_abs_change_rate(TICKER)
    print(m)

    print("\n=== KOSPI 월별 평균 |등락률| (2023-09~) ===")
    k = kospi_monthly_avg_abs_change_rate()
    print(k)

    print("\n=== 005930 일별 |등락률| 10일 이동평균, 변곡점 구간 (2026-01-05~2026-02-20) ===")
    d = daily_rolling10_abs_change_rate(TICKER, "2026-01-05", "2026-02-20")
    print(d.to_string(index=False))

    print(
        "\n=== 결론 (2026-08-17 확정, constants.py 반영됨) ===\n"
        "  2026-02-02(-6.3%)/02-03(+11.4%)를 기점으로 roll10이 1.3~2.0%대에서 3~6%대로 급변,\n"
        "  이후 데이터셋 끝(2026-07-31)까지 지속. STRESS_PERIOD_START=2026-02-02, END=2026-07-31."
    )
