# market_index_load.py — Task G-2
# KOSPI(KS11)/KOSDAQ(KQ11)/USD_KRW(USD/KRW)를 FinanceDataReader로 수집해
# market_indicators + indicator_meta에 적재한다.

import sys
from datetime import date, timedelta

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/특수문자를 인코딩하지 못해 죽는다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import FinanceDataReader as fdr
import pandas as pd

from db_manager import get_db_connection

# alias(indicator_code) -> FDR 심볼. 임시 매핑 방식(TASK_G_market_indicators_적재.md G-2 참고).
# indicator_meta 스키마에 source_symbol 컬럼이 없어 로더 내부 딕셔너리로만 관리한다.
INDICATOR_MAP = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
    "USD_KRW": "USD/KRW",
}

INDICATOR_META = {
    "KOSPI": {
        "name": "코스피 지수", "source": "FDR", "frequency": "D", "unit": "point",
        "note": "FDR 심볼: KS11 (alias 매핑, source_symbol 컬럼 없음)",
    },
    "KOSDAQ": {
        "name": "코스닥 지수", "source": "FDR", "frequency": "D", "unit": "point",
        "note": "FDR 심볼: KQ11 (alias 매핑, source_symbol 컬럼 없음)",
    },
    "USD_KRW": {
        "name": "원/달러 환율", "source": "FDR", "frequency": "D", "unit": "KRW",
        "note": "FDR 심볼: USD/KRW (alias 매핑, 1달러당 원화, value=Close 컬럼)",
    },
}


def upsert_indicator_meta(cur):
    """indicator_meta를 alias 3건으로 upsert. market_indicators 적재보다 먼저 실행해야 한다 (FK)."""
    query = """
        INSERT INTO indicator_meta (indicator_code, name, source, frequency, unit, note)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (indicator_code) DO UPDATE
            SET name = EXCLUDED.name,
                source = EXCLUDED.source,
                frequency = EXCLUDED.frequency,
                unit = EXCLUDED.unit,
                note = EXCLUDED.note
    """
    for code, meta in INDICATOR_META.items():
        cur.execute(query, (code, meta["name"], meta["source"], meta["frequency"], meta["unit"], meta["note"]))


def get_last_date(cur, indicator_code):
    """DB에서 해당 지표의 가장 최신 date를 가져온다."""
    cur.execute("SELECT MAX(date) FROM market_indicators WHERE indicator_code = %s", (indicator_code,))
    return cur.fetchone()[0]


def fetch_indicator_data(indicator_code, start_date=None):
    """FDR에서 alias에 대응하는 심볼 데이터를 조회해 적재용 튜플 리스트로 가공한다.
    DB에 쓰지 않는 순수 조회 함수 — 체크포인트 확인용으로 단독 호출 가능."""
    symbol = INDICATOR_MAP[indicator_code]
    df = fdr.DataReader(symbol, start_date) if start_date else fdr.DataReader(symbol)

    if df.empty:
        return []

    df.index.name = "Date"  # USD/KRW는 index.name이 None이라 reset_index 시 컬럼명이 달라짐 (KS11/KQ11은 'Date')
    df = df.reset_index()
    df = df.dropna(subset=["Close"])

    # 세 지표 모두 Close 컬럼을 종가로 제공 (KS11/KQ11: OHLCV+UpDown 등, USD/KRW: OHLCV+Adj Close)
    # USD/KRW는 float32 저장 소스라 1198.4000244140625 같은 이진 부동소수 잡음이 섞여 있어 반올림한다.
    return [
        (indicator_code, row["Date"].date(), round(float(row["Close"]), 4), row["Date"].date())
        for _, row in df.iterrows()
    ]


def upsert_market_indicators(cur, records):
    """G-1 계약: IS DISTINCT FROM + RETURNING으로 신규/개정 건수를 구분한다."""
    query = """
        INSERT INTO market_indicators (indicator_code, date, value, published_date)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (indicator_code, date) DO UPDATE
            SET value = EXCLUDED.value,
                published_date = EXCLUDED.published_date
            WHERE market_indicators.value IS DISTINCT FROM EXCLUDED.value
        RETURNING (xmax <> 0) AS was_update
    """
    inserted = 0
    updated = 0
    for rec in records:
        cur.execute(query, rec)
        result = cur.fetchone()
        if result is None:
            continue  # 값이 동일해 WHERE 절에 걸려 스킵된 기존 행
        if result[0]:
            updated += 1
        else:
            inserted += 1
    return inserted, updated


def update_indicator(cur, indicator_code):
    """지표 1건 증분 적재. 종목 순회 실패 격리를 위해 예외를 밖으로 던진다 (호출부에서 격리)."""
    last_date = get_last_date(cur, indicator_code)

    if last_date == date.today():
        return {"status": "스킵 (이미 최신)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else None
    records = fetch_indicator_data(indicator_code, start_date)

    if not records:
        return {"status": "스킵 (신규 데이터 없음)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    inserted, updated = upsert_market_indicators(cur, records)
    return {
        "status": "완료",
        "rows_fetched": len(records),
        "inserted": inserted,
        "updated": updated,
        "date_range": (records[0][1], records[-1][1]),
    }


def print_kospi_change_stats(cur):
    """검증용 출력. change_rate는 저장하지 않고 LAG(value)로 계산한다."""
    cur.execute(
        """
        SELECT date, value
        FROM market_indicators
        WHERE indicator_code = 'KOSPI'
        ORDER BY date
        """
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["value"] = df["value"].astype(float)  # NUMERIC -> Decimal로 반환되어 pandas 연산과 타입 충돌
    df["change_rate"] = df["value"].pct_change() * 100
    df = df.dropna(subset=["change_rate"])

    print("\n=== KOSPI 일별 등락률 기술통계 (저장값 아님, 검증용) ===")
    print(f"평균: {df['change_rate'].mean():.4f}% / 표준편차: {df['change_rate'].std():.4f}%")
    print(f"최대: {df['change_rate'].max():.4f}% ({df.loc[df['change_rate'].idxmax(), 'date']})")
    print(f"최소: {df['change_rate'].min():.4f}% ({df.loc[df['change_rate'].idxmin(), 'date']})")

    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q")
    quarterly = df.groupby("quarter")["change_rate"].apply(lambda s: s.abs().mean())
    print("\n분기별 평균 |등락률|:")
    for q, v in quarterly.tail(8).items():
        print(f"  {q}: {v:.4f}%")


if __name__ == "__main__":
    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")

        with conn.cursor() as cur:
            upsert_indicator_meta(cur)

            results = {}
            for code in INDICATOR_MAP:
                try:
                    results[code] = update_indicator(cur, code)
                except Exception as e:
                    print(f"❌ {code}: 적재 실패: {e}")
                    results[code] = {"status": f"실패 ({e})", "rows_fetched": 0, "inserted": 0, "updated": 0}

        conn.commit()

        print("\n=== 처리 결과 요약 ===")
        for code, r in results.items():
            print(f"{code}: {r['status']} | fetched={r['rows_fetched']} inserted={r['inserted']} updated={r['updated']}"
                  + (f" | range={r['date_range']}" if "date_range" in r else ""))
            if r["status"].startswith("실패"):
                print(f"  ⚠️ {code}: 적재 실패 — 확인 필요")

        with conn.cursor() as cur:
            print_kospi_change_stats(cur)