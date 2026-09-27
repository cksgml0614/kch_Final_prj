# 주가_공통.py
# 주가_초기적재.py / 주가_일일수집.py가 대등하게 참조하는 공용 로직(2026-08-23, 뉴스 트랙의
# 뉴스_공통.py 패턴과 통일). 두 파일 모두 이 파일만 import하고 서로를 참조하지 않는다.

import FinanceDataReader as fdr
from datetime import timedelta, date
from constants import ACTIVE_TICKERS
from db_manager import get_db_connection

TICKERS = ACTIVE_TICKERS  # 종목 마스터는 constants.py로 중앙화(2026-08-30). 이름은 하위 호환 유지


def get_last_date(cur, ticker):
    """DB에서 해당 종목의 가장 최신 날짜를 가져옵니다."""
    query = "SELECT MAX(date) FROM daily_stock_prices WHERE ticker = %s"
    cur.execute(query, (ticker,))
    result = cur.fetchone()[0]
    return result


def _needs_initial_load(ticker):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            return get_last_date(cur, ticker) is None


def update_stock_data(ticker, start_date_override=None):
    """
    종목 1건 수집+적재. start_date_override가 주어지면(초기적재용) last_date 유무와 무관하게
    그 날짜부터 수집한다. 없으면 기존 동작(증분: last_date+1, 최초 실행: 스킵)을 따른다.
    """
    with get_db_connection() as conn:
        if not conn:
            return "실패 (DB 연결 없음)"
        with conn.cursor() as cur:
            last_date = get_last_date(cur, ticker)

            if start_date_override:
                start_date = start_date_override
                print(f"🆕 {ticker}: 초기적재 지정 시작일 {start_date}부터 전체 적재를 시작합니다.")
            elif last_date:
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"🔄 {ticker}: {last_date}까지 데이터가 있네요. {start_date}부터 업데이트를 시작합니다.")
            else:
                print(f"⚠️ {ticker}: 저장된 데이터가 없습니다 — 일일수집 대상 아님. 주가_초기적재.py를 먼저 실행하세요.")
                return "스킵 (초기적재 필요)"

            if last_date == date.today():
                print(f"✅ {ticker}: 이미 최신 데이터입니다.")
                return "스킵 (이미 최신)"

            df = fdr.DataReader(ticker, start_date)

            if df.empty:
                print(f"📍 {ticker}: 새로 추가할 데이터가 없습니다 (주말/휴장일 등).")
                return "스킵 (신규 데이터 없음)"

            df = df.reset_index()
            df['ticker'] = ticker
            df = df.fillna(0)

            data_list = [
                (row['ticker'], row['Date'].date(), int(row['Open']), int(row['High']),
                 int(row['Low']), int(row['Close']), int(row['Volume']), float(row['Change']))
                for _, row in df.iterrows()
            ]

            insert_query = """
                           INSERT INTO daily_stock_prices (ticker, date, open, high, low, close, volume, change_rate)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (ticker, date) DO NOTHING; \
                           """
            cur.executemany(insert_query, data_list)
            print(f"🚀 {ticker}: {len(data_list)}건의 새로운 데이터 적재 완료!")
            return f"완료 ({len(data_list)}건)"


# ── 기간 지정 강제 재수집(2026-09-27) ────────────────────────────────────
# update_stock_data()는 last_date+1부터만 받고 INSERT도 ON CONFLICT DO NOTHING이라, 이미 적재된
# 날짜의 값이 소스에서 정정돼도 절대 반영되지 않는다. 오염/정정 대응용으로 [start, end]를 다시
# 받아 값이 다른 행만 덮어쓰는 별도 경로를 둔다(기존 증분 경로는 그대로 — DO NOTHING 유지).
# CLI: 주가_일일수집.py --force. dry_run=True면 변경 예정 내역만 출력.

_PRICE_COLS = ("open", "high", "low", "close", "volume", "change_rate")


def force_refetch_stock(cur, ticker, start, end, dry_run=False):
    df = fdr.DataReader(ticker, start.isoformat(), end.isoformat())
    records = []
    if not df.empty:
        df = df.reset_index().fillna(0)
        records = [
            (ticker, r["Date"].date(), int(r["Open"]), int(r["High"]), int(r["Low"]),
             int(r["Close"]), int(r["Volume"]), float(r["Change"]))
            for _, r in df.iterrows()
        ]
    cur.execute(
        f"SELECT date, {', '.join(_PRICE_COLS)} FROM daily_stock_prices "
        "WHERE ticker = %s AND date BETWEEN %s AND %s",
        (ticker, start, end),
    )
    existing = {row[0]: row[1:] for row in cur.fetchall()}

    def _differs(old, new):
        return any(abs(float(o) - float(n)) > 1e-9 for o, n in zip(old, new))

    new_rows = [r for r in records if r[1] not in existing]
    changed = [r for r in records if r[1] in existing and _differs(existing[r[1]], r[2:])]
    source_missing = sorted(set(existing) - {r[1] for r in records})

    tag = "[dry-run] " if dry_run else ""
    for r in changed:
        old = existing[r[1]]
        diffs = ", ".join(f"{c} {o}->{n}" for c, o, n in zip(_PRICE_COLS, old, r[2:]) if abs(float(o) - float(n)) > 1e-9)
        print(f"  {tag}{ticker} {r[1]}: {diffs} (정정)")
    for r in new_rows:
        print(f"  {tag}{ticker} {r[1]}: close={r[5]} (신규)")
    if source_missing:
        print(f"  ⚠️ {ticker}: DB에는 있으나 소스에 없는 날짜 {len(source_missing)}건(삭제하지 않음): {source_missing[:5]}")

    if not dry_run and (new_rows or changed):
        cur.executemany(
            """
            INSERT INTO daily_stock_prices (ticker, date, open, high, low, close, volume, change_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
                volume = EXCLUDED.volume, change_rate = EXCLUDED.change_rate
            """,
            new_rows + changed,
        )
    return {"fetched": len(records), "inserted": len(new_rows), "updated": len(changed),
            "status": "dry-run" if dry_run else "완료"}
