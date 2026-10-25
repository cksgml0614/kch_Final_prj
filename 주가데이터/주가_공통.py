# 주가_공통.py — 종목 일봉(OHLCV·등락률)을 FinanceDataReader에서 받아 daily_stock_prices에 넣는 공용 로직.
# 파이프라인 단계: 수집. 주요 진입 함수: update_stock_data(증분·초기적재), force_refetch_stock(기간 강제 재수집).
# 주가_초기적재·주가_일일수집·가격예측.종목_월간갱신이 이 파일만 import한다(두 수집 스크립트는 서로를 참조하지 않음).
# 배경: 결정기록.md의 "증분 수집은 덮어쓰지 않고, 정정은 --force로만", "확정 컷오프 전 KST 오늘 행은 저장하지 않는다"

import FinanceDataReader as fdr
from datetime import timedelta
from constants import ACTIVE_TICKERS, last_storable_date
from db_manager import get_db_connection

TICKERS = ACTIVE_TICKERS  # 종목 마스터는 constants.ACTIVE_TICKERS. 이 이름은 기존 import 호환용


def get_last_date(cur, ticker):
    """
    설명: daily_stock_prices에서 종목의 가장 최신 날짜를 돌려준다.
    주의 사항: 행이 없으면 None. 호출자의 커서를 쓴다(커밋하지 않음).
    입력 타입->반환 타입: (psycopg.Cursor, str) -> datetime.date|None
    """
    query = "SELECT MAX(date) FROM daily_stock_prices WHERE ticker = %s"
    cur.execute(query, (ticker,))
    result = cur.fetchone()[0]
    return result


def needs_initial_load(ticker):
    """
    설명: 종목에 저장된 행이 하나도 없으면(초기적재 대상) True를 돌려준다.
    주의 사항: DB 읽기. 접속 실패는 예외로 올라간다.
    입력 타입->반환 타입: (str) -> bool
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            return get_last_date(cur, ticker) is None


def update_stock_data(ticker, start_date_override=None):
    """
    설명: 종목 하나를 수집해 적재하고 결과 상태 문자열("완료 (N건)", "스킵 (...)", "실패 (...)")을 돌려준다.
    주의 사항: start_date_override(초기적재)가 있으면 그 날짜부터, 없으면 DB 최신일+1부터 받는다. 행이 없는 종목은 증분에서 건너뛴다.
    ON CONFLICT DO NOTHING이라 기존 행은 바꾸지 않는다. 컷오프 전 KST 오늘 행은 버린다.
    입력 타입->반환 타입: (str, str|None) -> str
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

            # 컷오프(constants.CLOSE_FINAL_CUTOFF_KST) 전 KST 오늘 행은 장중 잠정값이라 받지 않는다.
            # 저장하지 않으면 MAX(date)가 어제에 머물러 다음 실행이 MAX+1부터 확정값을 다시 받는다.
            last_ok = last_storable_date()
            if last_date and last_date >= last_ok:
                print(f"✅ {ticker}: 이미 최신 데이터입니다(저장 가능 마지막 날짜 {last_ok}).")
                return "스킵 (이미 최신)"

            df = fdr.DataReader(ticker, start_date)
            if not df.empty:
                df = df[df.index.date <= last_ok]

            if df.empty:
                print(f"📍 {ticker}: 새로 추가할 데이터가 없습니다 (주말/휴장일 또는 확정 전).")
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


# ── 기간 지정 강제 재수집 ──────────────────────────────────────────────
# 증분 경로가 덮어쓰지 않으므로, 소스에서 정정된 값은 이 경로(CLI: 주가_일일수집 --force)로만 반영된다.

_PRICE_COLS = ("open", "high", "low", "close", "volume", "change_rate")


def force_refetch_stock(cur, ticker, start, end, dry_run=False):
    """
    설명: [start, end]를 소스에서 다시 받아 DB에 없는 행은 넣고 값이 다른 행은 덮어쓴 뒤 건수 dict를 돌려준다.
    주의 사항: 커밋은 호출자가 한다. dry_run이면 쓰지 않고 변경 예정만 출력. 소스에 없는 DB 행은 지우지 않고 경고만.
    end는 저장 가능 마지막 날짜(컷오프)로 자른다.
    입력 타입->반환 타입: (psycopg.Cursor, str, date, date, bool) -> dict
    """
    end = min(end, last_storable_date())  # --force도 확정 전 KST 오늘 행은 쓰지 않는다
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
        """
        설명: DB 행 값과 소스 행 값이 하나라도 다르면 True.
        주의 사항: float로 바꿔 1e-9 넘게 다를 때만 다르다고 본다(NUMERIC·int 표현 차이 무시).
        입력 타입->반환 타입: (tuple, tuple) -> bool
        """
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
