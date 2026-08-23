# 뉴스_일일수집.py
# 신규 작성(2026-08-23, 사람 확인 완료). 주가데이터/시장지표 로더와 같은 패턴 —
# 종목별 daily_news(source='search_backfill') 최신 날짜 다음날부터 오늘까지를 하루씩
# 캐치업한다. 별도 체크포인트 파일 없음 — DB의 MAX(date) 자체가 체크포인트 역할.
#
# 뉴스_최초적재.py와의 역할 분리: 며칠 밀린 정도(기본 14일 이하)는 이 파일이 알아서 순차
# 캐치업하고, 그보다 오래 방치되어 공백이 크면(예: 몇 주 이상) 자동으로 그 전체 기간을 다
# 훑지 않고 경고만 출력한 뒤 중단한다 — 그 경우 뉴스_최초적재.py를 명시적으로 사용할 것.
#daily_stock_prices
# ⚠️ 동시 실행 금지: 뉴스_최초적재.py나 다른 search.naver.com 크롤러와 동시에 돌리면
# 요청 빈도 제한(403)이 유발된다(2026-08-23 실측).

from datetime import date, datetime, timedelta

from db_manager import get_db_connection
from 뉴스데이터.뉴스_공통 import SOURCE_NAME, crawl_day, upsert_articles

# 이 값을 넘는 공백은 "오래 방치됨"으로 보고 자동 캐치업하지 않는다(요청 폭주 방지 + 최초적재와
# 역할이 겹치지 않도록). 필요하면 사람이 뉴스_최초적재.py를 명시적으로 돌려야 한다.
#
# 근거(2026-08-23 실측, 조기종료 휴리스틱 적용 후): 005930 기준 일평균 소요 19.47초/일
# (1주일 136.3초 ÷ 7일). 14일 × 19.47초 ≈ 4.55분 — 무인 실행되는 일일수집 job이 감당하기
# 적절한 수준(수 분 내)으로 판단해 확정(사람 승인). 예산을 바꾸고 싶으면 이 계산식으로 재산정할 것
# (예: 10분 예산이면 약 30일).
MAX_CATCHUP_DAYS = 14

ACTIVE_TICKERS = ["005930"]  # 현재 활성 종목만(2026-08-23 사람 결정, STOCK_NAMES 전체가 아님)


def get_last_collected_date(cur, ticker):
    """source='search_backfill' 기준으로만 확인한다 — legacy/finance_crawl 등 다른 소스와
    섞이면 안 된다(사람 확인 완료)."""
    cur.execute(
        "SELECT MAX(date) FROM daily_news WHERE ticker = %s AND source = %s",
        (ticker, SOURCE_NAME),
    )
    return cur.fetchone()[0]


def collect_daily(ticker):
    """
    ticker 1건에 대해 MAX(date)+1 ~ 오늘까지를 하루씩 순차 캐치업한다(밀린 날짜 전부 처리 —
    "어제 하루"만이 아니다). 공백이 MAX_CATCHUP_DAYS를 넘으면 캐치업하지 않고 경고만 반환.
    """
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("DB 연결 실패 — collect_daily")

    try:
        with conn.cursor() as cur:
            last_date = get_last_collected_date(cur, ticker)

        today = date.today()
        if last_date is None:
            print(f"⚠️ [{ticker}] search_backfill 이력이 전혀 없습니다 — 뉴스_최초적재.py를 먼저 실행하세요.")
            return {"status": "스킵 (최초적재 필요)", "inserted": 0}

        if last_date >= today:
            print(f"✅ [{ticker}] 이미 최신입니다 (최근 수집일: {last_date}).")
            return {"status": "스킵 (이미 최신)", "inserted": 0}

        gap_days = (today - last_date).days
        if gap_days > MAX_CATCHUP_DAYS:
            print(
                f"⚠️ [{ticker}] 공백이 {gap_days}일로 너무 큽니다(기준 {MAX_CATCHUP_DAYS}일) — "
                f"자동 캐치업하지 않습니다. 뉴스_최초적재.py를 --start {(last_date + timedelta(days=1)).isoformat()} "
                f"--end {today.isoformat()}로 명시적으로 실행하세요."
            )
            return {"status": f"스킵 (공백 {gap_days}일 초과, 최초적재 필요)", "inserted": 0}

        print(f"🔄 [{ticker}] {last_date}까지 수집됨 — {last_date + timedelta(days=1)}부터 {today}까지 캐치업합니다.")

        cum_inserted = 0
        d = last_date + timedelta(days=1)
        while d <= today:
            day_result = crawl_day(ticker, d)
            if day_result["blocked"] or day_result["degraded"]:
                reason = "차단 의심" if day_result["blocked"] else "응답 저하 의심"
                print(f"   🛑 [{ticker}] {d}: {reason} — 이 날짜는 건너뛰고 다음 실행 때 재시도합니다.")
                d += timedelta(days=1)
                continue

            kept = [{**it, "published_at": datetime(d.year, d.month, d.day)} for it in day_result["kept"]]
            upsert_result = upsert_articles(conn, ticker, kept, source=SOURCE_NAME)
            cum_inserted += upsert_result["inserted"]
            print(
                f"   {d}: 필터통과 {len(kept)}건 (신규 {upsert_result['inserted']}/중복 {upsert_result['duplicate']}"
                f"/실패 {upsert_result['failed']})"
            )
            d += timedelta(days=1)

        return {"status": "완료", "inserted": cum_inserted}
    finally:
        conn.close()


if __name__ == "__main__":
    results = {}
    for ticker in ACTIVE_TICKERS:
        try:
            results[ticker] = collect_daily(ticker)
        except Exception as e:
            print(f"❌ [{ticker}] 일일수집 실패: {e}")
            results[ticker] = {"status": f"실패 ({e})", "inserted": 0}

    print("\n=== 종목별 일일수집 결과 요약 ===")
    for ticker, r in results.items():
        print(f"{ticker}: {r['status']} (신규 {r['inserted']}건)")
