# 뉴스_최초적재.py
# NaverSearchBackfill.py에서 분리(2026-08-23). 명시적 날짜범위(기본값: constants.NEWS_BACKFILL_START
# ~ NEWS_BACKFILL_END)를 크게 한 번 훑는 백필 전용 — 종목별 DB 최신 날짜를 보고 캐치업하는
# 일상 운영은 뉴스_일일수집.py 담당(개념적으로 겹치지 않게 역할 분리).
#
# ⚠️ 동시 실행 금지: 뉴스_일일수집.py나 다른 search.naver.com 크롤러와 동시에 돌리면
# 요청 빈도 제한(403)이 유발된다(2026-08-23 실측).

import csv
import os
import random
import time
from datetime import date, datetime, timedelta

from constants import NEWS_BACKFILL_END, NEWS_BACKFILL_START
from db_manager import get_db_connection
from 뉴스데이터.뉴스_공통 import (
    CONSECUTIVE_BLOCKED_DAYS_LIMIT,
    REQUEST_DELAY_RANGE,
    SOURCE_NAME,
    crawl_day,
    upsert_articles,
)


def run_backfill(tickers, start_date, end_date):
    """
    tickers x [start_date, end_date] 전 날짜를 순회해 수집한다. 종목 전체 기간을 다 모은 뒤
    한 번에 upsert하므로 소규모 검증용 — 수 시간짜리 대량 실행에는 run_backfill_resumable()을 쓸 것
    (이 함수는 중간에 끊기면 그때까지 모은 것이 전부 유실된다).
    """
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("DB 연결 실패 — run_backfill")

    summary = {}
    try:
        for ticker in tickers:
            print(f"\n=== [{ticker}] search_backfill 수집 시작 ({start_date} ~ {end_date}) ===")
            ticker_kept = []
            ticker_filtered = {}
            ticker_errors = []
            d = start_date
            while d <= end_date:
                day_result = crawl_day(ticker, d)
                ticker_kept.extend(
                    {**it, "published_at": datetime(d.year, d.month, d.day)} for it in day_result["kept"]
                )
                for k, v in day_result["filtered_out"].items():
                    ticker_filtered[k] = ticker_filtered.get(k, 0) + v
                ticker_errors.extend(day_result["page_errors"])
                flag = ""
                if day_result["blocked"]:
                    flag = " 🛑차단의심"
                elif day_result["degraded"]:
                    flag = " ⚠️응답저하의심(재검증 필요)"
                elif day_result.get("early_stopped"):
                    flag = " ⏭️조기종료"
                print(
                    f"   {d}: 페이지 {day_result['pages_fetched']}개, 후보 {day_result['candidates_total']}건, "
                    f"필터 통과 {len(day_result['kept'])}건, 필터 제외 {sum(day_result['filtered_out'].values())}건{flag}"
                )
                d += timedelta(days=1)

            upsert_result = upsert_articles(conn, ticker, ticker_kept, source=SOURCE_NAME)
            summary[ticker] = {**upsert_result, "filtered_out": ticker_filtered, "page_errors": ticker_errors}
            print(
                f"   ✅ [{ticker}] 필터 통과 시도 {upsert_result['attempted']} / 신규 {upsert_result['inserted']} "
                f"/ 중복 {upsert_result['duplicate']} / 실패 {upsert_result['failed']}"
            )
            print(f"   필터 제외 사유별 집계: {ticker_filtered}")
    finally:
        conn.close()

    print("\n=== 종목별 search_backfill 수집 요약 ===")
    any_zero = False
    for ticker, s in summary.items():
        inserted = s.get("inserted", 0)
        if inserted == 0:
            any_zero = True
        print(f"  {ticker}: 신규 {inserted} / 시도 {s.get('attempted', 0)} / 필터제외 {s.get('filtered_out')}")
    if any_zero:
        print("⚠️ 0건 종목이 있습니다 — 필터가 지나치게 엄격하거나 파이프라인 문제 가능성, 재확인 필요.")

    return summary


# ── 전면 백필용: 재개 가능 + 진행 로그 ───────────────────────────────────
DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "search_backfill_progress.csv")


def _load_checkpoint(checkpoint_path):
    """이미 완료된 (ticker, date) 집합을 반환한다. 파일 없으면 빈 집합."""
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            ticker, date_str = row[0], row[1]
            done.add((ticker, date_str))
    return done


def _append_checkpoint(checkpoint_path, ticker, d, kept, inserted, duplicate, failed):
    is_new = not os.path.exists(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["ticker", "date", "kept", "inserted", "duplicate", "failed"])
        writer.writerow([ticker, d.isoformat(), kept, inserted, duplicate, failed])


def run_backfill_resumable(tickers, start_date, end_date, checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    """
    전면 백필용. run_backfill()과 차이점:
    - 하루치를 수집할 때마다 즉시 DB에 upsert하고 체크포인트 파일에 기록한다
      (run_backfill()은 종목 전체 기간을 다 모은 뒤 한 번에 upsert하므로, 수 시간짜리
      실행 중 중단되면 그때까지 수집한 것이 전부 유실된다 — 전면 백필에는 부적합).
    - 재실행 시 체크포인트에 이미 있는 (ticker, date)는 재수집 없이 건너뛴다.
    - 진행률(%)과 누적 건수를 매 날짜마다 즉시 출력(flush)한다 — 터미널이나 로그 파일
      어느 쪽으로 리다이렉트해도 실시간으로 진행 상황을 볼 수 있다.

    중단하려면 Ctrl+C 또는 프로세스 종료. 같은 인자로 다시 실행하면 이어서 진행된다.
    """
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("DB 연결 실패 — run_backfill_resumable")

    done = _load_checkpoint(checkpoint_path)
    print(f"체크포인트 로드: {checkpoint_path} — 이미 완료된 (ticker,date) {len(done)}건", flush=True)

    all_days = []
    for ticker in tickers:
        d = start_date
        while d <= end_date:
            all_days.append((ticker, d))
            d += timedelta(days=1)
    total = len(all_days)
    remaining = [(t, d) for t, d in all_days if (t, d.isoformat()) not in done]
    print(f"전체 {total}건 중 남은 작업 {len(remaining)}건 (건너뜀 {total - len(remaining)}건)", flush=True)

    cum_inserted = 0
    cum_kept = 0
    consecutive_blocked_days = 0
    try:
        for i, (ticker, d) in enumerate(remaining, 1):
            day_result = crawl_day(ticker, d)

            if day_result["blocked"] or day_result["degraded"]:
                consecutive_blocked_days += 1
                reason = "차단 의심" if day_result["blocked"] else "응답 저하 의심(필터 0건인데 후보는 있었음)"
                print(
                    f"[{i}/{len(remaining)}] {ticker} {d}: 🛑 {reason}으로 이 날짜는 건너뜀 "
                    f"(체크포인트에 기록하지 않음 — 다음 실행 때 재시도됨). "
                    f"연속 문제일수 {consecutive_blocked_days}",
                    flush=True,
                )
                if consecutive_blocked_days >= CONSECUTIVE_BLOCKED_DAYS_LIMIT:
                    print(
                        f"\n🛑🛑 연속 {consecutive_blocked_days}일 차단 감지 — 실행을 중단합니다. "
                        f"동시에 다른 크롤러 프로세스를 돌리고 있지 않은지 확인하고, 잠시(수십 분) 기다린 뒤 "
                        f"같은 명령으로 재실행하면 체크포인트에 없는 날짜부터 이어집니다.",
                        flush=True,
                    )
                    raise RuntimeError(f"search.naver.com 연속 {consecutive_blocked_days}일 차단 — 백필 중단")
                time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
                continue

            consecutive_blocked_days = 0
            kept = [{**it, "published_at": datetime(d.year, d.month, d.day)} for it in day_result["kept"]]
            upsert_result = upsert_articles(conn, ticker, kept, source=SOURCE_NAME)

            cum_kept += len(kept)
            cum_inserted += upsert_result["inserted"]
            _append_checkpoint(
                checkpoint_path, ticker, d,
                len(kept), upsert_result["inserted"], upsert_result["duplicate"], upsert_result["failed"],
            )

            pct = i / len(remaining) * 100
            print(
                f"[{i}/{len(remaining)} {pct:5.1f}%] {ticker} {d}: 필터통과 {len(kept)}건 "
                f"(신규 {upsert_result['inserted']}/중복 {upsert_result['duplicate']}/실패 {upsert_result['failed']}) "
                f"| 누적 신규 {cum_inserted}건",
                flush=True,
            )
            if day_result["page_errors"]:
                print(f"   ⚠️ {ticker} {d} 페이지 오류 {len(day_result['page_errors'])}건: {day_result['page_errors']}", flush=True)
    finally:
        conn.close()

    print(f"\n=== 완료: 이번 실행에서 신규 {cum_inserted}건 적재 (필터통과 후보 {cum_kept}건) ===", flush=True)
    return {"inserted": cum_inserted, "kept": cum_kept}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="search.naver.com 날짜범위 뉴스 백필(최초적재)")
    parser.add_argument("--tickers", nargs="+", default=["005930"], help="예: --tickers 005930")
    parser.add_argument("--start", type=str, default=NEWS_BACKFILL_START, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=NEWS_BACKFILL_END, help="YYYY-MM-DD")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    run_backfill_resumable(
        args.tickers,
        datetime.strptime(args.start, "%Y-%m-%d").date(),
        datetime.strptime(args.end, "%Y-%m-%d").date(),
        checkpoint_path=args.checkpoint,
    )
