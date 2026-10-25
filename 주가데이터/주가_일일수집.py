# 주가_일일수집.py — 활성 종목의 주가 일일 증분 수집 CLI(기본), 기간 지정 강제 재수집(--force).
# 파이프라인 단계: 수집. 진입: python -m 주가데이터.주가_일일수집 (Actions daily_data_collection.yml·daily_pipeline.bat은 인자 없이 호출).
#   정정 예) python -m 주가데이터.주가_일일수집 --force --start 2026-09-17 --tickers 005930 --dry-run
# 증분 모드는 모든 종목 저장이 끝난 뒤 실패 기준에 걸리면 exit 1로 끝난다. --force에는 이 기준을 적용하지 않는다.
# 배경: 결정기록.md의 "증분 수집의 실패 판정과 종료 코드", "증분 수집은 덮어쓰지 않고, 정정은 --force로만"

import argparse
import sys
import time
from datetime import date

from db_manager import get_db_connection
from 주가데이터.주가_공통 import TICKERS, force_refetch_stock, update_stock_data

FAIL_RATIO_LIMIT = 0.10   # 실패 종목이 활성 종목의 이 비율 이상이면 exit 1
RETRY_WAIT_SECONDS = 10   # 재시도 전 대기(일시 오류 완화)


def _parse_args():
    """
    설명: CLI 인자를 읽는다. --start/--end/--tickers/--dry-run은 --force와 함께만 허용하고, --force에는 --start가 필요하다.
    주의 사항: 잘못된 조합은 argparse 오류(exit 2)로 끝난다.
    입력 타입->반환 타입: () -> argparse.Namespace
    """
    p = argparse.ArgumentParser(description="주가 증분 적재(기본) / 기간 지정 강제 재수집(--force)")
    p.add_argument("--force", action="store_true", help="[--start, --end] 구간을 소스에서 다시 받아 덮어쓴다")
    p.add_argument("--start", type=date.fromisoformat, help="--force 시작일 YYYY-MM-DD(필수)")
    p.add_argument("--end", type=date.fromisoformat, default=None, help="--force 종료일(기본: 오늘)")
    p.add_argument("--tickers", nargs="+", default=None, help="--force 대상 종목(기본: 전체 활성 종목)")
    p.add_argument("--dry-run", action="store_true", help="--force와 함께: DB에 쓰지 않고 변경 예정 내역만 출력")
    args = p.parse_args()
    if not args.force and (args.start or args.end or args.tickers or args.dry_run):
        p.error("--start/--end/--tickers/--dry-run은 --force와 함께만 쓸 수 있습니다")
    if args.force and not args.start:
        p.error("--force에는 --start가 필요합니다")
    return args


def run_force(args):
    """
    설명: 지정 종목·기간을 강제 재수집하고 종목별 변경·실패 요약을 출력한다.
    주의 사항: 종목마다 커밋(dry-run이면 롤백). 종목별 예외는 격리해 요약에 실패로 남기고, 종료 코드는 바꾸지 않는다.
    입력 타입->반환 타입: (argparse.Namespace) -> None
    """
    end = args.end or date.today()
    tickers = args.tickers or TICKERS
    print(f"=== 강제 재수집 {'(dry-run) ' if args.dry_run else ''}{args.start} ~ {end}: {len(tickers)}종목 ===")
    results = {}
    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")
        for ticker in tickers:
            with conn.cursor() as cur:
                try:
                    results[ticker] = force_refetch_stock(cur, ticker, args.start, end, dry_run=args.dry_run)
                except Exception as e:
                    print(f"❌ {ticker}: 강제 재수집 실패: {e}")
                    results[ticker] = {"status": f"실패 ({e})", "fetched": 0, "inserted": 0, "updated": 0}
            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
    print("\n=== 처리 결과 요약 ===")
    for ticker, r in results.items():
        if r["inserted"] or r["updated"] or r["status"].startswith("실패") or r["fetched"] == 0:
            print(f"{ticker}: {r['status']} | fetched={r['fetched']} inserted={r['inserted']} updated={r['updated']}")
    n_changed = sum(1 for r in results.values() if r["inserted"] or r["updated"])
    n_failed = sum(1 for r in results.values() if r["status"].startswith("실패"))
    print(f"총 {len(results)}종목: 변경 있음 {n_changed}, 실패 {n_failed}, 나머지는 변경 없음")


def _update_one(ticker):
    """
    설명: 종목 하나를 증분 수집하고 상태 문자열을 돌려준다.
    주의 사항: 예외는 출력 후 "실패 (...)" 문자열로 바꾼다(종목별 격리).
    입력 타입->반환 타입: (str) -> str
    """
    try:
        return update_stock_data(ticker)
    except Exception as e:
        print(f"❌ {ticker}: 업데이트 실패: {e}")
        return f"실패 ({e})"


def _is_failure(status):
    """
    설명: 상태 문자열이 실패면 True.
    주의 사항: "스킵 (...)"(신규 데이터 없음·이미 최신·초기적재 필요)은 실패가 아니다.
    입력 타입->반환 타입: (str) -> bool
    """
    return status.startswith("실패")


def _warn_lagging_tickers(tickers):
    """
    설명: 종목별 최신일을 읽어 전체 최신일보다 뒤처진 종목을 경고로 출력한다.
    주의 사항: 경고만(거래정지 종목은 매일 걸리므로 실패로 보지 않음). 조회 실패도 경고 후 넘어간다.
    입력 타입->반환 타입: (list[str]) -> None
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ticker, MAX(date) FROM daily_stock_prices WHERE ticker = ANY(%s) GROUP BY ticker",
                    (list(tickers),),
                )
                last = dict(cur.fetchall())
    except Exception as e:
        print(f"⚠️ 종목별 최신일 점검 생략(조회 실패): {e}")
        return
    if not last:
        return
    newest = max(last.values())
    lagging = {t: d for t, d in last.items() if d < newest}
    if lagging:
        print(f"⚠️ 최신일({newest})보다 뒤처진 종목 {len(lagging)}개(경고만): "
              + ", ".join(f"{t}={d}" for t, d in sorted(lagging.items())))


def run_incremental():
    """
    설명: 활성 종목 전체를 증분 수집하고, 실패 종목은 RETRY_WAIT_SECONDS 뒤 1회 재시도한 다음 요약·경고를 출력한다.
    주의 사항: 성공 0종목이거나 실패가 FAIL_RATIO_LIMIT 이상이면 저장이 다 끝난 뒤 sys.exit(1). 그보다 적은 실패는 경고만.
    입력 타입->반환 타입: () -> None
    """
    results = {ticker: _update_one(ticker) for ticker in TICKERS}

    first_failed = [t for t, s in results.items() if _is_failure(s)]
    if first_failed:
        print(f"\n🔁 실패 {len(first_failed)}종목을 {RETRY_WAIT_SECONDS}초 뒤 1회 재시도: {', '.join(first_failed)}")
        time.sleep(RETRY_WAIT_SECONDS)
        for ticker in first_failed:
            results[ticker] = _update_one(ticker)

    print("\n=== 처리 결과 요약 ===")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")

    failed = [t for t, s in results.items() if _is_failure(s)]
    recovered = [t for t in first_failed if t not in failed]
    needs_initial = [t for t, s in results.items() if s.startswith("스킵 (초기적재 필요)")]
    n_total, n_failed = len(results), len(failed)
    n_success = n_total - n_failed
    print(f"\n총 {n_total}종목: 성공 {n_success}(신규 데이터 없음·이미 최신 포함), 실패 {n_failed}, "
          f"재시도 {len(first_failed)}(복구 {len(recovered)})")

    if needs_initial:
        print(f"⚠️ 초기적재 필요로 건너뛴 종목(경고만): {', '.join(needs_initial)}")
    _warn_lagging_tickers(TICKERS)

    problems = []
    if n_success == 0:
        problems.append(f"① 성공 0종목(전체 {n_total}종목)")
    if n_total and n_failed / n_total >= FAIL_RATIO_LIMIT:
        problems.append(f"② 실패 {n_failed}/{n_total}종목({n_failed / n_total:.0%}) >= {FAIL_RATIO_LIMIT:.0%}")
    if failed and not problems:
        print(f"⚠️ 실패 {n_failed}종목(기준 {FAIL_RATIO_LIMIT:.0%} 미만이라 경고만): {', '.join(failed)}")

    if problems:
        print("\n❌ 주가 수집 문제 — 저장은 모두 끝났고, 종료 코드만 실패로 남긴다:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


if __name__ == "__main__":
    args = _parse_args()
    if args.force:
        run_force(args)
    else:
        run_incremental()
