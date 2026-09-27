# 주가_일일수집.py
# 주가_공통.py만 참조한다(주가_초기적재.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). 종목별 DB 최신 날짜 다음날부터 오늘까지 증분 수집한다.
# 아직 데이터가 전혀 없는 종목(초기적재 필요)은 처리하지 않고 경고만 출력한다.
#
# 2026-09-27: --force 모드 추가 — 이미 적재된 기간이라도 소스에서 다시 받아 값이 다른 행을
# 덮어쓴다(주가_공통.force_refetch_stock 참고). 인자 없이 실행하면 기존 증분 동작 그대로
# (GitHub Actions daily_data_collection.yml은 인자 없이 호출).
#   예) python -m 주가데이터.주가_일일수집 --force --start 2026-09-17 --tickers 005930 --dry-run

import argparse
from datetime import date

from db_manager import get_db_connection
from 주가데이터.주가_공통 import TICKERS, force_refetch_stock, update_stock_data


def _parse_args():
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


def run_incremental():
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


if __name__ == "__main__":
    args = _parse_args()
    if args.force:
        run_force(args)
    else:
        run_incremental()
