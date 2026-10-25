# 지표_일일수집.py — FDR·ECOS 지표 증분 적재 CLI(기본), 기간 지정 강제 재수집(--force).
# 파이프라인 단계: 수집. 진입: python -m 시장지표.지표_일일수집 (Actions daily_data_collection.yml·daily_pipeline.bat은 인자 없이 호출).
#   정정 예) python -m 시장지표.지표_일일수집 --force --start 2026-09-17 --codes KOSPI KOSDAQ --dry-run
# 증분 모드는 모든 upsert·commit 뒤에 일별 지표 신선도를 점검하고, 적재 실패나 한도 초과가 있으면 exit 1로 끝난다.
# 배경: 결정기록.md의 "일별 지표 신선도 점검과 종료 코드"

import argparse
import sys
from datetime import date

from config import redact
from db_manager import get_db_connection
from 시장지표.지표_공통 import (
    ECOS_INDICATOR_META, FDR_INDICATOR_MAP, print_section_summary, check_daily_indicator_lag,
    force_refetch_indicator, load_ecos_indicators, load_fdr_indicators,
)


def _parse_args():
    """
    설명: CLI 인자를 읽는다. --start/--end/--codes/--dry-run은 --force와 함께만 허용하고, --force에는 --start가 필요하다.
    주의 사항: 잘못된 조합은 argparse 오류(exit 2)로 끝난다.
    입력 타입->반환 타입: () -> argparse.Namespace
    """
    p = argparse.ArgumentParser(description="FDR+ECOS 지표 증분 적재(기본) / 기간 지정 강제 재수집(--force)")
    p.add_argument("--force", action="store_true", help="[--start, --end] 구간을 소스에서 다시 받아 덮어쓴다")
    p.add_argument("--start", type=date.fromisoformat, help="--force 시작일 YYYY-MM-DD(필수)")
    p.add_argument("--end", type=date.fromisoformat, default=None, help="--force 종료일(기본: 오늘)")
    p.add_argument("--codes", nargs="+", default=None,
                   help="--force 대상 지표 코드(기본: FDR+ECOS 전체). 예: KOSPI KOSDAQ")
    p.add_argument("--dry-run", action="store_true", help="--force와 함께: DB에 쓰지 않고 변경 예정 내역만 출력")
    args = p.parse_args()
    if not args.force and (args.start or args.end or args.codes or args.dry_run):
        p.error("--start/--end/--codes/--dry-run은 --force와 함께만 쓸 수 있습니다")
    if args.force and not args.start:
        p.error("--force에는 --start가 필요합니다")
    return args


def run_force(args):
    """
    설명: 지정 지표·기간을 강제 재수집하고 요약을 출력한다. 소스에서 0건을 받은 지표는 경고한다.
    주의 사항: 알 수 없는 코드가 있으면 시작 전에 SystemExit. 지표마다 커밋(dry-run이면 롤백). 종료 코드는 바꾸지 않는다.
    입력 타입->반환 타입: (argparse.Namespace) -> None
    """
    end = args.end or date.today()
    codes = args.codes or list(FDR_INDICATOR_MAP) + list(ECOS_INDICATOR_META)
    unknown = [c for c in codes if c not in FDR_INDICATOR_MAP and c not in ECOS_INDICATOR_META]
    if unknown:
        raise SystemExit(f"알 수 없는 지표 코드: {unknown}")
    print(f"=== 강제 재수집 {'(dry-run) ' if args.dry_run else ''}{args.start} ~ {end}: {codes} ===")
    results = {}
    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")
        for code in codes:
            with conn.cursor() as cur:
                try:
                    results[code] = force_refetch_indicator(cur, code, args.start, end, dry_run=args.dry_run)
                except Exception as e:
                    print(f"❌ {code}: 강제 재수집 실패: {redact(e)}")
                    results[code] = {"status": f"실패 ({redact(e)})", "rows_fetched": 0, "inserted": 0, "updated": 0}
            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
    print_section_summary("강제 재수집", results)
    for code, r in results.items():
        if r["rows_fetched"] == 0 and not r["status"].startswith("실패"):
            print(f"  ⚠️ {code}: 소스에서 받은 행이 0건 — 기간/소스 상태 확인 필요")


def run_incremental():
    """
    설명: FDR·ECOS 섹션을 차례로 적재·커밋하고 요약과 일별 지표 신선도를 출력한다.
    주의 사항: 섹션 전체 실패, 지표 적재 실패, lag 한도 초과가 하나라도 있으면 저장이 끝난 뒤 sys.exit(1). 경고 수준 lag는 종료 코드에 넣지 않는다.
    입력 타입->반환 타입: () -> None
    """
    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")

        with conn.cursor() as cur:
            try:
                fdr_results = load_fdr_indicators(cur)
            except Exception as e:
                print(f"❌ [FDR] 섹션 전체 실패: {e}")
                fdr_results = {}
        conn.commit()

        with conn.cursor() as cur:
            try:
                ecos_results = load_ecos_indicators(cur)
            except Exception as e:
                print(f"❌ [ECOS] 섹션 전체 실패: {redact(e)}")
                ecos_results = {}
        conn.commit()

        print_section_summary("FDR", fdr_results)
        print_section_summary("ECOS", ecos_results)

        problems = []
        if not fdr_results:
            problems.append("FDR 섹션 전체 실패")
        if not ecos_results:
            problems.append("ECOS 섹션 전체 실패")
        for code, r in {**fdr_results, **ecos_results}.items():
            if r["status"].startswith("실패"):
                problems.append(f"{code}: {r['status']}")

        with conn.cursor() as cur:
            lag = check_daily_indicator_lag(cur)
        conn.rollback()  # 조회만 했으므로 트랜잭션만 닫는다
        print("\n=== 일별 지표 신선도 (기준: daily_stock_prices 최신 거래일) ===")
        for code, r in lag.items():
            print(f"{code}: {r['status']} | 최신={r['last_date']} 기준={r['ref_date']} "
                  f"lag={r['lag']}거래일 (실패 한도 {r['limit']})")
            if r["status"] == "실패":
                problems.append(f"{code}: {r['lag']}거래일 뒤처짐(최신 {r['last_date']})")

    if problems:
        print("\n❌ 지표 수집 문제 — 저장은 모두 끝났고, 종료 코드만 실패로 남긴다:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


if __name__ == "__main__":
    args = _parse_args()
    if args.force:
        run_force(args)
    else:
        run_incremental()
