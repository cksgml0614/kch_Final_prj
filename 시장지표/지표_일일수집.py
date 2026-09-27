# 지표_일일수집.py
# 지표_공통.py만 참조한다(지표_초기적재.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). FDR+ECOS 지표를 증분 적재한다(only_uninitialized=False, 기본값).
#
# 2026-09-27: --force 모드 추가 — 이미 적재된 기간이라도 소스에서 다시 받아 값이 다르면
# 덮어쓴다(지표_공통.force_refetch_indicator 참고). 인자 없이 실행하면 기존 증분 동작 그대로
# (GitHub Actions daily_data_collection.yml은 인자 없이 호출).
#   예) python -m 시장지표.지표_일일수집 --force --start 2026-09-17 --codes KOSPI KOSDAQ --dry-run

import argparse
from datetime import date

from db_manager import get_db_connection
from 시장지표.지표_공통 import (
    ECOS_INDICATOR_META, FDR_INDICATOR_MAP, _print_section_summary,
    force_refetch_indicator, load_ecos_indicators, load_fdr_indicators,
)


def _parse_args():
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
                    print(f"❌ {code}: 강제 재수집 실패: {e}")
                    results[code] = {"status": f"실패 ({e})", "rows_fetched": 0, "inserted": 0, "updated": 0}
            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
    _print_section_summary("강제 재수집", results)
    for code, r in results.items():
        if r["rows_fetched"] == 0 and not r["status"].startswith("실패"):
            print(f"  ⚠️ {code}: 소스에서 받은 행이 0건 — 기간/소스 상태 확인 필요")


def run_incremental():
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
                print(f"❌ [ECOS] 섹션 전체 실패: {e}")
                ecos_results = {}
        conn.commit()

        _print_section_summary("FDR", fdr_results)
        _print_section_summary("ECOS", ecos_results)


if __name__ == "__main__":
    args = _parse_args()
    if args.force:
        run_force(args)
    else:
        run_incremental()
