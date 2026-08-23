# 지표_일일수집.py
# 지표_공통.py만 참조한다(지표_초기적재.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). FDR+ECOS 지표를 증분 적재한다(only_uninitialized=False, 기본값).

from db_manager import get_db_connection
from 시장지표.지표_공통 import load_ecos_indicators, load_fdr_indicators, _print_section_summary

if __name__ == "__main__":
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
