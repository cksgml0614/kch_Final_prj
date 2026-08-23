# 지표_초기적재.py
# 지표_공통.py만 참조한다(지표_일일수집.py를 import하지 않음 — 2026-08-23 재작업,
# 뉴스_공통 패턴과 통일). FDR/ECOS 두 로더 모두 애초에 "전체 기간 조회"가 초기적재 동작
# 그 자체이므로(고정 시작일 상수 없음, 사람 확인), only_uninitialized=True로 호출해
# 아직 한 번도 적재 안 된 지표만 대상으로 삼는다.

from db_manager import get_db_connection
from 시장지표.지표_공통 import load_ecos_indicators, load_fdr_indicators, _print_section_summary

if __name__ == "__main__":
    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")

        with conn.cursor() as cur:
            try:
                fdr_results = load_fdr_indicators(cur, only_uninitialized=True)
            except Exception as e:
                print(f"❌ [FDR] 초기적재 섹션 전체 실패: {e}")
                fdr_results = {}
        conn.commit()

        with conn.cursor() as cur:
            try:
                ecos_results = load_ecos_indicators(cur, only_uninitialized=True)
            except Exception as e:
                print(f"❌ [ECOS] 초기적재 섹션 전체 실패: {e}")
                ecos_results = {}
        conn.commit()

        _print_section_summary("FDR", fdr_results)
        _print_section_summary("ECOS", ecos_results)
