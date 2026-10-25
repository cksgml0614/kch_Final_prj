# 지표_초기적재.py — market_indicators에 행이 하나도 없는 지표만 소스의 전체 기간으로 적재하는 CLI.
# 파이프라인 단계: 수집(최초 1회, 수동). 진입: python -m 시장지표.지표_초기적재
# 고정 시작일 없이 전체 기간을 받는다(only_uninitialized=True). 섹션·지표별 예외는 격리해 요약에 남기고 종료 코드는 0이다.

from config import redact
from db_manager import get_db_connection
from 시장지표.지표_공통 import load_ecos_indicators, load_fdr_indicators, print_section_summary

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
                print(f"❌ [ECOS] 초기적재 섹션 전체 실패: {redact(e)}")
                ecos_results = {}
        conn.commit()

        print_section_summary("FDR", fdr_results)
        print_section_summary("ECOS", ecos_results)
