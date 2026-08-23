# 지표_공통.py
# 지표_초기적재.py / 지표_일일수집.py가 대등하게 참조하는 공용 로직(2026-08-23 재작업,
# 뉴스_공통.py 패턴과 통일). 두 파일 모두 이 파일만 import하고 서로를 참조하지 않는다.
# FDR(KOSPI/KOSDAQ/USD_KRW)+ECOS(거시지표 6종) 두 소스를 함수 단위로 명확히 분리했고,
# 호출부(초기적재/일일수집)가 각각 독립적으로 예외 처리해 한쪽이 실패해도 다른 쪽은
# 계속 진행되게 한다. 공통 upsert 로직(db_utils.py)은 그대로 재사용 — G-1 계약(적재 순서,
# published_date 등)을 한 곳에서만 지킨다.


import sys
from calendar import monthrange
from datetime import date, timedelta

import FinanceDataReader as fdr
import requests

from config import Config
from db_manager import get_db_connection
from 시장지표.db_utils import get_last_date, upsert_indicator_meta, upsert_market_indicators

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── FDR(KOSPI/KOSDAQ/USD_KRW) ──────────────────────────────────────────
FDR_INDICATOR_MAP = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
    "USD_KRW": "USD/KRW",
}

FDR_INDICATOR_META = {
    "KOSPI": {
        "name": "코스피 지수", "source": "FDR", "frequency": "D", "unit": "point",
        "note": "FDR 심볼: KS11 (alias 매핑, source_symbol 컬럼 없음)",
    },
    "KOSDAQ": {
        "name": "코스닥 지수", "source": "FDR", "frequency": "D", "unit": "point",
        "note": "FDR 심볼: KQ11 (alias 매핑, source_symbol 컬럼 없음)",
    },
    "USD_KRW": {
        "name": "원/달러 환율", "source": "FDR", "frequency": "D", "unit": "KRW",
        "note": "FDR 심볼: USD/KRW (alias 매핑, 1달러당 원화, value=Close 컬럼)",
    },
}


def fetch_fdr_indicator_data(indicator_code, start_date=None):
    """FDR에서 alias에 대응하는 심볼 데이터를 조회해 적재용 튜플 리스트로 가공한다.
    start_date=None이면 FDR 기본 동작(전체 기간 조회)을 따른다 — 이 소스는 애초에
    고정 초기 시작일 개념이 없다(2026-08-23 사람 확인, constants.py에 상수화하지 않음)."""
    symbol = FDR_INDICATOR_MAP[indicator_code]
    df = fdr.DataReader(symbol, start_date) if start_date else fdr.DataReader(symbol)

    if df.empty:
        return []

    df.index.name = "Date"
    df = df.reset_index()
    df = df.dropna(subset=["Close"])

    return [
        (indicator_code, row["Date"].date(), round(float(row["Close"]), 4), row["Date"].date())
        for _, row in df.iterrows()
    ]


def update_fdr_indicator(cur, indicator_code, start_date=None):
    """FDR 지표 1건 증분 적재. start_date가 주어지면(초기적재용) 그 날짜부터, 아니면 last_date+1부터."""
    last_date = get_last_date(cur, indicator_code)

    if last_date == date.today():
        return {"status": "스킵 (이미 최신)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    if start_date:
        fetch_start = start_date
    else:
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else None

    records = fetch_fdr_indicator_data(indicator_code, fetch_start)

    if not records:
        return {"status": "스킵 (신규 데이터 없음)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    inserted, updated = upsert_market_indicators(cur, records)
    return {
        "status": "완료", "rows_fetched": len(records), "inserted": inserted, "updated": updated,
        "date_range": (records[0][1], records[-1][1]),
    }


def load_fdr_indicators(cur, only_uninitialized=False):
    """FDR 지표(KOSPI/KOSDAQ/USD_KRW) 전체를 처리하고 결과 dict를 반환한다.
    only_uninitialized=True면 아직 한 번도 적재 안 된(last_date 없는) 지표만 대상으로 한다(초기적재용)."""
    upsert_indicator_meta(cur, FDR_INDICATOR_META)

    results = {}
    for code in FDR_INDICATOR_MAP:
        try:
            if only_uninitialized and get_last_date(cur, code) is not None:
                results[code] = {"status": "스킵 (이미 초기적재됨)", "rows_fetched": 0, "inserted": 0, "updated": 0}
                continue
            # FDR은 고정 시작일 없이 "전체 기간 조회"가 초기적재 동작 그 자체이므로 start_date=None 그대로 둔다
            results[code] = update_fdr_indicator(cur, code, start_date=None)
        except Exception as e:
            print(f"❌ [FDR] {code}: 적재 실패: {e}")
            results[code] = {"status": f"실패 ({e})", "rows_fetched": 0, "inserted": 0, "updated": 0}
    return results


# ── ECOS(기준금리·국고채3/10년·CPI·M2·선행지수) ─────────────────────────
ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
PAGE_SIZE = 10000

ECOS_INDICATOR_META = {
    "ECOS_722Y001_0101000": {
        "name": "한국은행 기준금리", "source": "ECOS", "frequency": "D", "unit": "연%",
        "ecos_stat_code": "722Y001", "ecos_item_code": "0101000",
        "note": "일별. published_date = date",
    },
    "ECOS_817Y002_010200000": {
        "name": "국고채(3년)", "source": "ECOS", "frequency": "D", "unit": "연%",
        "ecos_stat_code": "817Y002", "ecos_item_code": "010200000",
        "note": "일별. published_date = date",
    },
    "ECOS_817Y002_010210000": {
        "name": "국고채(10년)", "source": "ECOS", "frequency": "D", "unit": "연%",
        "ecos_stat_code": "817Y002", "ecos_item_code": "010210000",
        "note": "일별. published_date = date",
    },
    "ECOS_901Y009_0": {
        "name": "소비자물가지수(총지수)", "source": "ECOS", "frequency": "M", "unit": "2020=100",
        "ecos_stat_code": "901Y009", "ecos_item_code": "0",
        "note": "월별. date=기준월 1일. 공표일 확인 어려워 published_date=기준월 다음달 말일로 "
                "보수적 설정 (개정 이력 미보존 — 최신값만 유지)",
    },
    "ECOS_161Y005_BBHS00": {
        "name": "M2(평잔, 계절조정계열)", "source": "ECOS", "frequency": "M", "unit": "십억원",
        "ecos_stat_code": "161Y005", "ecos_item_code": "BBHS00",
        "note": "월별. date=기준월 1일. 공표일 확인 어려워 published_date=기준월 다음달 말일로 "
                "보수적 설정 (개정 이력 미보존 — 최신값만 유지)",
    },
    "ECOS_901Y067_I16E": {
        "name": "선행지수 순환변동치", "source": "ECOS", "frequency": "M", "unit": "2020=100",
        "ecos_stat_code": "901Y067", "ecos_item_code": "I16E",
        "note": "월별. date=기준월 1일. 공표일 확인 어려워 published_date=기준월 다음달 말일로 "
                "보수적 설정 (개정 이력 미보존 — 최신값만 유지)",
    },
}


def _next_month_end(d):
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(year, month, monthrange(year, month)[1])


def fetch_ecos_series(stat_code, item_code, cycle, start, end):
    """ECOS StatisticSearch를 페이지 분할해 전체 구간을 조회한다."""
    all_rows = []
    begin = 1
    while True:
        end_idx = begin + PAGE_SIZE - 1
        url = (f"{ECOS_BASE_URL}/{Config.ECOS_API_KEY}/json/kr/{begin}/{end_idx}/"
               f"{stat_code}/{cycle}/{start}/{end}/{item_code}")
        r = requests.get(url, timeout=30)
        r.encoding = "utf-8"
        data = r.json()

        if "RESULT" in data:
            code = data["RESULT"].get("CODE", "")
            if code == "INFO-200":
                break
            raise RuntimeError(f"ECOS API 오류 [{code}]: {data['RESULT'].get('MESSAGE', '')}")

        if "StatisticSearch" not in data:
            raise RuntimeError(f"ECOS API 응답 형식 예상과 다름: {data}")

        rows = data["StatisticSearch"].get("row", [])
        all_rows.extend(rows)

        total = int(data["StatisticSearch"].get("list_total_count", len(rows)))
        if len(all_rows) >= total or not rows:
            break
        begin = end_idx + 1

    return all_rows


def parse_ecos_records(indicator_code, cycle, rows):
    records = []
    skipped_missing = 0
    for row in rows:
        raw_value = row.get("DATA_VALUE")
        if raw_value is None or raw_value == "":
            skipped_missing += 1
            continue
        value = round(float(raw_value), 4)

        if cycle == "D":
            d = date(int(row["TIME"][:4]), int(row["TIME"][4:6]), int(row["TIME"][6:8]))
            published = d
        elif cycle == "M":
            d = date(int(row["TIME"][:4]), int(row["TIME"][4:6]), 1)
            published = _next_month_end(d)
        else:
            raise ValueError(f"미지원 주기: {cycle}")

        records.append((indicator_code, d, value, published))

    return records, skipped_missing


def update_ecos_indicator(cur, indicator_code, meta, start_override=None):
    """ECOS 지표 1건 증분 적재. start_override가 주어지면(초기적재용) 그 값으로 시작하고,
    아니면 기존 동작(증분: last_date 다음 / 최초: '19000101' 센티널 = 전체 기간)을 따른다."""
    cycle = meta["frequency"]
    last_date = get_last_date(cur, indicator_code)

    if last_date == date.today():
        return {"status": "스킵 (이미 최신)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    if start_override:
        start = start_override
        end = date.today().strftime("%Y%m%d") if cycle == "D" else date.today().strftime("%Y%m")
    elif cycle == "D":
        start = (last_date + timedelta(days=1)).strftime("%Y%m%d") if last_date else "19000101"
        end = date.today().strftime("%Y%m%d")
    else:  # "M"
        if last_date:
            y, m = (last_date.year + 1, 1) if last_date.month == 12 else (last_date.year, last_date.month + 1)
            start = f"{y:04d}{m:02d}"
        else:
            start = "190001"
        end = date.today().strftime("%Y%m")

    rows = fetch_ecos_series(meta["ecos_stat_code"], meta["ecos_item_code"], cycle, start, end)
    if not rows:
        return {"status": "스킵 (신규 데이터 없음)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    records, skipped_missing = parse_ecos_records(indicator_code, cycle, rows)
    if not records:
        return {"status": "스킵 (전량 결측)", "rows_fetched": len(rows), "inserted": 0, "updated": 0}

    inserted, updated = upsert_market_indicators(cur, records)
    result = {
        "status": "완료", "rows_fetched": len(records), "inserted": inserted, "updated": updated,
        "date_range": (records[0][1], records[-1][1]),
    }
    if skipped_missing:
        result["skipped_missing"] = skipped_missing
    return result


def load_ecos_indicators(cur, only_uninitialized=False):
    """ECOS 지표 6종 전체를 처리하고 결과 dict를 반환한다.
    only_uninitialized=True면 아직 한 번도 적재 안 된 지표만 대상으로 한다(초기적재용).
    ECOS_API_KEY가 없으면 이 섹션 전체를 건너뛰고 명확히 경고한다(FDR 쪽은 계속 진행)."""
    if not Config.ECOS_API_KEY:
        print("⚠️ [ECOS] ECOS_API_KEY가 .env에 없어 ECOS 섹션 전체를 건너뜁니다.")
        return {code: {"status": "스킵 (ECOS_API_KEY 없음)", "rows_fetched": 0, "inserted": 0, "updated": 0}
                for code in ECOS_INDICATOR_META}

    upsert_indicator_meta(cur, ECOS_INDICATOR_META)

    results = {}
    for code, meta in ECOS_INDICATOR_META.items():
        try:
            if only_uninitialized and get_last_date(cur, code) is not None:
                results[code] = {"status": "스킵 (이미 초기적재됨)", "rows_fetched": 0, "inserted": 0, "updated": 0}
                continue
            results[code] = update_ecos_indicator(cur, code, meta)
        except Exception as e:
            print(f"❌ [ECOS] {code}: 적재 실패: {e}")
            results[code] = {"status": f"실패 ({e})", "rows_fetched": 0, "inserted": 0, "updated": 0}
    return results


def _print_section_summary(section_name, results):
    print(f"\n=== [{section_name}] 처리 결과 ===")
    for code, r in results.items():
        line = f"{code}: {r['status']} | fetched={r['rows_fetched']} inserted={r['inserted']} updated={r['updated']}"
        if "date_range" in r:
            line += f" | range={r['date_range']}"
        if "skipped_missing" in r:
            line += f" | 결측 스킵={r['skipped_missing']}"
        print(line)
        if r["status"].startswith("실패"):
            print(f"  ⚠️ {code}: 적재 실패 — 확인 필요")

