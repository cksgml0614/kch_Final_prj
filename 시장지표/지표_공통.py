# 지표_공통.py — 시장지표 9종을 소스에서 받아 market_indicators에 넣는 공용 로직과 일별 지표 신선도 점검.
# 소스: FDR(USD_KRW), ECOS(KOSPI·KOSDAQ + 기준금리·국고채 3/10년·CPI·M2·선행지수). 소스별 함수가 따로 있고 호출부가 섹션마다 예외를 격리한다.
# 파이프라인 단계: 수집. 주요 진입 함수: load_fdr_indicators, load_ecos_indicators, check_daily_indicator_lag, force_refetch_indicator.
# 지표_초기적재·지표_일일수집이 이 파일만 import한다(서로 참조하지 않음). DB 쓰기는 db_utils를 거친다.
# 배경: 결정기록.md의 "KOSPI·KOSDAQ는 ECOS에서 받고 코드명은 유지", "일별 지표 신선도 점검과 종료 코드", "월별 지표의 published_date"


import sys
from calendar import monthrange
from datetime import date, timedelta

import FinanceDataReader as fdr
import requests

from config import Config, redact
from constants import last_storable_date
from 시장지표.db_utils import get_last_date, upsert_indicator_meta, upsert_market_indicators

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── FDR(USD_KRW) ──────────────────────────────────────────────────────
FDR_INDICATOR_MAP = {
    "USD_KRW": "USD/KRW",
}

FDR_INDICATOR_META = {
    "USD_KRW": {
        "name": "원/달러 환율", "source": "FDR", "frequency": "D", "unit": "KRW",
        "note": "FDR 심볼: USD/KRW (alias 매핑, 1달러당 원화, value=Close 컬럼)",
    },
}


def fetch_fdr_indicator_data(indicator_code, start_date=None):
    """
    설명: FDR에서 지표 심볼을 start_date부터 받아 (code, date, value=Close, published_date=date) 튜플 리스트로 돌려준다.
    주의 사항: start_date=None이면 전체 기간(FDR 기본 동작, 고정 시작일 없음). 컷오프 전 KST 오늘 행과 Close 결측 행은 뺀다.
    입력 타입->반환 타입: (str, str|None) -> list[tuple]
    """
    symbol = FDR_INDICATOR_MAP[indicator_code]
    df = fdr.DataReader(symbol, start_date) if start_date else fdr.DataReader(symbol)

    if df.empty:
        return []

    df.index.name = "Date"
    df = df.reset_index()
    df = df[df["Date"].dt.date <= last_storable_date()]  # 확정 전 KST 오늘 행 제외(주가와 같은 컷오프)
    df = df.dropna(subset=["Close"])

    return [
        (indicator_code, row["Date"].date(), round(float(row["Close"]), 4), row["Date"].date())
        for _, row in df.iterrows()
    ]


def update_fdr_indicator(cur, indicator_code, start_date=None):
    """
    설명: FDR 지표 하나를 적재하고 결과 dict(status·rows_fetched·inserted·updated·date_range)를 돌려준다.
    주의 사항: start_date가 있으면 그 날짜부터, 없으면 DB 최신일+1부터(행이 없으면 전체 기간). 커밋은 호출자가 한다.
    입력 타입->반환 타입: (psycopg.Cursor, str, str|None) -> dict
    """
    last_date = get_last_date(cur, indicator_code)

    if last_date and last_date >= last_storable_date():
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
    """
    설명: 메타를 먼저 upsert한 뒤 FDR 지표 전체를 적재하고 {code: 결과 dict}를 돌려준다.
    주의 사항: only_uninitialized=True(초기적재)면 행이 없는 지표만 처리한다. 지표별 예외는 격리해 "실패 (...)"로 남긴다.
    입력 타입->반환 타입: (psycopg.Cursor, bool) -> dict
    """
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


# ── ECOS(KOSPI/KOSDAQ·기준금리·국고채3/10년·CPI·M2·선행지수) ─────────────
ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
PAGE_SIZE = 10000

ECOS_INDICATOR_META = {
    # KOSPI/KOSDAQ는 ECOS_<통계표>_<항목> 명명 규칙의 예외다 — 피처 컬럼명·신선도 점검이 이 이름을 키로 쓴다.
    # note 문자열은 실행마다 indicator_meta.note에 저장되는 값이라, 고치면 DB가 바뀐다.
    "KOSPI": {
        "name": "코스피 지수", "source": "ECOS", "frequency": "D", "unit": "point",
        "ecos_stat_code": "802Y001", "ecos_item_code": "0001000",
        "note": "일별 종가. published_date = date. 2026-10-03 FDR(KS11)에서 ECOS로 전환 "
                "(FDR GitHub 캐시 2026-09-17 이후 미갱신). 코드명은 피처 호환 때문에 유지",
    },
    "KOSDAQ": {
        "name": "코스닥 지수", "source": "ECOS", "frequency": "D", "unit": "point",
        "ecos_stat_code": "802Y001", "ecos_item_code": "0089000",
        "note": "일별 종가. published_date = date. 2026-10-03 FDR(KQ11)에서 ECOS로 전환 "
                "(FDR GitHub 캐시 2026-09-17 이후 미갱신). 코드명은 피처 호환 때문에 유지",
    },
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
    """
    설명: d가 속한 달의 다음 달 말일을 돌려준다(월별 지표의 published_date).
    주의 사항: 12월이면 다음 해 1월 말일.
    입력 타입->반환 타입: (date) -> date
    """
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(year, month, monthrange(year, month)[1])


def fetch_ecos_series(stat_code, item_code, cycle, start, end):
    """
    설명: ECOS StatisticSearch를 PAGE_SIZE 단위로 나눠 [start, end] 전체 행(dict 목록)을 돌려준다.
    주의 사항: 결과 없음(INFO-200)은 빈 목록. 그 밖의 API 오류·형식 이상은 RuntimeError. 요청 예외는 API 키가 든 URL을 빼고 다시 던진다.
    입력 타입->반환 타입: (str, str, str, str, str) -> list[dict]
    """
    all_rows = []
    begin = 1
    while True:
        end_idx = begin + PAGE_SIZE - 1
        url = (f"{ECOS_BASE_URL}/{Config.ECOS_API_KEY}/json/kr/{begin}/{end_idx}/"
               f"{stat_code}/{cycle}/{start}/{end}/{item_code}")
        try:
            r = requests.get(url, timeout=30)
            r.encoding = "utf-8"
            data = r.json()
        except requests.RequestException as e:
            # requests 예외 문자열에는 요청 URL(API 키가 든 경로)이 들어간다.
            # 예외 클래스명과 요청 식별자만 남기고, 원 예외는 traceback에 붙이지 않는다(from None).
            raise RuntimeError(
                f"ECOS 요청 실패({type(e).__name__}): stat={stat_code} item={item_code} "
                f"cycle={cycle} {start}~{end} rows={begin}~{end_idx}"
            ) from None

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
    """
    설명: ECOS 행을 (code, date, value, published_date) 튜플 리스트로 바꾸고 값이 빈 행 수를 함께 돌려준다.
    주의 사항: 일별은 published_date = date, 월별은 date = 기준월 1일·published_date = 다음 달 말일. 그 밖의 주기는 ValueError.
    입력 타입->반환 타입: (str, str, list[dict]) -> (list[tuple], int)
    """
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
    """
    설명: ECOS 지표 하나를 적재하고 결과 dict를 돌려준다.
    주의 사항: start_override가 있으면 그 값부터, 없으면 DB 최신일 다음(일별 다음 날, 월별 다음 달)부터. 행이 없으면 1900년부터(전체 기간).
    최신일이 오늘이면 건너뛴다. 커밋은 호출자가 한다.
    입력 타입->반환 타입: (psycopg.Cursor, str, dict, str|None) -> dict
    """
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
    """
    설명: 메타를 먼저 upsert한 뒤 ECOS 지표 8종을 적재하고 {code: 결과 dict}를 돌려준다.
    주의 사항: ECOS_API_KEY가 없으면 전부 "스킵"으로 돌려주고 경고만(FDR 섹션은 계속). only_uninitialized=True면 행이 없는 지표만.
    지표별 예외는 격리하고, 메시지는 redact로 비밀값을 가린다.
    입력 타입->반환 타입: (psycopg.Cursor, bool) -> dict
    """
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
            print(f"❌ [ECOS] {code}: 적재 실패: {redact(e)}")
            results[code] = {"status": f"실패 ({redact(e)})", "rows_fetched": 0, "inserted": 0, "updated": 0}
    return results


def print_section_summary(section_name, results):
    """
    설명: 섹션(FDR·ECOS·강제 재수집)의 지표별 결과를 한 줄씩 출력하고 실패 지표는 경고를 덧붙인다.
    주의 사항: 출력만 한다(종료 코드는 호출자가 정한다).
    입력 타입->반환 타입: (str, dict) -> None
    """
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



# ── 일별 지표 신선도 점검 ──────────────────────────────────────────────
# lag = 지표 최신일 뒤 ~ daily_stock_prices 최신일까지의 거래일 수(휴장일은 세지 않음). 1 이상이면 경고, 한도 이상이면 실패.
# 월별 지표는 대상이 아니다. 한도 근거(기준금리만 4)는 결정기록.md "일별 지표 신선도 점검과 종료 코드".
DAILY_LAG_LIMITS = {
    "KOSPI": 2, "KOSDAQ": 2, "USD_KRW": 3,
    "ECOS_722Y001_0101000": 4, "ECOS_817Y002_010200000": 2, "ECOS_817Y002_010210000": 2,
}


def check_daily_indicator_lag(cur, limits=DAILY_LAG_LIMITS):
    """
    설명: limits의 지표마다 {last_date, ref_date, lag, limit, status("정상"/"경고"/"실패")}를 계산해 돌려준다.
    주의 사항: DB 읽기만. 기준일 = daily_stock_prices 최신일이라 주가 수집이 함께 멈추면 lag가 0으로 보인다. 지표 행이 없으면 실패.
    입력 타입->반환 타입: (psycopg.Cursor, dict) -> dict
    """
    cur.execute("SELECT MAX(date) FROM daily_stock_prices")
    ref_date = cur.fetchone()[0]
    results = {}
    for code, limit in limits.items():
        last = get_last_date(cur, code)
        if last is None or ref_date is None:
            results[code] = {"last_date": last, "ref_date": ref_date, "lag": None, "limit": limit, "status": "실패"}
            continue
        cur.execute("SELECT COUNT(DISTINCT date) FROM daily_stock_prices WHERE date > %s AND date <= %s",
                    (last, ref_date))
        lag = cur.fetchone()[0]
        status = "실패" if lag >= limit else ("경고" if lag >= 1 else "정상")
        results[code] = {"last_date": last, "ref_date": ref_date, "lag": lag, "limit": limit, "status": status}
    return results


# ── 기간 지정 강제 재수집 ──────────────────────────────────────────────
# 증분 수집은 최신일 다음부터만 받으므로, 이미 저장된 날짜의 소스 정정은 이 경로(CLI: 지표_일일수집 --force)로만 반영된다.
# 같은 upsert_market_indicators를 쓴다. 과거 값을 덮어쓰므로 먼저 --dry-run으로 확인한다.

def _fetch_indicator_range(indicator_code, start, end):
    """
    설명: 지표 하나를 소스에서 [start, end]로 다시 받아 적재용 튜플 리스트로 돌려준다.
    주의 사항: 범위로 조회한 뒤 날짜로 거른다(Yahoo 하루치 조회는 다른 날 봉을 돌려준다). FDR은 end를 컷오프로 자른다.
    알 수 없는 코드는 ValueError, ECOS 키가 없으면 RuntimeError.
    입력 타입->반환 타입: (str, date, date) -> list[tuple]
    """
    if indicator_code in FDR_INDICATOR_MAP:
        end = min(end, last_storable_date())  # --force도 확정 전 KST 오늘 행은 쓰지 않는다
        df = fdr.DataReader(FDR_INDICATOR_MAP[indicator_code], start.isoformat(), end.isoformat())
        if df.empty:
            return []
        df.index.name = "Date"
        df = df.reset_index().dropna(subset=["Close"])
        records = [(indicator_code, r["Date"].date(), round(float(r["Close"]), 4), r["Date"].date())
                   for _, r in df.iterrows()]
    elif indicator_code in ECOS_INDICATOR_META:
        if not Config.ECOS_API_KEY:
            raise RuntimeError("ECOS_API_KEY가 .env에 없음")
        meta = ECOS_INDICATOR_META[indicator_code]
        cycle = meta["frequency"]
        fmt = "%Y%m%d" if cycle == "D" else "%Y%m"
        rows = fetch_ecos_series(meta["ecos_stat_code"], meta["ecos_item_code"], cycle,
                                 start.strftime(fmt), end.strftime(fmt))
        records, _ = parse_ecos_records(indicator_code, cycle, rows or [])
    else:
        raise ValueError(f"알 수 없는 지표 코드: {indicator_code}")
    return [r for r in records if start <= r[1] <= end]


def force_refetch_indicator(cur, indicator_code, start, end, dry_run=False):
    """
    설명: [start, end]를 소스에서 다시 받아 날짜별 변경(old -> new)·신규를 출력하고, dry_run이 아니면 upsert한 뒤 결과 dict를 돌려준다.
    주의 사항: 소스에 없는 DB 행은 지우지 않고 경고만. 커밋은 호출자가 한다(dry_run이면 쓰지 않음).
    입력 타입->반환 타입: (psycopg.Cursor, str, date, date, bool) -> dict
    """
    records = _fetch_indicator_range(indicator_code, start, end)
    cur.execute(
        "SELECT date, value FROM market_indicators WHERE indicator_code = %s AND date BETWEEN %s AND %s",
        (indicator_code, start, end),
    )
    existing = {d: float(v) for d, v in cur.fetchall()}
    new_dates = [r for r in records if r[1] not in existing]
    changed = [r for r in records if r[1] in existing and abs(existing[r[1]] - r[2]) > 1e-9]
    source_missing = sorted(set(existing) - {r[1] for r in records})

    tag = "[dry-run] " if dry_run else ""
    for r in changed:
        print(f"  {tag}{indicator_code} {r[1]}: {existing[r[1]]} -> {r[2]} (정정)")
    for r in new_dates:
        print(f"  {tag}{indicator_code} {r[1]}: (없음) -> {r[2]} (신규)")
    if source_missing:
        print(f"  ⚠️ {indicator_code}: DB에는 있으나 소스에 없는 날짜 {len(source_missing)}건(삭제하지 않음): "
              f"{source_missing[:5]}{' ...' if len(source_missing) > 5 else ''}")

    result = {"status": "dry-run" if dry_run else "완료", "rows_fetched": len(records),
              "inserted": len(new_dates), "updated": len(changed)}
    if records:
        result["date_range"] = (records[0][1], records[-1][1])
    if not dry_run and records:
        inserted, updated = upsert_market_indicators(cur, records)
        result["inserted"], result["updated"] = inserted, updated
    return result
