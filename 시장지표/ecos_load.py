# ecos_load.py — Task G-3
# 한국은행 ECOS Open API로 거시지표 6종을 수집해 market_indicators + indicator_meta에 적재한다.
# 지표 코드는 G-3-1에서 StatisticTableList/StatisticItemList로 실측 조회해 확정했다
# (TASK_G_market_indicators_적재.md G-3 참고). 반도체 수출금액지수는 2026-08-16 사람 판단으로 제외.

import sys
from calendar import monthrange
from datetime import date, timedelta

import requests

from config import Config
from db_manager import get_db_connection
from 시장지표.db_utils import get_last_date, upsert_indicator_meta, upsert_market_indicators

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/특수문자를 인코딩하지 못해 죽는다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
# 실측: 국고채(3년) 일별 전체 6,878건이 단일 요청으로 반환됨. 여유 있게 잡고 초과 시 페이지 분할.
PAGE_SIZE = 10000

INDICATOR_META = {
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
    """월별 지표 공표일 보수적 추정: 기준월 다음달 말일."""
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(year, month, monthrange(year, month)[1])


def fetch_ecos_series(stat_code, item_code, cycle, start, end):
    """ECOS StatisticSearch를 페이지 분할해 전체 구간을 조회한다.
    DB에 쓰지 않는 순수 조회 함수 — 체크포인트 확인용으로 단독 호출 가능."""
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
            if code == "INFO-200":  # 해당 조건에 데이터 없음 — 오류 아님
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


def parse_records(indicator_code, cycle, rows):
    """ECOS row -> (indicator_code, date, value, published_date) 튜플 리스트로 변환.
    DATA_VALUE 결측(빈 문자열)은 건너뛴다."""
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


def update_indicator(cur, indicator_code, meta):
    """지표 1건 증분 적재. 종목 순회 실패 격리를 위해 예외를 밖으로 던진다 (호출부에서 격리)."""
    cycle = meta["frequency"]
    last_date = get_last_date(cur, indicator_code)

    if last_date == date.today():
        return {"status": "스킵 (이미 최신)", "rows_fetched": 0, "inserted": 0, "updated": 0}

    if cycle == "D":
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

    records, skipped_missing = parse_records(indicator_code, cycle, rows)
    if not records:
        return {"status": "스킵 (전량 결측)", "rows_fetched": len(rows), "inserted": 0, "updated": 0}

    inserted, updated = upsert_market_indicators(cur, records)
    result = {
        "status": "완료",
        "rows_fetched": len(records),
        "inserted": inserted,
        "updated": updated,
        "date_range": (records[0][1], records[-1][1]),
    }
    if skipped_missing:
        result["skipped_missing"] = skipped_missing
    return result


if __name__ == "__main__":
    if not Config.ECOS_API_KEY:
        raise SystemExit("ECOS_API_KEY가 .env에 없습니다")

    with get_db_connection() as conn:
        if not conn:
            raise SystemExit("DB 연결 실패")

        with conn.cursor() as cur:
            upsert_indicator_meta(cur, INDICATOR_META)

            results = {}
            for code, meta in INDICATOR_META.items():
                try:
                    results[code] = update_indicator(cur, code, meta)
                except Exception as e:
                    print(f"❌ {code}: 적재 실패: {e}")
                    results[code] = {"status": f"실패 ({e})", "rows_fetched": 0, "inserted": 0, "updated": 0}

        conn.commit()

        print("\n=== 처리 결과 요약 ===")
        for code, r in results.items():
            line = f"{code}: {r['status']} | fetched={r['rows_fetched']} inserted={r['inserted']} updated={r['updated']}"
            if "date_range" in r:
                line += f" | range={r['date_range']}"
            if "skipped_missing" in r:
                line += f" | 결측 스킵={r['skipped_missing']}"
            print(line)
            if r["status"].startswith("실패"):
                print(f"  ⚠️ {code}: 적재 실패 — 확인 필요")
