# db_utils.py — indicator_meta·market_indicators에 쓰는 공용 함수. 지표_공통의 FDR·ECOS 로더가 함께 쓴다.
# 파이프라인 단계: 수집(저장). 테이블 사용 계약(메타 먼저 적재, published_date 명시, 값이 다를 때만 갱신)을 이 파일에서만 지킨다.
# 배경: 결정기록.md의 "개정값은 최신값만 유지하고 신규·개정 건수를 센다"


def upsert_indicator_meta(cur, meta_dict):
    """
    설명: {indicator_code: {name, source, frequency, unit?, ecos_stat_code?, ecos_item_code?, note?}}를 indicator_meta에 upsert한다.
    주의 사항: market_indicators 적재보다 먼저 불러야 한다(FK). 기존 메타는 모든 컬럼을 덮어쓴다. 커밋은 호출자가 한다.
    입력 타입->반환 타입: (psycopg.Cursor, dict) -> None
    """
    query = """
        INSERT INTO indicator_meta (indicator_code, name, source, frequency, unit, ecos_stat_code, ecos_item_code, note)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (indicator_code) DO UPDATE
            SET name = EXCLUDED.name,
                source = EXCLUDED.source,
                frequency = EXCLUDED.frequency,
                unit = EXCLUDED.unit,
                ecos_stat_code = EXCLUDED.ecos_stat_code,
                ecos_item_code = EXCLUDED.ecos_item_code,
                note = EXCLUDED.note
    """
    for code, meta in meta_dict.items():
        cur.execute(query, (
            code, meta["name"], meta["source"], meta["frequency"], meta.get("unit"),
            meta.get("ecos_stat_code"), meta.get("ecos_item_code"), meta.get("note"),
        ))


def get_last_date(cur, indicator_code):
    """
    설명: market_indicators에서 지표의 가장 최신 date를 돌려준다.
    주의 사항: 행이 없으면 None. date는 기준일이다(published_date 아님).
    입력 타입->반환 타입: (psycopg.Cursor, str) -> date|None
    """
    cur.execute("SELECT MAX(date) FROM market_indicators WHERE indicator_code = %s", (indicator_code,))
    return cur.fetchone()[0]


def upsert_market_indicators(cur, records):
    """
    설명: (indicator_code, date, value, published_date) 튜플을 upsert하고 (신규 건수, 개정 건수)를 돌려준다.
    주의 사항: 값이 같은 기존 행은 건드리지 않고 세지 않는다. 값이 다르면 value와 published_date를 함께 바꾼다. 커밋은 호출자가 한다.
    입력 타입->반환 타입: (psycopg.Cursor, list[tuple]) -> (int, int)
    """
    query = """
        INSERT INTO market_indicators (indicator_code, date, value, published_date)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (indicator_code, date) DO UPDATE
            SET value = EXCLUDED.value,
                published_date = EXCLUDED.published_date
            WHERE market_indicators.value IS DISTINCT FROM EXCLUDED.value
        RETURNING (xmax <> 0) AS was_update
    """
    inserted = 0
    updated = 0
    for rec in records:
        cur.execute(query, rec)
        result = cur.fetchone()
        if result is None:
            continue  # 값이 동일해 WHERE 절에 걸려 스킵된 기존 행
        if result[0]:
            updated += 1
        else:
            inserted += 1
    return inserted, updated