# db_utils.py
# market_indicators / indicator_meta 적재 공통 유틸. G-2(FDR)·G-3(ECOS) 로더가 공유한다.
# G-1 계약(적재 순서, published_date 명시, IS DISTINCT FROM upsert)을 한 곳에서 지킨다.


def upsert_indicator_meta(cur, meta_dict):
    """meta_dict: {indicator_code: {name, source, frequency, unit, note, ecos_stat_code?, ecos_item_code?}}
    market_indicators 적재보다 먼저 호출해야 한다 (FK)."""
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
    """DB에서 해당 지표의 가장 최신 date를 가져온다."""
    cur.execute("SELECT MAX(date) FROM market_indicators WHERE indicator_code = %s", (indicator_code,))
    return cur.fetchone()[0]


def upsert_market_indicators(cur, records):
    """records: (indicator_code, date, value, published_date) 튜플 리스트.
    G-1 계약: IS DISTINCT FROM + RETURNING으로 신규/개정 건수를 구분한다."""
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