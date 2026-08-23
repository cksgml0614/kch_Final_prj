-- indicator_meta + market_indicators 테이블 생성. 2026-08-23 기준 실제 운영 DB 스키마를
-- information_schema/pg_constraint/pg_indexes로 직접 조회해 작성 — Database/migrations/002가
-- 반영된 최종 상태다. FK 때문에 indicator_meta(부모)를 market_indicators(자식)보다 먼저 만든다
-- — 적재 순서 계약(indicator_meta upsert → market_indicators insert)과 동일한 순서.

CREATE TABLE IF NOT EXISTS indicator_meta (
    indicator_code  VARCHAR(32) PRIMARY KEY,
    name            VARCHAR(128) NOT NULL,
    source          VARCHAR(32) NOT NULL,
    frequency       VARCHAR(8) NOT NULL,
    unit            VARCHAR(32),
    ecos_stat_code  VARCHAR(32),
    ecos_item_code  VARCHAR(32),
    note            TEXT,

    CONSTRAINT ck_indicator_meta_source CHECK (source IN ('FDR', 'ECOS')),
    CONSTRAINT ck_indicator_meta_frequency CHECK (frequency IN ('D', 'W', 'M', 'Q'))
);

CREATE TABLE IF NOT EXISTS market_indicators (
    indicator_code  VARCHAR(32) NOT NULL,
    date            DATE NOT NULL,
    value           NUMERIC,
    unit            VARCHAR(20),   -- ⚠️ DEPRECATED. indicator_meta.unit 사용, 여기는 항상 NULL로 둘 것
    published_date  DATE NOT NULL, -- DEFAULT 없음 — 코드에서 명시 대입 필수 (누락 시 INSERT 즉시 실패)

    PRIMARY KEY (indicator_code, date),
    CONSTRAINT fk_market_indicators_meta FOREIGN KEY (indicator_code)
        REFERENCES indicator_meta (indicator_code)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_indicators_date ON market_indicators (date);
CREATE INDEX IF NOT EXISTS idx_indicators_code_published ON market_indicators (indicator_code, published_date DESC);
