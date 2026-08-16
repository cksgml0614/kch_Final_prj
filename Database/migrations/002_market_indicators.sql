-- =============================================================================
-- 002_market_indicators.sql  (Task G-1)
--
-- market_indicators long format 확정 + indicator_meta 신설.
-- 001은 뉴스 스키마(TASK_개정판_데이터_재구축.md Task B)용으로 예약되어 있어 002를 사용한다.
--
-- 적용 시점 기준 상태: market_indicators 0행 (적재 이력 없음)
--   → NOT NULL 컬럼 추가와 타입 변경을 backfill 없이 안전하게 수행할 수 있다.
--     행이 존재하는 DB에서 이 파일을 돌리면 published_at SET NOT NULL 단계에서
--     실패하는데, 이는 의도된 동작이다(조용히 NULL을 허용하지 않는다).
--
-- 실행: psql -f Database/migrations/002_market_indicators.sql
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. indicator_meta — 지표 메타 정보 (FK 대상이므로 먼저 생성)
-- -----------------------------------------------------------------------------
-- indicator_code 명명 규칙 (확정):
--   ECOS : 'ECOS_<통계표코드>_<항목코드>'   예) ECOS_722Y001_0101000
--   FDR  : 심볼 그대로                      예) KS11, KQ11
CREATE TABLE IF NOT EXISTS indicator_meta (
    indicator_code VARCHAR(32) PRIMARY KEY,
    name           VARCHAR(128) NOT NULL,
    source         VARCHAR(32)  NOT NULL,  -- 'FDR' | 'ECOS'
    frequency      VARCHAR(8)   NOT NULL,  -- 'D' | 'W' | 'M' | 'Q'
    unit           VARCHAR(32),            -- 단위의 유일한 출처(single source of truth)
    ecos_stat_code VARCHAR(32),            -- ECOS 통계표코드 (FDR 지표는 NULL)
    ecos_item_code VARCHAR(32),            -- ECOS 항목코드   (FDR 지표는 NULL)
    note           TEXT,
    CONSTRAINT ck_indicator_meta_source    CHECK (source    IN ('FDR', 'ECOS')),
    CONSTRAINT ck_indicator_meta_frequency CHECK (frequency IN ('D', 'W', 'M', 'Q'))
);

COMMENT ON TABLE  indicator_meta IS
    '지표 메타 정보. market_indicators.indicator_code의 FK 대상이므로, 지표 적재 전에 반드시 여기에 행이 먼저 존재해야 한다.';
COMMENT ON COLUMN indicator_meta.indicator_code IS
    '명명 규칙 — ECOS: ECOS_<통계표코드>_<항목코드> / FDR: 심볼 그대로(KS11 등)';
COMMENT ON COLUMN indicator_meta.unit IS
    '단위의 유일한 출처. market_indicators.unit은 사용하지 않는다.';
COMMENT ON COLUMN indicator_meta.note IS
    '지표별 비고. 개정(revision)되는 지표는 "개정 이력 미보존 — 최신값만 유지" 를 반드시 명시한다.';

-- -----------------------------------------------------------------------------
-- 2. market_indicators — 컬럼 정합화
-- -----------------------------------------------------------------------------

-- 2-1. indicator_code 길이를 indicator_meta와 통일 (VARCHAR(20) -> VARCHAR(32))
--      PK 컬럼이라 인덱스가 재구축되지만, 확장(widening)이라 데이터 손실은 없다.
ALTER TABLE market_indicators
    ALTER COLUMN indicator_code TYPE VARCHAR(32);

-- 2-2. published_date 추가 — 미래 정보 누수 방지의 핵심 컬럼
--      DEFAULT를 두지 않는다. 일별 지표라도 로더 코드에서 published_date = date 를
--      명시적으로 대입해야 하며, 기입 누락은 INSERT 시점에 즉시 실패시킨다.
ALTER TABLE market_indicators
    ADD COLUMN IF NOT EXISTS published_date DATE;

ALTER TABLE market_indicators
    ALTER COLUMN published_date SET NOT NULL;

-- 2-3. FK — 메타 정보 없는 지표 코드의 적재를 차단
--      (ADD CONSTRAINT는 IF NOT EXISTS를 지원하지 않아 DO 블록으로 멱등 처리)
--
--      ⚠️ ON DELETE RESTRICT이므로 지표 폐기 시 삭제 순서가 강제된다.
--         indicator_meta 행을 먼저 지우려 하면 FK 위반으로 거부되므로 반드시 아래 순서를 지킬 것:
--           1) DELETE FROM market_indicators WHERE indicator_code = '<코드>';
--           2) DELETE FROM indicator_meta    WHERE indicator_code = '<코드>';
--         (CASCADE로 두지 않은 것은 의도적이다. 메타 한 줄 삭제로 수천 건의
--          지표 데이터가 조용히 사라지는 사고를 막는다.)
--         지표 코드 '변경'은 ON UPDATE CASCADE가 걸려 있으므로
--         indicator_meta.indicator_code만 UPDATE하면 자식 행이 따라 바뀐다.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_market_indicators_meta'
    ) THEN
        ALTER TABLE market_indicators
            ADD CONSTRAINT fk_market_indicators_meta
            FOREIGN KEY (indicator_code)
            REFERENCES indicator_meta (indicator_code)
            ON UPDATE CASCADE
            ON DELETE RESTRICT;
    END IF;
END
$$;

-- 2-4. G-4 피처 조회용 인덱스
--      "예측일 D에 대해 published_date < D 인 값 중 가장 최근" 조회 패턴을 지원한다.
CREATE INDEX IF NOT EXISTS idx_indicators_code_published
    ON market_indicators (indicator_code, published_date DESC);

-- -----------------------------------------------------------------------------
-- 3. 컬럼 의미 주석 — 설계 결정을 스키마 자체에 기록
-- -----------------------------------------------------------------------------
COMMENT ON TABLE  market_indicators IS
    'long format 시장/거시지표. PK가 (indicator_code, date)이므로 개정치는 최신값으로 덮어쓰며 이력을 보존하지 않는다(ON CONFLICT DO UPDATE). 로더는 값이 실제로 변경된 건수를 반드시 로그로 출력한다.';
COMMENT ON COLUMN market_indicators.date IS
    '지표의 기준일(reference date). 월별 지표는 해당 월 1일로 통일한다.';
COMMENT ON COLUMN market_indicators.published_date IS
    '해당 값이 실제로 공표된 날. 일별 지표는 date와 동일. 월별 지표는 실제 공표일을 넣되 확인이 어려우면 보수적으로 기준월 다음달 말일을 쓰고 indicator_meta.note에 그 사실을 기록한다. 학습 피처 조인은 반드시 published_date < 예측일 조건을 건다.';
COMMENT ON COLUMN market_indicators.value IS
    '지표값. 등락률 등 파생값은 저장하지 않고 조회 시 LAG(value)로 계산한다.';
COMMENT ON COLUMN market_indicators.unit IS
    'DEPRECATED — 사용하지 않는다. 단위는 indicator_meta.unit을 참조할 것. (기존 정의 호환을 위해 컬럼만 남겨둠, 항상 NULL)';

COMMIT;

-- =============================================================================
-- 적용 후 기대 스키마
--
-- indicator_meta
--   indicator_code VARCHAR(32) PK
--   name VARCHAR(128) NOT NULL, source VARCHAR(32) NOT NULL CHECK(FDR|ECOS),
--   frequency VARCHAR(8) NOT NULL CHECK(D|W|M|Q), unit VARCHAR(32),
--   ecos_stat_code VARCHAR(32), ecos_item_code VARCHAR(32), note TEXT
--
-- market_indicators
--   indicator_code VARCHAR(32) NOT NULL  -> FK indicator_meta(indicator_code)
--   date           DATE        NOT NULL
--   value          NUMERIC
--   unit           VARCHAR(20)  (DEPRECATED, 항상 NULL)
--   published_date DATE        NOT NULL
--   PK (indicator_code, date)
--   INDEX idx_indicators_date, idx_indicators_code_published
--
-- 로더가 지켜야 할 계약 (G-2 / G-3 구현 시)
--   1) 지표 적재 전 indicator_meta upsert 선행 (FK 위반 방지)
--   2) published_date 명시 대입 (일별: = date)
--   3) INSERT ... ON CONFLICT (indicator_code, date) DO UPDATE
--        SET value = EXCLUDED.value, published_date = EXCLUDED.published_date
--        WHERE market_indicators.value IS DISTINCT FROM EXCLUDED.value
--      -> WHERE 절 덕분에 값이 실제로 바뀐 행만 UPDATE되고, RETURNING으로 개정 건수를 셀 수 있다
--   4) 지표별 성공/실패/적재 건수 요약 출력, 0건 지표는 경고
--   5) 지표 폐기 시 삭제 순서: market_indicators 먼저, indicator_meta 나중
--      (ON DELETE RESTRICT — 위 2-3 주석 참고)
-- =============================================================================